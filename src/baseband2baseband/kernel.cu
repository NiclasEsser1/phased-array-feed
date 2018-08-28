#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "kernel.cuh"
#include "baseband2baseband.cuh"
#include "cudautil.cuh"

/*
  This kernel is used to :
  1. unpack the incoming data reading from ring buffer and reorder the order from TFTFP to PFT;
*/
__global__ void unpack_kernel(int64_t *dbuf_in,  cufftComplex *dbuf_rt1, size_t offset_rt1)
{
  size_t loc_in, loc_rt1;
  int64_t tmp;
  
  /* 
     Loc for the input array, it is in continuous order, it is in (STREAM_BUF_NDFSTP)T(NCHK_NIC)F(NSAMP_DF)T(NCHAN_CHK)F(NPOL_SAMP)P order
     This is for entire setting, since gridDim.z =1 and blockDim.z = 1, we can simply it to the latter format;
     Becareful here, if these number are not 1, we need to use a different format;
   */
  //loc_in = blockIdx.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z +
  //  blockIdx.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z +
  //  blockIdx.z * blockDim.x * blockDim.y * blockDim.z +
  //  threadIdx.x * blockDim.y * blockDim.z +
  //  threadIdx.y * blockDim.z +
  //  threadIdx.z;
  loc_in = blockIdx.x * gridDim.y * blockDim.x * blockDim.y +
    blockIdx.y * blockDim.x * blockDim.y +
    threadIdx.x * blockDim.y +
    threadIdx.y;
  tmp = BSWAP_64(dbuf_in[loc_in]);
  
  // Put the data into PFT order  
  loc_rt1 = blockIdx.y * gridDim.x * blockDim.x * blockDim.y +
    threadIdx.y * gridDim.x * blockDim.x +
    blockIdx.x * blockDim.x +
    threadIdx.x;
  
  dbuf_rt1[loc_rt1].x = (int16_t)((tmp & 0x000000000000ffffULL));  
  dbuf_rt1[loc_rt1].y = (int16_t)((tmp & 0x00000000ffff0000ULL) >> 16);
  
  loc_rt1 = loc_rt1 + offset_rt1;
  dbuf_rt1[loc_rt1].x = (int16_t)((tmp & 0x0000ffff00000000ULL) >> 32);
  dbuf_rt1[loc_rt1].y = (int16_t)((tmp & 0xffff000000000000ULL) >> 48);
}

/* 
   This is a combination of swap_select_transpose_kernel and swap_kernel, it is is used to :
   1. swap the halves of CUDA FFT output, we need to do that because CUDA FFT put the centre frequency at bin 0;
   2. drop the first three and last two points of the swapped 32-points FFT output, which will reduce to oversample rate to 1;
      for 64 points FFT, drop the first and last five points;
   3. drop the edge of passband to give a good number for reverse FFT;
   4. reorder the FFT data from PFTF to PTF;
   5. swap the halves to make sure that the centre frequency in each reverse FFT segment is in bin 0;
   6. we can also easily do de-dispersion here, which is not here yet;
   7. if we do de-dispersion here, we need to use loc1 to locate chirp, not the loc2 because loc2 is the swapped index.
*/
__global__ void swap_select_transpose_swap_kernel(cufftComplex *dbuf_rt1, cufftComplex *dbuf_rt2, size_t offset_rt1, size_t offset_rt2)
{
  int remainder1, remainder2, loc1, loc2;
  size_t loc_rt1, loc_rt2;
  cufftComplex p1, p2;

  remainder1 = (threadIdx.x + CUFFT_MOD1)%CUFFT_NX1;
  if(remainder1 < NCHAN_KEEP_CHAN)
    {
      loc1 = blockIdx.x * NCHAN_KEEP_CHAN + remainder1 - NCHAN_EDGE;
      if((loc1 >= 0) && (loc1 < NCHAN_KEEP_BAND))
	{
	  loc_rt1 = blockIdx.x * gridDim.y * blockDim.x +
	    blockIdx.y * blockDim.x +
	    threadIdx.x;

	  remainder2 = (loc1 + CUFFT_MOD2)%CUFFT_NX2;
	  loc2 = remainder2 + loc1 - loc1%CUFFT_NX2;
	  
	  loc_rt2 = blockIdx.y * NCHAN_KEEP_BAND + loc2;  
	  
	  p1 = dbuf_rt1[loc_rt1];
	  dbuf_rt2[loc_rt2].x = p1.x;
	  dbuf_rt2[loc_rt2].y = p1.y;

	  loc_rt2 = loc_rt2 + offset_rt2;

	  p2 = dbuf_rt1[loc_rt1 + offset_rt1];
	  dbuf_rt2[loc_rt2].x = p2.x;
	  dbuf_rt2[loc_rt2].y = p2.y;
	}
    }
}

/* 
   This kernel will make the scale calculation of fold mode easier:
   1. reorder the second FFT data from PTFT order to FTP;
   2. pad the dbuf_rt1.x with the sum of dbuf_rt2.x + dbuf_rt2.y (after transpose);
   3. pad the dbuf_rt1.y with the sum of dbuf_rt2.x ** 2 + dbuf_rt2.y ** 2 (after transpose);
*/
__global__ void transpose_pad_kernel(cufftComplex *dbuf_rt2, size_t offset_rt2, cufftComplex *dbuf_rt1)
{
  size_t loc_rt2, loc_rt1;
  float x, y;
  cufftComplex p1, p2;
  
  loc_rt2 = blockIdx.x * gridDim.y * blockDim.x +
    blockIdx.y * blockDim.x +
    threadIdx.x;
  
  loc_rt1 = NPOL_SAMP * (blockIdx.y * gridDim.x * blockDim.x +
			 blockIdx.x * blockDim.x +
			 threadIdx.x);
  
  p1 = dbuf_rt2[loc_rt2];
  x  = p1.x;
  y  = p1.y;
  dbuf_rt1[loc_rt1].x         = x + y;
  dbuf_rt1[loc_rt1].y         = x * x + y * y;

  p2 = dbuf_rt2[loc_rt2 + offset_rt2];
  x  = p2.x;
  y  = p2.y;  
  dbuf_rt1[loc_rt1 + 1].x     = x + y;
  dbuf_rt1[loc_rt1 + 1].y     = x * x + y * y;
}

/*
  This kernel will get the sum of all elements in dbuf_rt1, which is the buffer for each stream
 */
__global__ void sum_kernel(cufftComplex *dbuf_rt1, cufftComplex *dbuf_rt2)
{
  extern __shared__ cufftComplex sum_sdata[];
  size_t tid, loc, s;
  
  tid = threadIdx.x;
  loc = blockIdx.x * gridDim.y * (blockDim.x * 2) +
    blockIdx.y * (blockDim.x * 2) +
    threadIdx.x;
  sum_sdata[tid].x = dbuf_rt1[loc].x + dbuf_rt1[loc + blockDim.x].x; 
  sum_sdata[tid].y = dbuf_rt1[loc].y + dbuf_rt1[loc + blockDim.x].y;
  __syncthreads();

  /* do reduction in shared mem */
  for (s=blockDim.x/2; s>0; s>>=1)
    {
      if (tid < s)
  	{
  	  sum_sdata[tid].x += sum_sdata[tid + s].x;
  	  sum_sdata[tid].y += sum_sdata[tid + s].y;
  	}
      __syncthreads();
    }

  /* write result of this block to global mem */
  if (tid == 0)
    dbuf_rt2[blockIdx.x * gridDim.y + blockIdx.y] = sum_sdata[0];
}

/*
  This kernel calculate the mean of (samples and square of samples, which are padded in buf_rt1 for fold mode, or buf_rt2 for search mode). 
 */
__global__ void mean_kernel(cufftComplex *buf_rt1, size_t offset_rt1, float *ddat_offs, float *dsquare_mean, int nstream, float scl_ndim)
{
  int i;
  size_t loc_freq, loc;
  float dat_offs = 0, square_mean = 0;
  
  loc_freq = threadIdx.x;
  
  for (i = 0; i < nstream; i++)
    {
      loc = loc_freq + i * offset_rt1;
      dat_offs    += (buf_rt1[loc].x / scl_ndim);
      square_mean += (buf_rt1[loc].y / scl_ndim);
    }
  
  ddat_offs[loc_freq]    += dat_offs;
  dsquare_mean[loc_freq] += square_mean;
}

/*
  This kernel is used to calculate the scale of data based on the mean calculate by mean_kernel
*/
__global__ void scale_kernel(float *ddat_offs, float *dsquare_mean, float *ddat_scl)
{
  size_t loc_freq = threadIdx.x;
  ddat_scl[loc_freq] = SCL_NSIG * sqrtf(dsquare_mean[loc_freq] - ddat_offs[loc_freq] * ddat_offs[loc_freq]) / SCL_INT8;
}

/* 
   This is the speedup version with dat_scl and dat_offs calculated from data
*/
__global__ void transpose_scale_kernel(cufftComplex *dbuf_rt2, int8_t *dbuf_out_fold, size_t offset_rt2, float *ddat_offs, float *ddat_scl)
{
  // For 27 seconds data, 64-point FFT option needs around 720 ms
  // 32-point FFT option needs around 520ms
  __shared__ int8_t tile[NPOL_SAMP * NDIM_POL][TILE_DIM][TILE_DIM];

  int i, x, y;
  size_t loc, loc_rt2, loc_out;
  cufftComplex p1, p2;

  x = threadIdx.x;
  loc = blockIdx.x * gridDim.y * blockDim.x * TILE_DIM +
    blockIdx.y * blockDim.x * TILE_DIM +
    x;

  for (i = 0; i < TILE_DIM; i += NROWBLOCK_TRANS)
    {
      y = threadIdx.y + i;
      loc_rt2 = loc + y * blockDim.x;
	
      p1 = dbuf_rt2[loc_rt2];
      p2 = dbuf_rt2[loc_rt2 + offset_rt2];

      tile[0][y][x] = __float2int_rz(p1.x) >> SCALE;
      tile[1][y][x] = __float2int_rz(p1.y) >> SCALE;
      tile[2][y][x] = __float2int_rz(p2.x) >> SCALE;
      tile[3][y][x] = __float2int_rz(p2.y) >> SCALE;
    }

  __syncthreads(); // sync all threads in the same block;
  
  //loc = blockIdx.x * gridDim.y * blockDim.x * blockDim.y * TILE_DIM / NROWBLOCK_TRANS +
  loc = blockIdx.x * gridDim.y * blockDim.x * TILE_DIM +
    blockIdx.y * blockDim.x +
    x;
    
  for (i = 0; i < TILE_DIM; i += NROWBLOCK_TRANS)
    {
      y = threadIdx.y + i;
      loc_out = (loc + y * gridDim.y * blockDim.x) * NPOL_SAMP * NDIM_POL;
      
      dbuf_out_fold[loc_out]     = tile[0][x][y];
      dbuf_out_fold[loc_out + 1] = tile[1][x][y];
      dbuf_out_fold[loc_out + 2] = tile[2][x][y];
      dbuf_out_fold[loc_out + 3] = tile[3][x][y];
    }
}

///* 
//   This kernel can be used to record float data without scaling;
//*/
//__global__ void transpose_float_kernel(cufftComplex *dbuf_rt2, float *dbuf_out_fold, size_t offset_rt2)
//{
//  __shared__ float tile[NPOL_SAMP * NDIM_POL][TILE_DIM][TILE_DIM];
//
//  int i, x, y;
//  size_t loc, loc_rt2, loc_out;
//  cufftComplex p1, p2;
//
//  x = threadIdx.x;
//  loc = blockIdx.x * gridDim.y * blockDim.x * TILE_DIM +
//    blockIdx.y * blockDim.x * TILE_DIM +
//    x;
//
//  for (i = 0; i < TILE_DIM; i += NROWBLOCK_TRANS)
//    {
//      y = threadIdx.y + i;
//      loc_rt2 = loc + y * blockDim.x;
//      
//      p1 = dbuf_rt2[loc_rt2];
//      p2 = dbuf_rt2[loc_rt2 + offset_rt2];
//
//      tile[0][y][x] = p1.x;
//      tile[1][y][x] = p1.y;
//      tile[2][y][x] = p2.x;
//      tile[3][y][x] = p2.y;
//    }
//
//  __syncthreads(); // sync all threads in the same block;
//  
//  loc = blockIdx.x * gridDim.y * blockDim.x * blockDim.y * TILE_DIM / NROWBLOCK_TRANS +
//    blockIdx.y * blockDim.x +
//    x;
//    
//  for (i = 0; i < TILE_DIM; i += NROWBLOCK_TRANS)
//    {
//      y = threadIdx.y + i;
//      loc_out = (loc + y * gridDim.y * blockDim.x) * NPOL_SAMP * NDIM_POL;
//      
//      dbuf_out_fold[loc_out]     = tile[0][x][y];
//      dbuf_out_fold[loc_out + 1] = tile[1][x][y];
//      dbuf_out_fold[loc_out + 2] = tile[2][x][y];
//      dbuf_out_fold[loc_out + 3] = tile[3][x][y];
//    }
//}
