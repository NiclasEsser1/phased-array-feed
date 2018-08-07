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
   This kernel is used to :
   1. swap the halves of CUDA FFT output, we need to do that because CUDA FFT put the centre frequency at bin 0;
   2. drop the first three and last two points of the swapped 32-points FFT output, which will reduce to oversample rate to 1;
      for 64 points FFT, drop the first and last five points;
   3. drop the edge of passband to give a good number for reverse FFT;
   4. reorder the FFT data from PFTF to PTF;
   5. we can also easily do de-dispersion here, which is not here yet.
*/
__global__ void swap_select_transpose_kernel(cufftComplex *dbuf_rt1, cufftComplex *dbuf_rt, size_t offset_rt1, size_t offset_rt)
{
  int mod, loc;
  size_t loc_rt1, loc_rt;
  cufftComplex p1, p2;

  mod = (threadIdx.x + CUFFT_MOD1)%CUFFT_NX1;	
  if(mod < NCHAN_KEEP1)
    {
      loc = blockIdx.x * NCHAN_KEEP1 + mod - NCHAN_EDGE;
      if((loc >= 0) && (loc < NCHAN_KEEP2))
	{
	  loc_rt1 = blockIdx.x * gridDim.y * blockDim.x +
	    blockIdx.y * blockDim.x +
	    threadIdx.x;

	  loc_rt = blockIdx.y * NCHAN_KEEP2 + loc;  

	  p1 = dbuf_rt1[loc_rt1];
	  dbuf_rt[loc_rt].x = p1.x;
	  dbuf_rt[loc_rt].y = p1.y;

	  loc_rt = loc_rt + offset_rt;

	  p2 = dbuf_rt1[loc_rt1 + offset_rt1];
	  dbuf_rt[loc_rt].x = p2.x;
	  dbuf_rt[loc_rt].y = p2.y;
	}
    }
}

/* 
   The kernel is used to :
   1. swap the data to make sure that the centre frequency in each reverse FFT segment is in bin 0;
 */
__global__ void swap_kernel(cufftComplex *dbuf_rt, cufftComplex *dbuf_rt2, size_t offset)
{
  size_t loc_rt2, loc_rt, loc;

  loc     = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x;
  loc_rt  = loc + threadIdx.x;
  loc_rt2 = loc + (threadIdx.x + CUFFT_MOD2) % CUFFT_NX2;
  
  dbuf_rt2[loc_rt2]          = dbuf_rt[loc_rt];
  dbuf_rt2[loc_rt2 + offset] = dbuf_rt[loc_rt + offset];
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
  if(remainder1 < NCHAN_KEEP1)
    {
      loc1 = blockIdx.x * NCHAN_KEEP1 + remainder1 - NCHAN_EDGE;
      if((loc1 >= 0) && (loc1 < NCHAN_KEEP2))
	{
	  loc_rt1 = blockIdx.x * gridDim.y * blockDim.x +
	    blockIdx.y * blockDim.x +
	    threadIdx.x;

	  remainder2 = (loc1 + CUFFT_MOD2)%CUFFT_NX2;
	  //loc2 = remainder2 + CUFFT_NX2 * (int)(loc1 / CUFFT_NX2);
	  //loc2 = remainder2 + CUFFT_NX2 * (int)(loc1 / CUFFT_NX2);
	  //loc2 = remainder2 + CUFFT_NX2 * (int)(loc1 / NCHAN_KEEP1);
	  //loc2 = remainder2 + CUFFT_NX2 * (int)(loc1 / CUFFT_NX2);
	  loc2 = remainder2 + loc1 - loc1 % CUFFT_NX2;
	  
	  loc_rt2 = blockIdx.y * NCHAN_KEEP2 + loc2;  
	  
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
  #ifdef FOLD_MODE
    ddat_scl[loc_freq] = SCL_NSIG * sqrtf(dsquare_mean[loc_freq] - ddat_offs[loc_freq] * ddat_offs[loc_freq]) / SCL_INT8;
  #else
    ddat_scl[loc_freq] = SCL_NSIG * sqrtf(dsquare_mean[loc_freq] - ddat_offs[loc_freq] * ddat_offs[loc_freq]) / SCL_UINT8;
  #endif
}

/*
  The kernle here is used to :
  1. reorder data from PTFT to TFP;
  2. scale it to required range
*/
__global__ void transpose_scale_kernel(cufftComplex *dbuf_rt2, int8_t *dbuf_out_fold, size_t offset_rt2, int scale)
{
  size_t loc_rt2, loc_out, loc;
  cufftComplex p1, p2;

  loc = blockIdx.x * gridDim.y * blockDim.x ;
  loc_rt2 = loc + blockIdx.y * blockDim.x + threadIdx.x;
  loc_out = (loc + threadIdx.x * gridDim.y + blockIdx.y) * NPOL_SAMP * NDIM_POL;;
  
  p1 = dbuf_rt2[loc_rt2];
  p2 = dbuf_rt2[loc_rt2 + offset_rt2];

  dbuf_out_fold[loc_out]     = __float2int_rz(p1.x) >> scale;
  dbuf_out_fold[loc_out + 1] = __float2int_rz(p1.y) >> scale;
  dbuf_out_fold[loc_out + 2] = __float2int_rz(p2.x) >> scale;
  dbuf_out_fold[loc_out + 3] = __float2int_rz(p2.y) >> scale;
}

///*
//  This is a speedup version of previous kernel. It has better profermance with long time series. It uses share memory to get the speedup.
//  However, the speedup here only works with input array.
//*/
//__global__ void transpose_scale_kernel2(cufftComplex *dbuf_rt2, int8_t *dbuf_out_fold, size_t offset_rt2, int scale)
//{
//  __shared__ int8_t tile[NPOL_SAMP * NDIM_POL][NCHAN_FOLD][CUFFT_NX2]; // Here we have bank conflict problem, need more time to fix it.
//
//  int i, x, y;
//  size_t loc, loc_rt2, loc_out;
//  cufftComplex p1, p2;
//
//  x = threadIdx.x;
//  loc = blockIdx.x * blockDim.x * blockDim.y * NCHAN_FOLD / CUFFT_NX2;
//  
//  for (i = 0; i < NCHAN_FOLD; i += CUFFT_NX2)
//    {
//      y = threadIdx.y + i;
//      
//      loc_rt2 = loc + y * blockDim.x  + x;
//     
//      p1 = dbuf_rt2[loc_rt2];
//      p2 = dbuf_rt2[loc_rt2 + offset_rt2];
//
//      
//      tile[0][y][x] = __float2int_rz(p1.x) >> scale;
//      tile[1][y][x] = __float2int_rz(p1.y) >> scale;
//      tile[2][y][x] = __float2int_rz(p2.x) >> scale;
//      tile[3][y][x] = __float2int_rz(p2.y) >> scale;
//    }
//
//  __syncthreads(); // sync all threads in the same block;
//
//  for (i = 0; i < NCHAN_FOLD; i += CUFFT_NX2)
//    {
//      y = threadIdx.y + i;
//
//      loc_out = (loc + x * NCHAN_FOLD  + y) * NPOL_SAMP * NDIM_POL;
//      dbuf_out_fold[loc_out]     = tile[0][y][x];
//      dbuf_out_fold[loc_out + 1] = tile[1][y][x];
//      dbuf_out_fold[loc_out + 2] = tile[2][y][x];
//      dbuf_out_fold[loc_out + 3] = tile[3][y][x];
//    }
//}

///*
//  This is a speedup version of previous kernel. 
//  The speedup works with input and output array.
//  There is a bank conflict problem, which cause the speed low than best.
//  Need more time to fix it and get the best speed.
//  Leave it for now
//*/
//__global__ void transpose_scale_kernel3(cufftComplex *dbuf_rt2, int8_t *dbuf_out_fold, size_t offset_rt2, int scale)
//{
//  __shared__ int8_t tile[NPOL_SAMP * NDIM_POL][TILE_DIM][TILE_DIM];
//
//  int i, x, y;
//  size_t loc, loc_rt2, loc_out;
//  cufftComplex p1, p2;
//
//  x = threadIdx.x;
//  loc = (blockIdx.x * gridDim.y * blockDim.x * blockDim.y +
//	 blockIdx.y * blockDim.x * blockDim.y) * TILE_DIM / NROWBLOCK_TRANS +
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
//      tile[0][y][x] = __float2int_rz(p1.x) >> scale;
//      tile[1][y][x] = __float2int_rz(p1.y) >> scale;
//      tile[2][y][x] = __float2int_rz(p2.x) >> scale;
//      tile[3][y][x] = __float2int_rz(p2.y) >> scale;
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

/* 
   This is the speedup version with dat_scl and dat_offs calculated from data
*/
__global__ void transpose_scale_kernel4(cufftComplex *dbuf_rt2, int8_t *dbuf_out_fold, size_t offset_rt2, float *ddat_offs, float *ddat_scl)
{
  // For 27 seconds data, 64-point FFT option needs around 720 ms
  // 32-point FFT option needs around 520ms
  __shared__ int8_t tile[NPOL_SAMP * NDIM_POL][TILE_DIM][TILE_DIM];

  int i, x, y;
  size_t loc, loc_rt2, loc_out, loc_freq;
  cufftComplex p1, p2;

  x = threadIdx.x;
  loc = blockIdx.x * gridDim.y * blockDim.x * TILE_DIM +
    blockIdx.y * blockDim.x * TILE_DIM +
    x;

  for (i = 0; i < TILE_DIM; i += NROWBLOCK_TRANS)
    {
      y = threadIdx.y + i;
      loc_rt2 = loc + y * blockDim.x;
	
      loc_freq = blockIdx.y * TILE_DIM + y;
      
      p1 = dbuf_rt2[loc_rt2];
      p2 = dbuf_rt2[loc_rt2 + offset_rt2];

      //if(ddat_scl[loc_freq] == 0)
      //	{
      //	  tile[0][y][x] = __float2int_rz(p1.x);
      //	  tile[1][y][x] = __float2int_rz(p1.y);
      //	  tile[2][y][x] = __float2int_rz(p2.x);
      //	  tile[3][y][x] = __float2int_rz(p2.y);
      //	}
      //else
      //	{	  
      //	  tile[0][y][x] = __float2int_rz((p1.x - ddat_offs[loc_freq]) / ddat_scl[loc_freq]);
      //	  tile[1][y][x] = __float2int_rz((p1.y - ddat_offs[loc_freq]) / ddat_scl[loc_freq]);
      //	  tile[2][y][x] = __float2int_rz((p2.x - ddat_offs[loc_freq]) / ddat_scl[loc_freq]);
      //	  tile[3][y][x] = __float2int_rz((p2.y - ddat_offs[loc_freq]) / ddat_scl[loc_freq]);
      //	}
      //tile[0][y][x] = __float2int_rz(p1.x) >> 14;
      //tile[1][y][x] = __float2int_rz(p1.y) >> 14;
      //tile[2][y][x] = __float2int_rz(p2.x) >> 14;
      //tile[3][y][x] = __float2int_rz(p2.y) >> 14;

      tile[0][y][x] = __float2int_rz(p1.x) >> FOLD_SCALE;
      tile[1][y][x] = __float2int_rz(p1.y) >> FOLD_SCALE;
      tile[2][y][x] = __float2int_rz(p2.x) >> FOLD_SCALE;
      tile[3][y][x] = __float2int_rz(p2.y) >> FOLD_SCALE;
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


/*
  This kernel will add data in frequency, detect and scale the added data;
  The detail for the add here is different from the normal sum as we need to put two polarisation togethere here;
 */
__global__ void add_detect_scale_kernel(cufftComplex *dbuf_rt2, uint8_t *dbuf_out_search, size_t offset_rt2, float *ddat_offs, float *ddat_scl)
{
  extern __shared__ float scale_sdata[];
  size_t tid, loc1, loc2, loc11, loc22, loc_freq, s;
  float power;
  
  tid = threadIdx.x;

  loc1 = blockIdx.x * gridDim.y * (blockDim.x * 2) +
    blockIdx.y * (blockDim.x * 2) +
    threadIdx.x;
  loc11 = loc1 + blockDim.x;

  loc2 = loc1 + offset_rt2;
  loc22 = loc2 + blockDim.x;
  
  /* Put two polarisation into shared memory at the same time */
  scale_sdata[tid] =
    dbuf_rt2[loc1].x * dbuf_rt2[loc1].x +
    dbuf_rt2[loc11].x * dbuf_rt2[loc11].x +
    dbuf_rt2[loc1].y * dbuf_rt2[loc1].y +
    dbuf_rt2[loc11].y * dbuf_rt2[loc11].y +
    dbuf_rt2[loc2].x * dbuf_rt2[loc2].x +
    dbuf_rt2[loc22].x * dbuf_rt2[loc22].x +
    dbuf_rt2[loc2].y * dbuf_rt2[loc2].y +
    dbuf_rt2[loc22].y * dbuf_rt2[loc22].y;

  __syncthreads();

  /* do reduction in shared mem */
  for (s=blockDim.x/2; s>0; s>>=1)
    {
      if (tid < s)
	scale_sdata[tid] += scale_sdata[tid + s];
      __syncthreads();
    }
  
  /* write result of this block to global mem */
  if (tid == 0)
    {
      loc_freq = blockIdx.y;
      power = scale_sdata[0]/(NPOL_SAMP * NDIM_POL * CUFFT_NX1 * NCHAN_KEEP2 / NCHAN_SEARCH)/(NPOL_SAMP * NDIM_POL * CUFFT_NX1 * NCHAN_KEEP2 / NCHAN_SEARCH);

      dbuf_out_search[blockIdx.x * gridDim.y + blockIdx.y] = __float2uint_rz((power - ddat_offs[loc_freq]) / ddat_scl[loc_freq]);// scale it;
    }
}

/*
   This kernel will make the scale calculation of search mode easier, the input is PTF data and the output is padded data
   1. add data in frequency and get the channels into NCHAN_SEARCH;
   2. detect the added data;
   3. pad the dbuf_rt1.x with power;
   4. pad the dbuf_rt1.y with the power of power;
   5. the importtant here is that the order of padded data is in FT;
 */
__global__ void add_detect_pad_kernel(cufftComplex *dbuf_rt2, cufftComplex *dbuf_rt1, size_t offset_rt2)
{
  extern __shared__ float pad_sdata[];
  size_t tid, loc1, loc11, loc2, loc22, s;
  float power, power2;
  
  tid = threadIdx.x;
  loc1 = blockIdx.x * gridDim.y * (blockDim.x * 2) +
    blockIdx.y * (blockDim.x * 2) +
    threadIdx.x;
  loc11 = loc1 + blockDim.x;
  loc2 = loc1 + offset_rt2;
  loc22 = loc2 + blockDim.x;
  
  /* Put two polarisation into shared memory at the same time */
  pad_sdata[tid] =
    dbuf_rt2[loc1].x * dbuf_rt2[loc1].x +
    dbuf_rt2[loc11].x * dbuf_rt2[loc11].x +
    dbuf_rt2[loc1].y * dbuf_rt2[loc1].y +
    dbuf_rt2[loc11].y * dbuf_rt2[loc11].y +
    dbuf_rt2[loc2].x * dbuf_rt2[loc2].x +
    dbuf_rt2[loc22].x * dbuf_rt2[loc22].x +
    dbuf_rt2[loc2].y * dbuf_rt2[loc2].y +
    dbuf_rt2[loc22].y * dbuf_rt2[loc22].y;
  __syncthreads();

  /* do reduction in shared mem */
  for (s=blockDim.x/2; s>0; s>>=1)
    {
      if (tid < s)
	pad_sdata[tid] += pad_sdata[tid + s];
      __syncthreads();
    }
  
  /* write result of this block to global mem */
  if (tid == 0)
    {
      power = pad_sdata[0]/(NPOL_SAMP * NDIM_POL * CUFFT_NX1 * NCHAN_KEEP2 / NCHAN_SEARCH)/(NPOL_SAMP * NDIM_POL * CUFFT_NX1 * NCHAN_KEEP2 / NCHAN_SEARCH);
      power2 = power * power;

      dbuf_rt1[blockIdx.y * gridDim.x + blockIdx.x].x = power;
      dbuf_rt1[blockIdx.y * gridDim.x + blockIdx.x].y = power2;
    }
}