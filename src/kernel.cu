#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "kernel.cuh"
#include "cudautil.cuh"

/*
  This kernel is used to :
  1. unpack the data reading from ring buffer;
  2. reorder the order from TFTFP to PFT;
  3. put the data into cufftComplex array;
*/
__global__ void unpack_kernel(int64_t *dbuf_in,  cufftComplex *dbuf_out, uint64_t offset_out)
{
  uint64_t loc_in, loc_out;
  int64_t tmp;
  
  /* 
     Loc for the input array, it is in continuous order, it is in (STREAM_BUF_NDFSTP)T(NCHUNK_NIC)F(NSAMP_DF)T(NCHAN_CHUNK)F(NPOL_SAMP)P order
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
  loc_out = blockIdx.y * gridDim.x * blockDim.x * blockDim.y +
    threadIdx.y * gridDim.x * blockDim.x +
    blockIdx.x * blockDim.x +
    threadIdx.x;
  
  dbuf_out[loc_out].x = (int16_t)((tmp & 0x000000000000ffffULL));  
  dbuf_out[loc_out].y = (int16_t)((tmp & 0x00000000ffff0000ULL) >> 16);
  
  loc_out = loc_out + offset_out;
  dbuf_out[loc_out].x = (int16_t)((tmp & 0x0000ffff00000000ULL) >> 32);
  dbuf_out[loc_out].y = (int16_t)((tmp & 0xffff000000000000ULL) >> 48);
}

/* 
   This kernel is used to :
   1.  swap the halves of CUDA FFT output, we need to do that because CUDA FFT put the centre frequency at bin 0;
   2.1 drop the first 3 and last 2 points of the swapped 32-points FFT output, which will reduce to oversample rate to 1;
       for 64 points FFT, drop the first and last 5 points;
       for 128 points FFT, we need to drop the first and last 10 points;
   2.2 drop some channels (depends on the setup) at each end of band to give a good number for the accumulation in frequency;
   3.  reorder the FFT data from PFTF to PTF;
*/
__global__ void swap_select_transpose_ptf_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t offset_in, uint64_t offset_out, int cufft_nx, int cufft_mod, int nchan_keep_chan, int nchan_keep_band, int nchan_edge)
{
  int remainder, loc;
  int64_t loc_in, loc_out;
  cufftComplex p1, p2;

  remainder = (threadIdx.x + cufft_mod)%cufft_nx;
  if(remainder < nchan_keep_chan)
    {
      loc = blockIdx.x * nchan_keep_chan + remainder - nchan_edge;
      if((loc >= 0) && (loc < nchan_keep_band))
	{
	  loc_in = blockIdx.x * gridDim.y * blockDim.x +
	    blockIdx.y * blockDim.x +
	    threadIdx.x;

	  loc_out = blockIdx.y * nchan_keep_band + loc;  

	  p1 = dbuf_in[loc_in];
	  dbuf_out[loc_out].x = p1.x;
	  dbuf_out[loc_out].y = p1.y;

	  loc_out = loc_out + offset_out;

	  p2 = dbuf_in[loc_in + offset_in];
	  dbuf_out[loc_out].x = p2.x;
	  dbuf_out[loc_out].y = p2.y;
	}
    }
}

/* 
   This kernel is used to :
   1. swap the halves of CUDA FFT output, we need to do that because CUDA FFT put the centre frequency at bin 0;
   2. drop the first 3 and last 2 points of the swapped 32-points FFT output, which will reduce to oversample rate to 1;
      for 64 points FFT, drop the first and last 5 points;
      for 128 points FFT, we need to drop the first and last 10 points;
      for 1024 points FFT, we drop the first and last 80 points;
   3. reorder the FFT data from PFTF to PFT;
*/
__global__ void swap_select_transpose_pft_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t offset_in, uint64_t offset_out, int cufft_nx, int cufft_mod, int nchan_keep_chan)
{
  int remainder, loc;
  int64_t loc_in, loc_out;
  cufftComplex p1, p2;

  remainder = (threadIdx.x + cufft_mod)%cufft_nx;
  if(remainder < nchan_keep_chan)
    {
      loc = blockIdx.x * nchan_keep_chan + remainder;
      loc_in = blockIdx.x * gridDim.y * blockDim.x +
	blockIdx.y * blockDim.x +
	threadIdx.x;
      
      loc_out = gridDim.y * loc + blockIdx.y;  // The profermance here may have problem;
      
      p1 = dbuf_in[loc_in];
      dbuf_out[loc_out].x = p1.x;
      dbuf_out[loc_out].y = p1.y;
      
      loc_out = loc_out + offset_out;
      
      p2 = dbuf_in[loc_in + offset_in];
      dbuf_out[loc_out].x = p2.x;
      dbuf_out[loc_out].y = p2.y;
    }
}

/* 
   This kernel is used to :
   1. swap the halves of CUDA FFT output, we need to do that because CUDA FFT put the centre frequency at bin 0;
   2. drop the first 3 and last 2 points of the swapped 32-points FFT output, which will reduce to oversample rate to 1;
      for 64 points FFT, drop the first and last 5 points;
      for 128 points FFT, we need to drop the first and last 10 points;
      for 1024 points FFT, we drop the first and last 80 points;
   3. reorder the FFT data from PFTF to PFT;

   this kernel is planned to optimize the swap_select_transpose_pft_kernel, the performance of swap_select_transpose_pft is not very good;
   The tranpose part of this kernel works with any size of matrix;
*/
__global__ void swap_select_transpose_pft1_kernel(cufftComplex* dbuf_in, cufftComplex *dbuf_out, int n, int m, uint64_t offset_in, uint64_t offset_out, int cufft_nx, int cufft_mod, int nchan_keep_chan)
{
  int i;
  int64_t loc_in, loc_out;
  int reminder;
  
  __shared__ cufftComplex tile[2][TILE_DIM][TILE_DIM + 1];
  
  // Load matrix into tile
  // Every Thread loads in this case 4 elements into tile.  
  int i_n = blockIdx.x * TILE_DIM + threadIdx.x;
  int i_m = blockIdx.y * TILE_DIM + threadIdx.y; // <- threadIdx.y only between 0 and 7
  for (i = 0; i < TILE_DIM; i += NROWBLOCK_TRANS)
    {
      if(i_n < n  && (i_m+i) < m)
  	{
  	  //loc_in = blockIdx.z * (gridDim.x * TILE_DIM) * (gridDim.y * TILE_DIM) + (i_m+i)*n + i_n;
	  loc_in = blockIdx.z * m * n + (i_m+i)*n + i_n;
  	  tile[0][threadIdx.y+i][threadIdx.x].x = dbuf_in[loc_in].x;
  	  tile[0][threadIdx.y+i][threadIdx.x].y = dbuf_in[loc_in].y;
  	  tile[1][threadIdx.y+i][threadIdx.x].x = dbuf_in[loc_in + offset_in].x;
  	  tile[1][threadIdx.y+i][threadIdx.x].y = dbuf_in[loc_in + offset_in].y;
        }
    }
  __syncthreads();
  
  i_n = blockIdx.y * TILE_DIM + threadIdx.x; 
  i_m = blockIdx.x * TILE_DIM + threadIdx.y;
  for (i = 0; i < TILE_DIM; i += NROWBLOCK_TRANS)
    {
      reminder = (i_m + i + cufft_mod) % cufft_nx;      
      if(i_n < m  && (i_m+i) < n && reminder < nchan_keep_chan)
  	{
  	  //loc_out = blockIdx.z * nchan_keep_chan * (gridDim.y * TILE_DIM) + reminder*m + i_n;
	  loc_out = blockIdx.z * nchan_keep_chan * m + reminder*m + i_n;
  	  dbuf_out[loc_out].x              = tile[0][threadIdx.x][threadIdx.y + i].x; // <- multiply by m, non-squared!
  	  dbuf_out[loc_out].y              = tile[0][threadIdx.x][threadIdx.y + i].y; // <- multiply by m, non-squared!
  	  dbuf_out[loc_out + offset_out].x = tile[1][threadIdx.x][threadIdx.y + i].x; // <- multiply by m, non-squared!
  	  dbuf_out[loc_out + offset_out].y = tile[1][threadIdx.x][threadIdx.y + i].y; // <- multiply by m, non-squared!
  	}
    }
}

///* A example of accumulate */
//template <unsigned int blockSize>
//__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n)
//{
//  extern __shared__ int sdata[];
//  //extern __shared__ cufftComplex sdata[];
//  unsigned int tid = threadIdx.x;
//  //unsigned int i = blockIdx.x*(blockSize*2) + tid;
//  unsigned int i = blockIdx.x*gridDim.y*(blockSize*2) +
//    blockIdx.y*(blockSize*2) +
//    threadIdx.x;
//  //unsigned int gridSize = blockSize*2*gridDim.x;
//  unsigned int gridSize = (blockSize*2)*gridDim.x*gridDim.y;
//
//  sdata[tid] = 0;
//  
//  while (i < n)
//    {
//      sdata[tid] += g_idata[i] + g_idata[i+blockSize];
//      i += gridSize;
//    }
//  __syncthreads();
//  
//  if (blockSize >= 1024)
//    if (tid < 512)
//      sdata[tid] += sdata[tid + 512];
//  __syncthreads();
//  
//  if (blockSize >= 512)
//    if (tid < 256)
//      sdata[tid] += sdata[tid + 256];
//  __syncthreads();
//
//  if (blockSize >= 256)
//    if (tid < 128)
//      sdata[tid] += sdata[tid + 128];
//  __syncthreads();
//
//  if (blockSize >= 128)
//    if (tid < 64)
//      sdata[tid] += sdata[tid + 64];
//  __syncthreads();
//    
//  if (tid < 32)
//    {
//      if (blockSize >= 64)
//	sdata[tid] += sdata[tid + 32];
//      if (blockSize >= 32)
//	sdata[tid] += sdata[tid + 16];
//      if (blockSize >= 16)
//	sdata[tid] += sdata[tid + 8];
//      if (blockSize >= 8)
//	sdata[tid] += sdata[tid + 4];
//      if (blockSize >= 4)
//	sdata[tid] += sdata[tid + 2];  
//      if (blockSize >= 2)
//	sdata[tid] += sdata[tid + 1];
//    }
//  
//  if (tid == 0)
//    //g_odata[blockIdx.x] = sdata[0];
//    g_odata[blockIdx.x * gridDim.y + blockIdx.y] = sdata[0];
//}
//

/*
  This kernel accumulates all elements in each channel
*/
__global__ void accumulate_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out)
{
  extern volatile __shared__ cufftComplex accumulate_sdata[];
  uint64_t tid, loc, s;
  
  tid = threadIdx.x;
  loc = blockIdx.x * gridDim.y * (blockDim.x * 2) +
    blockIdx.y * (blockDim.x * 2) +
    threadIdx.x;
  accumulate_sdata[tid].x = dbuf_in[loc].x + dbuf_in[loc + blockDim.x].x; 
  accumulate_sdata[tid].y = dbuf_in[loc].y + dbuf_in[loc + blockDim.x].y;
  __syncthreads();

  /* do reduction in shared mem */
  for (s=blockDim.x/2; s>0; s>>=1)
    {
      if (tid < s)
  	{
  	  accumulate_sdata[tid].x += accumulate_sdata[tid + s].x;
  	  accumulate_sdata[tid].y += accumulate_sdata[tid + s].y;
  	}
      __syncthreads();
    }

  /* write result of this block to global mem */
  if (tid == 0)
    {
      dbuf_out[blockIdx.x * gridDim.y + blockIdx.y].x = accumulate_sdata[0].x;
      dbuf_out[blockIdx.x * gridDim.y + blockIdx.y].y = accumulate_sdata[0].y;
    }
}

/*
  This kernel calculate the mean of samples and the mean of sample square 
*/
__global__ void mean_kernel(cufftComplex *buf_in, uint64_t offset_in, float *ddat_offs, float *dsquare_mean, int nstream, float scl_ndim)
{
  int i;
  uint64_t loc_freq, loc;
  float dat_offs = 0, square_mean = 0;
  
  loc_freq = threadIdx.x;
  
  for (i = 0; i < nstream; i++)
    {
      loc = loc_freq + i * offset_in;
      dat_offs    += (buf_in[loc].x / scl_ndim);
      square_mean += (buf_in[loc].y / scl_ndim);
    }
  
  ddat_offs[loc_freq]    += dat_offs;
  dsquare_mean[loc_freq] += square_mean;
}

/*
  This kernel is used to calculate the scale of data based on the mean calculated by mean_kernel
*/
__global__ void scale_kernel(float *ddat_offs, float *dsquare_mean, float *ddat_scl, float scl_nsig, float scl_uint8)
{
  uint64_t loc_freq  = threadIdx.x;
  ddat_scl[loc_freq] = scl_nsig * sqrtf(dsquare_mean[loc_freq] - ddat_offs[loc_freq] * ddat_offs[loc_freq]) / scl_uint8;
}

/*
  This kernel is used to calculate the scale of data based on the mean calculated by mean_kernel
*/
__global__ void scale1_kernel(cufftComplex *mean, float *ddat_scl, float scl_nsig, float scl_uint8)
{
  uint64_t loc_freq  = threadIdx.x;
  ddat_scl[loc_freq] = scl_nsig * sqrtf(mean[loc_freq].y - mean[loc_freq].x * mean[loc_freq].x) / scl_uint8;
}

/*
  This kernel is used to calculate the scale of data based on the mean calculated by mean_kernel
*/
__global__ void scale2_kernel(cufftComplex *offset_scale, float scl_nsig, float scl_uint8)
{
  uint64_t loc_freq  = threadIdx.x;
  offset_scale[loc_freq].y = scl_nsig * sqrtf(offset_scale[loc_freq].y - offset_scale[loc_freq].x * offset_scale[loc_freq].x) / scl_uint8;
}

/*
  This kernel will detect data, accumulate it in frequency and scale it;
  The accumulation here is different from the normal accumulation as we need to put two polarisation togethere here;
 */
__global__ void detect_faccumulate_scale_kernel(cufftComplex *dbuf_in, uint8_t *dbuf_out, uint64_t offset_in, float *ddat_offs, float *ddat_scl)
{
  extern volatile __shared__ float scale_sdata[];
  uint64_t tid, loc1, loc2, loc11, loc22, loc_freq, s;
  float power;
  
  tid = threadIdx.x;
  loc1 = blockIdx.x * gridDim.y * (blockDim.x * 2) +
    blockIdx.y * (blockDim.x * 2) +
    threadIdx.x;
  loc2 = loc1 + offset_in;
  
  loc11 = loc1 + blockDim.x;
  loc22 = loc2 + blockDim.x;
  
  /* Put two polarisation into shared memory at the same time */
  scale_sdata[tid] =
    dbuf_in[loc1].x * dbuf_in[loc1].x +
    dbuf_in[loc11].x * dbuf_in[loc11].x +    
    dbuf_in[loc1].y * dbuf_in[loc1].y +
    dbuf_in[loc11].y * dbuf_in[loc11].y +
    
    dbuf_in[loc2].x * dbuf_in[loc2].x +
    dbuf_in[loc22].x * dbuf_in[loc22].x +    
    dbuf_in[loc2].y * dbuf_in[loc2].y +
    dbuf_in[loc22].y * dbuf_in[loc22].y;

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
      power = scale_sdata[0];

      if(ddat_scl[loc_freq] == 0.0)
	//dbuf_out[blockIdx.x * gridDim.y + blockIdx.y] = __float2uint_rz(power);
	dbuf_out[blockIdx.x * gridDim.y + gridDim.y - blockIdx.y - 1] = __float2uint_rz(power); // Reverse frequency order
      else
	//dbuf_out[blockIdx.x * gridDim.y + blockIdx.y] = __float2uint_rz((power - ddat_offs[loc_freq]) / ddat_scl[loc_freq] + OFFS_UINT8);
	dbuf_out[blockIdx.x * gridDim.y + gridDim.y - blockIdx.y - 1] = __float2uint_rz((power - ddat_offs[loc_freq]) / ddat_scl[loc_freq] + OFFS_UINT8); // Reverse frequency order
    }
}

/*
  This kernel will detect data, accumulate it in frequency and scale it;
  The accumulation here is different from the normal accumulation as we need to put two polarisation togethere here;
 */
__global__ void detect_faccumulate_scale1_kernel(cufftComplex *dbuf_in, uint8_t *dbuf_out, uint64_t offset_in, cufftComplex *offset_scale)
{
  extern volatile __shared__ float scale_sdata[];
  uint64_t tid, loc1, loc2, loc11, loc22, loc_freq, s;
  float power;
  
  tid = threadIdx.x;
  loc1 = blockIdx.x * gridDim.y * (blockDim.x * 2) +
    blockIdx.y * (blockDim.x * 2) +
    threadIdx.x;
  loc2 = loc1 + offset_in;
  
  loc11 = loc1 + blockDim.x;
  loc22 = loc2 + blockDim.x;
  
  /* Put two polarisation into shared memory at the same time */
  scale_sdata[tid] =
    dbuf_in[loc1].x * dbuf_in[loc1].x +
    dbuf_in[loc11].x * dbuf_in[loc11].x +    
    dbuf_in[loc1].y * dbuf_in[loc1].y +
    dbuf_in[loc11].y * dbuf_in[loc11].y +
    
    dbuf_in[loc2].x * dbuf_in[loc2].x +
    dbuf_in[loc22].x * dbuf_in[loc22].x +    
    dbuf_in[loc2].y * dbuf_in[loc2].y +
    dbuf_in[loc22].y * dbuf_in[loc22].y;

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
      power = scale_sdata[0];

      if(offset_scale[loc_freq].y == 0.0)
	//dbuf_out[blockIdx.x * gridDim.y + blockIdx.y] = __float2uint_rz(power);
	dbuf_out[blockIdx.x * gridDim.y + gridDim.y - blockIdx.y - 1] = __float2uint_rz(power); // Reverse frequency order
      else
	//dbuf_out[blockIdx.x * gridDim.y + blockIdx.y] = __float2uint_rz((power - offset_scale[loc_freq].x) / offset_scale[loc_freq].y + OFFS_UINT8);
	dbuf_out[blockIdx.x * gridDim.y + gridDim.y - blockIdx.y - 1] = __float2uint_rz((power - offset_scale[loc_freq].x) / offset_scale[loc_freq].y + OFFS_UINT8); // Reverse frequency order
    }
}

/*
   This kernel will make the scale calculation of search mode easier, the input is PTF data and the output is padded data in FT
   1.  detect the data;
   2.  accumulate data in frequency and get the channels into NCHAN;
   3.1 pad the dbuf_out.x with power;
   3.2 pad the dbuf_out.y with the power of power;
   4.  reorder the data into FT;
 */
__global__ void detect_faccumulate_pad_transpose_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t offset_in)
{
  extern __shared__ float pad_sdata[];
  uint64_t tid, loc1, loc11, loc2, loc22, s;
  float power, power_square;
  
  tid = threadIdx.x;
  loc1 = blockIdx.x * gridDim.y * (blockDim.x * 2) +
    blockIdx.y * (blockDim.x * 2) +
    threadIdx.x;
  loc2 = loc1 + offset_in;
  
  loc11 = loc1 + blockDim.x;
  loc22 = loc2 + blockDim.x;
  
  /* Put two polarisation into shared memory at the same time */
  pad_sdata[tid] =
    dbuf_in[loc1].x * dbuf_in[loc1].x +
    dbuf_in[loc11].x * dbuf_in[loc11].x +    
    dbuf_in[loc1].y * dbuf_in[loc1].y +
    dbuf_in[loc11].y * dbuf_in[loc11].y +
    
    dbuf_in[loc2].x * dbuf_in[loc2].x +
    dbuf_in[loc22].x * dbuf_in[loc22].x +    
    dbuf_in[loc2].y * dbuf_in[loc2].y +
    dbuf_in[loc22].y * dbuf_in[loc22].y;
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
      //power = pad_sdata[0]/(NPOL_SAMP * NDIM_POL * CUFFT_NX * NSAMP_AVE)/(NPOL_SAMP * NDIM_POL * CUFFT_NX * NSAMP_AVE);
      power = pad_sdata[0];
      power_square = power * power;

      dbuf_out[blockIdx.y * gridDim.x + blockIdx.x].x = power;
      dbuf_out[blockIdx.y * gridDim.x + blockIdx.x].y = power_square;
    }
}