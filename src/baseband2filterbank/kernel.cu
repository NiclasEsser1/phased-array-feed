#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "kernel.cuh"
#include "baseband2filterbank.cuh"
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

  mod = (threadIdx.x + CUFFT_MOD)%CUFFT_NX;	
  if(mod < NCHAN_KEEP_CHAN)
    {
      loc = blockIdx.x * NCHAN_KEEP_CHAN + mod - NCHAN_EDGE;
      if((loc >= 0) && (loc < NCHAN_KEEP_BAND))
	{
	  loc_rt1 = blockIdx.x * gridDim.y * blockDim.x +
	    blockIdx.y * blockDim.x +
	    threadIdx.x;

	  loc_rt = blockIdx.y * NCHAN_KEEP_BAND + loc;  

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
  ddat_scl[loc_freq] = SCL_NSIG * sqrtf(dsquare_mean[loc_freq] - ddat_offs[loc_freq] * ddat_offs[loc_freq]) / SCL_UINT8;
}

/*
  This kernel will add data in frequency, detect and scale the added data;
  The detail for the add here is different from the normal sum as we need to put two polarisation togethere here;
 */
__global__ void add_detect_scale_kernel(cufftComplex *dbuf_rt2, uint8_t *dbuf_out, size_t offset_rt2, float *ddat_offs, float *ddat_scl)
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
      power = scale_sdata[0]/(NPOL_SAMP * NDIM_POL * CUFFT_NX * NSAMP_AVE)/(NPOL_SAMP * NDIM_POL * CUFFT_NX * NSAMP_AVE);

      dbuf_out[blockIdx.x * gridDim.y + blockIdx.y] = __float2uint_rz((power - ddat_offs[loc_freq]) / ddat_scl[loc_freq]);// scale it;
    }
}

/*
   This kernel will make the scale calculation of search mode easier, the input is PTF data and the output is padded data
   1. add data in frequency and get the channels into NCHAN;
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
      power = pad_sdata[0]/(NPOL_SAMP * NDIM_POL * CUFFT_NX * NSAMP_AVE)/(NPOL_SAMP * NDIM_POL * CUFFT_NX * NSAMP_AVE);
      power2 = power * power;

      dbuf_rt1[blockIdx.y * gridDim.x + blockIdx.x].x = power;
      dbuf_rt1[blockIdx.y * gridDim.x + blockIdx.x].y = power2;
    }
}