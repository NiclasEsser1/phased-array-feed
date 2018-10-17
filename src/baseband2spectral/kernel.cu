#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "kernel.cuh"
#include "baseband2spectral.cuh"
#include "cudautil.cuh"

/*
  This kernel is used to :
  1. unpack the incoming data reading from ring buffer and reorder the order from TFTFP to PFT;
*/
__global__ void unpack_kernel(int64_t *dbuf_in,  cufftComplex *dbuf_rtc, uint64_t offset_rtc)
{
  uint64_t loc_in, loc_rtc;
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
  loc_rtc = blockIdx.y * gridDim.x * blockDim.x * blockDim.y +
    threadIdx.y * gridDim.x * blockDim.x +
    blockIdx.x * blockDim.x +
    threadIdx.x;
  
  dbuf_rtc[loc_rtc].x = (int16_t)((tmp & 0x000000000000ffffULL));  
  dbuf_rtc[loc_rtc].y = (int16_t)((tmp & 0x00000000ffff0000ULL) >> 16);
  
  loc_rtc = loc_rtc + offset_rtc;
  dbuf_rtc[loc_rtc].x = (int16_t)((tmp & 0x0000ffff00000000ULL) >> 32);
  dbuf_rtc[loc_rtc].y = (int16_t)((tmp & 0xffff000000000000ULL) >> 48);
}

/* 
   This kernel is used to :
   1. swap the halves of CUDA FFT output, we need to do that because CUDA FFT put the centre frequency at bin 0;
   2. drop the first three and last two points of the swapped 32-points FFT output, which will reduce to oversample rate to 1;
   for 64 points FFT, drop the first and last five points;
   3. Reorder the data from PFTF to PFT;
   4. Detect it;
*/
__global__ void swap_select_transpose_detect_kernel(cufftComplex *dbuf_rtc, float *dbuf_rtf1, uint64_t offset_rtc)
{
  int remainder, loc;
  uint64_t loc_rtc, loc_rtf1;
  cufftComplex p1, p2;

  remainder = (threadIdx.x + CUFFT_MOD)%CUFFT_NX;
  if(remainder < NCHAN_KEEP_CHAN)
    {
      loc_rtc = blockIdx.x * gridDim.y * blockDim.x +
	blockIdx.y * blockDim.x +
	threadIdx.x;
      p1 = dbuf_rtc[loc_rtc];      
      p2 = dbuf_rtc[loc_rtc + offset_rtc];

      loc = blockIdx.x * NCHAN_KEEP_CHAN + remainder;
      loc_rtf1 = blockDim.y * loc + blockIdx.y; // FT order      
      dbuf_rtf1[loc_rtf1] = p1.x * p1.x + p1.y * p1.y + p2.x * p2.x + p2.y * p2.y;
    }
}

/*
  This kernel will get the sum of all elements in dbuf_rt1, which is the buffer for each stream
 */
__global__ void sum_kernel(float *dbuf_rtf1, float *dbuf_rtf2)
{
  extern __shared__ float sum_sdata[];
  uint64_t tid, loc, s;
  
  tid = threadIdx.x;
  loc = blockIdx.x * gridDim.y * (blockDim.x * 2) +
    blockIdx.y * (blockDim.x * 2) +
    threadIdx.x;
  sum_sdata[tid] = dbuf_rtf1[loc] + dbuf_rtf1[loc + blockDim.x]; 
  __syncthreads();

  /* do reduction in shared mem */
  for (s=blockDim.x/2; s>0; s>>=1)
    {
      if (tid < s)
	sum_sdata[tid] += sum_sdata[tid + s];
      __syncthreads();
    }

  /* write result of this block to global mem */
  if (tid == 0)
    dbuf_rtf2[blockIdx.x * gridDim.y + blockIdx.y] = sum_sdata[0];
}