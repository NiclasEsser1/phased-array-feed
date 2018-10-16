#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "kernel.cuh"
#include "baseband2power.cuh"
#include "cudautil.cuh"


/*
  This kernel is used to :
  1. unpack the incoming data reading from ring buffer;
  2. reorder the order from TFTFP to PFT;
  3. detect the data
*/
__global__ void unpack_detect_kernel(int64_t *dbuf_in,  float *dbuf_rt1)
{
  uint64_t loc_in, loc_rt1;
  int64_t tmp;
  int16_t p1x, p1y, p2x, p2y;
  
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
  
  // Put the data into FT order  
  loc_rt1 = blockIdx.y * gridDim.x * blockDim.x * blockDim.y +
    threadIdx.y * gridDim.x * blockDim.x +
    blockIdx.x * blockDim.x +
    threadIdx.x;

  p1x = (int16_t)((tmp & 0x000000000000ffffULL));  
  p1y = (int16_t)((tmp & 0x00000000ffff0000ULL) >> 16);
  p2x = (int16_t)((tmp & 0x0000ffff00000000ULL) >> 32);
  p2y = (int16_t)((tmp & 0xffff000000000000ULL) >> 48);
  
  dbuf_rt1[loc_rt1] = p1x * p1x + p1y * p1y + p2x * p2x + p2y * p2y;
}


/*
  This kernel will get the sum of all elements in dbuf_rt1, which is the buffer for each stream
 */
__global__ void sum_kernel(float *dbuf_rt1, float *dbuf_rt2)
{
  extern __shared__ float sum_sdata[];
  uint64_t tid, loc, s;
  
  tid = threadIdx.x;
  loc = blockIdx.x * gridDim.y * (blockDim.x * 2) +
    blockIdx.y * (blockDim.x * 2) +
    threadIdx.x;
  sum_sdata[tid] = dbuf_rt1[loc];
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
    dbuf_rt2[blockIdx.x * gridDim.y + blockIdx.y] = sum_sdata[0];
}
