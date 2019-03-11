#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <inttypes.h>
#include "constants.h"

/* For raw data unpack to get ready for forward FFT */
__global__ void unpack_kernel(int64_t *dbuf_in,  cufftComplex *dbuf_out, uint64_t offset_out);
__global__ void unpack1_kernel(int64_t *dbuf_in,  cufftComplex *dbuf_out, uint64_t offset_out, cufftComplex *dbuf_out_zoom, uint64_t offset_out_zoom, int zoom_start_chunk, int zoom_nchunk);

/* Use after forward FFT to get ready for further steps */
__global__ void swap_select_transpose_ptf_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t offset_in, uint64_t offset_out, int cufft_nx, int cufft_mod, int nchan_keep_chan, int nchan_keep_band, int nchan_edge);
__global__ void swap_select_transpose_pft_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t offset_in, uint64_t offset_out, int cufft_nx, int cufft_mod, int nchan_keep_chan);
__global__ void swap_select_transpose_pft1_kernel(cufftComplex* dbuf_in, cufftComplex *dbuf_out, int n, int m, uint64_t offset_in, uint64_t offset_out, int cufft_nx, int cufft_mod, int nchan_keep_chan);

/* The following 4 kernels are for scale calculation */
__global__ void taccumulate_complex_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out);  // Share between fold and search mode
__global__ void mean_kernel(cufftComplex *buf_in, uint64_t offset_in, float *ddat_offs, float *dsquare_mean, int nstream, float scl_ndim); // Share between fold and search mode
//__global__ void scale_kernel(float *ddat_offs, float *dsquare_mean, float *ddat_scl); // Share between fold and search mode
__global__ void scale_kernel(float *ddat_offs, float *dsquare_mean, float *ddat_scl, float scl_nsig, float scl_format);
__global__ void scale1_kernel(cufftComplex *mean, float *ddat_scl, float scl_nsig, float scl_format);
__global__ void scale2_kernel(cufftComplex *offset_scale, float scl_nsig, float scl_format);
__global__ void scale3_kernel(cufftComplex *offset_scale, uint64_t offset_in, int nstream, float scl_nsig, float scl_format);

/* The following are only for search mode */
__global__ void detect_faccumulate_scale_kernel(cufftComplex *dbuf_in, uint8_t *dbuf_out, uint64_t offset_in, float *ddat_offs, float *ddat_scl);
__global__ void detect_faccumulate_scale1_kernel(cufftComplex *dbuf_in, uint8_t *dbuf_out, uint64_t offset_in, cufftComplex *offset_scale);
__global__ void detect_faccumulate_pad_transpose_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t offset_in);
__global__ void transpose_kernel(float* dbuf_in, float *dbuf_out, uint64_t offset, int n, int m);
__global__ void saccumulate_kernel(float *dbuf, uint64_t offset, int nstream);
__global__ void swap_select_transpose_swap_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t offset_in, uint64_t offset_out, int cufft_nx, int cufft_mod, int nchan_keep_chan);
__global__ void transpose_pad_kernel(cufftComplex *dbuf_out, uint64_t offset_out, cufftComplex *dbuf_in);
__global__ void transpose_scale_kernel(cufftComplex *dbuf_in, int8_t *dbuf_out, int n, int m, uint64_t offset, cufftComplex *offset_scale);
__global__ void transpose_complex_kernel(cufftComplex* dbuf_in, uint64_t offset, cufftComplex *dbuf_out);

// Modified from the example here, https://devtalk.nvidia.com/default/topic/1038617/cuda-programming-and-performance/understanding-and-adjusting-mark-harriss-array-reduction/
// The original code is given by Mark Harris 
//template <unsigned int blockSize>
//__global__ void cuda_reduction(float *array_in, float *reduct, size_t array_len)
//{
//  extern volatile __shared__ float sdata[];
//  size_t  tid        = threadIdx.x,
//    gridSize   = blockSize * gridDim.x,
//    i          = blockIdx.x * blockSize + tid;
//  sdata[tid] = 0;
//  while (i < array_len)
//    { sdata[tid] += array_in[i];
//      i += gridSize; }
//  __syncthreads();
//  if (blockSize >= 512)
//    { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
//  if (blockSize >= 256)
//    { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
//  if (blockSize >= 128)
//    { if (tid <  64) sdata[tid] += sdata[tid + 64]; __syncthreads(); }
//  if (tid < 32)
//    { if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//      if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//      if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//      if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
//      if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
//      if (blockSize >= 2)  sdata[tid] += sdata[tid + 1]; }
//  if (tid == 0) reduct[blockIdx.x] = sdata[0];
//}

template <unsigned int blockSize>
__global__ void reduce6_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t naccumulate)
{
  extern volatile __shared__ cufftComplex sdata[];
  uint64_t i   = threadIdx.x;
  uint64_t tid = i;
  uint64_t loc; 

  sdata[tid].x = 0;
  sdata[tid].y = 0;
  
  while (i < naccumulate)
    { 
      loc = blockIdx.x*gridDim.y*naccumulate + blockIdx.y*naccumulate + i;
      //printf("HERE %"PRIu64"\t%"PRIu64"\t%d\t%d\t%d\t%"PRIu64"\n", tid, i, (int)blockIdx.x, (int)gridDim.y, (int)blockIdx.y, loc);
      sdata[tid].x += dbuf_in[loc].x;
      sdata[tid].y += dbuf_in[loc].y;
      
      i += blockSize;
    }
  __syncthreads();
  
  if (blockSize >= 1024)
    {
      if (tid < 512)
	{
	  //printf("HERE 1024\n");
	  sdata[tid].x += sdata[tid + 512].x;
	  sdata[tid].y += sdata[tid + 512].y;
	}
    }
  __syncthreads();
  
  if (blockSize >= 512)
    {
      if (tid < 256)
	{
	  //printf("HERE 512\n");
	  sdata[tid].x += sdata[tid + 256].x;
	  sdata[tid].y += sdata[tid + 256].y;
	}
    }
  __syncthreads();

  if (blockSize >= 256)
    {
      if (tid < 128)
	{
	  //printf("HERE 256\n");
	  sdata[tid].x += sdata[tid + 128].x;
	  sdata[tid].y += sdata[tid + 128].y;
	}
    }
  __syncthreads();

  if (blockSize >= 128)
    {
      if (tid < 64)
	{
	  //printf("HERE 128\n");
	  sdata[tid].x += sdata[tid + 64].x;
	  sdata[tid].y += sdata[tid + 64].y;
	}
    }
  __syncthreads();
    
  if (tid < 32)
    {
      if (blockSize >= 64)
	{
	  if (tid < 32)
	    {
	      //printf("HERE 64\n");
	      sdata[tid].x += sdata[tid + 32].x;
	      sdata[tid].y += sdata[tid + 32].y;
	    }
	}
      if (blockSize >= 32)
	{
	  if (tid < 16)
	    {
	      //printf("HERE 32\n");
	      sdata[tid].x += sdata[tid + 16].x;
	      sdata[tid].y += sdata[tid + 16].y;
	    }
	}
      if (blockSize >= 16)
	{
	  if (tid < 8)
	    {
	      //printf("HERE 16\n");
	      sdata[tid].x += sdata[tid + 8].x;
	      sdata[tid].y += sdata[tid + 8].y;
	    }
	}
      if (blockSize >= 8)
	{
	  if (tid < 4)
	    {
	      //printf("HERE 8\n");
	      sdata[tid].x += sdata[tid + 4].x;
	      sdata[tid].y += sdata[tid + 4].y;
	    }
	}
      if (blockSize >= 4)
	{
	  if (tid < 2)
	    {
	      //printf("HERE 4\n");
	      sdata[tid].x += sdata[tid + 2].x;
	      sdata[tid].y += sdata[tid + 2].y;
	    }
	}
      if (blockSize >= 2)
	{
	  if (tid < 1)
	    {
	      //printf("HERE 2\n");
	      sdata[tid].x += sdata[tid + 1].x;
	      sdata[tid].y += sdata[tid + 1].y;
	    }
	}
    }
  
  if (tid == 0)
    {
      dbuf_out[blockIdx.x * gridDim.y + blockIdx.y].x = sdata[0].x;
      dbuf_out[blockIdx.x * gridDim.y + blockIdx.y].y = sdata[0].y;
    }
}


// Modified from the example here, https://devtalk.nvidia.com/default/topic/1038617/cuda-programming-and-performance/understanding-and-adjusting-mark-harriss-array-reduction/
// The original code is given by Mark Harris
// Accumulation accorss multiple streams
template <unsigned int blockSize>
__global__ void reduce7_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t offset_in, uint64_t naccumulate, int nstream)
{
  extern volatile __shared__ cufftComplex sdata[];
  uint64_t i   = threadIdx.x, j;
  uint64_t tid = i;
  uint64_t loc; 

  sdata[tid].x = 0;
  sdata[tid].y = 0;
  
  while (i < naccumulate)
    {
      for(j = 0; j < nstream; j++)
	{
	  loc = blockIdx.x*gridDim.y*naccumulate + blockIdx.y*naccumulate + j * offset_in + i;
	  sdata[tid].x += dbuf_in[loc].x;
	  sdata[tid].y += dbuf_in[loc].y;
	}
      i += blockSize;
    }
  __syncthreads();
  
  if (blockSize >= 1024)
    {
      if (tid < 512)
	{
	  //printf("HERE 1024\n");
	  sdata[tid].x += sdata[tid + 512].x;
	  sdata[tid].y += sdata[tid + 512].y;
	}
    }
  __syncthreads();
  
  if (blockSize >= 512)
    {
      if (tid < 256)
	{
	  //printf("HERE 512\n");
	  sdata[tid].x += sdata[tid + 256].x;
	  sdata[tid].y += sdata[tid + 256].y;
	}
    }
  __syncthreads();

  if (blockSize >= 256)
    {
      if (tid < 128)
	{
	  //printf("HERE 256\n");
	  sdata[tid].x += sdata[tid + 128].x;
	  sdata[tid].y += sdata[tid + 128].y;
	}
    }
  __syncthreads();

  if (blockSize >= 128)
    {
      if (tid < 64)
	{
	  //printf("HERE 128\n");
	  sdata[tid].x += sdata[tid + 64].x;
	  sdata[tid].y += sdata[tid + 64].y;
	}
    }
  __syncthreads();
    
  if (tid < 32)
    {
      if (blockSize >= 64)
	{
	  if (tid < 32)
	    {
	      //printf("HERE 64\n");
	      sdata[tid].x += sdata[tid + 32].x;
	      sdata[tid].y += sdata[tid + 32].y;
	    }
	}
      if (blockSize >= 32)
	{
	  if (tid < 16)
	    {
	      //printf("HERE 32\n");
	      sdata[tid].x += sdata[tid + 16].x;
	      sdata[tid].y += sdata[tid + 16].y;
	    }
	}
      if (blockSize >= 16)
	{
	  if (tid < 8)
	    {
	      //printf("HERE 16\n");
	      sdata[tid].x += sdata[tid + 8].x;
	      sdata[tid].y += sdata[tid + 8].y;
	    }
	}
      if (blockSize >= 8)
	{
	  if (tid < 4)
	    {
	      //printf("HERE 8\n");
	      sdata[tid].x += sdata[tid + 4].x;
	      sdata[tid].y += sdata[tid + 4].y;
	    }
	}
      if (blockSize >= 4)
	{
	  if (tid < 2)
	    {
	      //printf("HERE 4\n");
	      sdata[tid].x += sdata[tid + 2].x;
	      sdata[tid].y += sdata[tid + 2].y;
	    }
	}
      if (blockSize >= 2)
	{
	  if (tid < 1)
	    {
	      //printf("HERE 2\n");
	      sdata[tid].x += sdata[tid + 1].x;
	      sdata[tid].y += sdata[tid + 1].y;
	    }
	}
    }
  
  if (tid == 0)
    {
      dbuf_out[blockIdx.x * gridDim.y + blockIdx.y].x = sdata[0].x;
      dbuf_out[blockIdx.x * gridDim.y + blockIdx.y].y = sdata[0].y;
    }
}

// reduce7 + mean
template <unsigned int blockSize>
__global__ void reduce8_kernel(cufftComplex *dbuf_in, float *ddat_offs, float *dsquare_mean, uint64_t offset_in, uint64_t naccumulate, int nstream, float scl_ndim)
{
  extern volatile __shared__ cufftComplex sdata[];
  int j;
  uint64_t i   = threadIdx.x;
  uint64_t tid = i;
  uint64_t loc; 

  sdata[tid].x = 0;
  sdata[tid].y = 0;
  
  while (i < naccumulate)
    {
      for(j = 0; j < nstream; j++)
	{
	  loc = blockIdx.x*gridDim.y*naccumulate + blockIdx.y*naccumulate + j * offset_in + i;
	  sdata[tid].x += dbuf_in[loc].x;
	  sdata[tid].y += dbuf_in[loc].y;
	}
      i += blockSize;
    }
  __syncthreads();
  
  if (blockSize >= 1024)
    {
      if (tid < 512)
	{
	  //printf("HERE 1024\n");
	  sdata[tid].x += sdata[tid + 512].x;
	  sdata[tid].y += sdata[tid + 512].y;
	}
    }
  __syncthreads();
  
  if (blockSize >= 512)
    {
      if (tid < 256)
	{
	  //printf("HERE 512\n");
	  sdata[tid].x += sdata[tid + 256].x;
	  sdata[tid].y += sdata[tid + 256].y;
	}
    }
  __syncthreads();

  if (blockSize >= 256)
    {
      if (tid < 128)
	{
	  //printf("HERE 256\n");
	  sdata[tid].x += sdata[tid + 128].x;
	  sdata[tid].y += sdata[tid + 128].y;
	}
    }
  __syncthreads();

  if (blockSize >= 128)
    {
      if (tid < 64)
	{
	  //printf("HERE 128\n");
	  sdata[tid].x += sdata[tid + 64].x;
	  sdata[tid].y += sdata[tid + 64].y;
	}
    }
  __syncthreads();
    
  if (tid < 32)
    {
      if (blockSize >= 64)
	{
	  if (tid < 32)
	    {
	      //printf("HERE 64\n");
	      sdata[tid].x += sdata[tid + 32].x;
	      sdata[tid].y += sdata[tid + 32].y;
	    }
	}
      if (blockSize >= 32)
	{
	  if (tid < 16)
	    {
	      //printf("HERE 32\n");
	      sdata[tid].x += sdata[tid + 16].x;
	      sdata[tid].y += sdata[tid + 16].y;
	    }
	}
      if (blockSize >= 16)
	{
	  if (tid < 8)
	    {
	      //printf("HERE 16\n");
	      sdata[tid].x += sdata[tid + 8].x;
	      sdata[tid].y += sdata[tid + 8].y;
	    }
	}
      if (blockSize >= 8)
	{
	  if (tid < 4)
	    {
	      //printf("HERE 8\n");
	      sdata[tid].x += sdata[tid + 4].x;
	      sdata[tid].y += sdata[tid + 4].y;
	    }
	}
      if (blockSize >= 4)
	{
	  if (tid < 2)
	    {
	      //printf("HERE 4\n");
	      sdata[tid].x += sdata[tid + 2].x;
	      sdata[tid].y += sdata[tid + 2].y;
	    }
	}
      if (blockSize >= 2)
	{
	  if (tid < 1)
	    {
	      //printf("HERE 2\n");
	      sdata[tid].x += sdata[tid + 1].x;
	      sdata[tid].y += sdata[tid + 1].y;
	    }
	}
    }
  
  if (tid == 0)
    {
      ddat_offs[blockIdx.x * gridDim.y + blockIdx.y]    += (sdata[0].x/scl_ndim);
      dsquare_mean[blockIdx.x * gridDim.y + blockIdx.y] += (sdata[0].y/scl_ndim);
    }
}

// reduce8 + cufftComplex output
template <unsigned int blockSize>
__global__ void reduce9_kernel(cufftComplex *dbuf_in, cufftComplex *offset_scale, uint64_t offset_in, uint64_t naccumulate, int nstream, float scl_ndim)
{
  extern volatile __shared__ cufftComplex sdata[];
  int j;
  uint64_t i   = threadIdx.x;
  uint64_t tid = i;
  uint64_t loc; 

  sdata[tid].x = 0;
  sdata[tid].y = 0;
  
  while (i < naccumulate)
    {
      for(j = 0; j < nstream; j++)
	{
	  loc = blockIdx.x*gridDim.y*naccumulate + blockIdx.y*naccumulate + j * offset_in + i;
	  sdata[tid].x += dbuf_in[loc].x;
	  sdata[tid].y += dbuf_in[loc].y;
	}
      i += blockSize;
    }
  __syncthreads();
  
  if (blockSize >= 1024)
    {
      if (tid < 512)
	{
	  //printf("HERE 1024\n");
	  sdata[tid].x += sdata[tid + 512].x;
	  sdata[tid].y += sdata[tid + 512].y;
	}
    }
  __syncthreads();
  
  if (blockSize >= 512)
    {
      if (tid < 256)
	{
	  //printf("HERE 512\n");
	  sdata[tid].x += sdata[tid + 256].x;
	  sdata[tid].y += sdata[tid + 256].y;
	}
    }
  __syncthreads();

  if (blockSize >= 256)
    {
      if (tid < 128)
	{
	  //printf("HERE 256\n");
	  sdata[tid].x += sdata[tid + 128].x;
	  sdata[tid].y += sdata[tid + 128].y;
	}
    }
  __syncthreads();

  if (blockSize >= 128)
    {
      if (tid < 64)
	{
	  //printf("HERE 128\n");
	  sdata[tid].x += sdata[tid + 64].x;
	  sdata[tid].y += sdata[tid + 64].y;
	}
    }
  __syncthreads();
    
  if (tid < 32)
    {
      if (blockSize >= 64)
	{
	  if (tid < 32)
	    {
	      //printf("HERE 64\n");
	      sdata[tid].x += sdata[tid + 32].x;
	      sdata[tid].y += sdata[tid + 32].y;
	    }
	}
      if (blockSize >= 32)
	{
	  if (tid < 16)
	    {
	      //printf("HERE 32\n");
	      sdata[tid].x += sdata[tid + 16].x;
	      sdata[tid].y += sdata[tid + 16].y;
	    }
	}
      if (blockSize >= 16)
	{
	  if (tid < 8)
	    {
	      //printf("HERE 16\n");
	      sdata[tid].x += sdata[tid + 8].x;
	      sdata[tid].y += sdata[tid + 8].y;
	    }
	}
      if (blockSize >= 8)
	{
	  if (tid < 4)
	    {
	      //printf("HERE 8\n");
	      sdata[tid].x += sdata[tid + 4].x;
	      sdata[tid].y += sdata[tid + 4].y;
	    }
	}
      if (blockSize >= 4)
	{
	  if (tid < 2)
	    {
	      //printf("HERE 4\n");
	      sdata[tid].x += sdata[tid + 2].x;
	      sdata[tid].y += sdata[tid + 2].y;
	    }
	}
      if (blockSize >= 2)
	{
	  if (tid < 1)
	    {
	      //printf("HERE 2\n");
	      sdata[tid].x += sdata[tid + 1].x;
	      sdata[tid].y += sdata[tid + 1].y;
	    }
	}
    }
  
  if (tid == 0)
    {
      offset_scale[blockIdx.x * gridDim.y + blockIdx.y].x += sdata[0].x/scl_ndim;
      offset_scale[blockIdx.x * gridDim.y + blockIdx.y].y += sdata[0].y/scl_ndim;
    }
}

/*
  This kernel will detect data, accumulate it in frequency and scale it;
  The accumulation here is different from the normal accumulation as we need to put two polarisation togethere here;
 */
template <unsigned int blockSize>
__global__ void detect_faccumulate_scale2_kernel(cufftComplex *dbuf_in, uint8_t *dbuf_out, uint64_t offset_in, uint64_t naccumulate, cufftComplex *offset_scale)
{
  extern volatile __shared__ float scale_sdata[];
  uint64_t i   = threadIdx.x;
  uint64_t tid = i;
  uint64_t loc;
  int loc_freq;

  scale_sdata[tid] = 0;
  while (i < naccumulate)
    {
      loc = blockIdx.x*gridDim.y*naccumulate + blockIdx.y*naccumulate + i;
      scale_sdata[tid] += (dbuf_in[loc].x*dbuf_in[loc].x +
			   dbuf_in[loc].y*dbuf_in[loc].y +
			   dbuf_in[loc + offset_in].x*dbuf_in[loc + offset_in].x +
			   dbuf_in[loc + offset_in].y*dbuf_in[loc + offset_in].y);			   
      i += blockSize;
    }  
  __syncthreads();

  if (blockSize >= 1024)
    {
      if (tid < 512)
	scale_sdata[tid] += scale_sdata[tid + 512];
    }
  __syncthreads();
  
  if (blockSize >= 512)
    {
      if (tid < 256)
	scale_sdata[tid] += scale_sdata[tid + 256];
    }
  __syncthreads();

  if (blockSize >= 256)
    {
      if (tid < 128)
	scale_sdata[tid] += scale_sdata[tid + 128];
    }
  __syncthreads();

  if (blockSize >= 128)
    {
      if (tid < 64)
	scale_sdata[tid] += scale_sdata[tid + 64];
    }
  __syncthreads();
    
  if (tid < 32)
    {
      if (blockSize >= 64)
	{
	  if (tid < 32)
	    scale_sdata[tid] += scale_sdata[tid + 32];
	}
      if (blockSize >= 32)
	{
	  if (tid < 16)
	    scale_sdata[tid] += scale_sdata[tid + 16];
	}
      if (blockSize >= 16)
	{
	  if (tid < 8)
	    scale_sdata[tid] += scale_sdata[tid + 8];
	}
      if (blockSize >= 8)
	{
	  if (tid < 4)
	    scale_sdata[tid] += scale_sdata[tid + 4];
	}
      if (blockSize >= 4)
	{
	  if (tid < 2)
	    scale_sdata[tid] += scale_sdata[tid + 2];
	}
      if (blockSize >= 2)
	{
	  if (tid < 1)
	    scale_sdata[tid] += scale_sdata[tid + 1];
	}
    }
  
  if (tid == 0)
    {
      loc_freq = blockIdx.y;

      if(offset_scale[loc_freq].y == 0.0)
	//dbuf_out[blockIdx.x * gridDim.y + blockIdx.y] = __float2uint_rz(scale_sdata[0]);
	dbuf_out[blockIdx.x * gridDim.y + gridDim.y - blockIdx.y - 1] = __float2uint_rz(scale_sdata[0]); // Reverse frequency order
      else
	//dbuf_out[blockIdx.x * gridDim.y + blockIdx.y] = __float2uint_rz((scale_sdata[0] - offset_scale[loc_freq].x) / offset_scale[loc_freq].y + OFFS_UINT8);
	dbuf_out[blockIdx.x * gridDim.y + gridDim.y - blockIdx.y - 1] = __float2uint_rz((scale_sdata[0] - offset_scale[loc_freq].x) / offset_scale[loc_freq].y + OFFS_UINT8); // Reverse frequency order
    }
}


/*
  This kernel will detect data, accumulate it in frequency and scale it;
  The accumulation here is different from the normal accumulation as we need to put two polarisation togethere here;
 */
template <unsigned int blockSize>
__global__ void detect_faccumulate_scale2_spectral_faccumulate_kernel(cufftComplex *dbuf_in, uint8_t *dbuf_out1, float *dbuf_out2, uint64_t offset_in, uint64_t offset_out, uint64_t naccumulate, cufftComplex *offset_scale)
{
  extern volatile __shared__ float scale_sdata[];
  uint64_t i   = threadIdx.x, j;
  uint64_t tid = threadIdx.x;
  uint64_t loc;
  int loc_freq;
  float aa, bb, u, v;
    
  for(j = 0 ; j < NDATA_PER_SAMP_RT; j ++)
    scale_sdata[tid + j*blockDim.x] = 0;
  
  while (i < naccumulate)
    {
      loc = blockIdx.x*gridDim.y*naccumulate + blockIdx.y*naccumulate + i;
      aa = dbuf_in[loc].x*dbuf_in[loc].x + dbuf_in[loc].y*dbuf_in[loc].y;
      bb = dbuf_in[loc + offset_in].x*dbuf_in[loc + offset_in].x + dbuf_in[loc + offset_in].y*dbuf_in[loc + offset_in].y;
      
      u = 2 * (dbuf_in[loc].x * dbuf_in[loc + offset_in].x + dbuf_in[loc].y * dbuf_in[loc + offset_in].y);
      v = 2 * (dbuf_in[loc].x * dbuf_in[loc + offset_in].y - dbuf_in[loc + offset_in].x * dbuf_in[loc].y);
      // v = 2 * (dbuf_in[loc + offset_in].x * dbuf_in[loc].y - dbuf_in[loc].x * dbuf_in[loc + offset_in].y);
      
      scale_sdata[tid] += (aa + bb);
      scale_sdata[tid + blockDim.x] += (aa - bb);
      scale_sdata[tid + blockDim.x*2] += u;
      scale_sdata[tid + blockDim.x*3] += v;
      scale_sdata[tid + blockDim.x*4] += aa;
      scale_sdata[tid + blockDim.x*5] += bb;
      
      i += blockSize;
    }  
  __syncthreads();

  if (blockSize >= 1024)
    {
      if (tid < 512)
	{
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    scale_sdata[tid + j*blockDim.x] += scale_sdata[tid + j*blockDim.x + 512];
	}
    }
  __syncthreads();
  
  if (blockSize >= 512)
    {
      if (tid < 256)
	{
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    scale_sdata[tid + j*blockDim.x] += scale_sdata[tid + j*blockDim.x + 256];
	}
    }
  __syncthreads();

  if (blockSize >= 256)
    {
      if (tid < 128)
	{
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    scale_sdata[tid + j*blockDim.x] += scale_sdata[tid + j*blockDim.x + 128];
	}
    }
  __syncthreads();

  if (blockSize >= 128)
    {
      if (tid < 64)
	{
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    scale_sdata[tid + j*blockDim.x] += scale_sdata[tid + j*blockDim.x + 64];
	}
    }
  __syncthreads();
    
  if (tid < 32)
    {
      if (blockSize >= 64)
	{
	  if (tid < 32)
	    {
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		scale_sdata[tid + j*blockDim.x] += scale_sdata[tid + j*blockDim.x + 32];
	    }
	}
      if (blockSize >= 32)
	{
	  if (tid < 16)
	    {
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		scale_sdata[tid + j*blockDim.x] += scale_sdata[tid + j*blockDim.x + 16];
	    }
	}
      if (blockSize >= 16)
	{
	  if (tid < 8)
	    {
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		scale_sdata[tid + j*blockDim.x] += scale_sdata[tid + j*blockDim.x + 8];
	    }
	}
      if (blockSize >= 8)
	{
	  if (tid < 4)
	    {
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		scale_sdata[tid + j*blockDim.x] += scale_sdata[tid + j*blockDim.x + 4];
	    }
	}
      if (blockSize >= 4)
	{
	  if (tid < 2)
	    {
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		scale_sdata[tid + j*blockDim.x] += scale_sdata[tid + j*blockDim.x + 2];
	    }
	}
      if (blockSize >= 2)
	{
	  if (tid < 1)
	    {
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		scale_sdata[tid + j*blockDim.x] += scale_sdata[tid + j*blockDim.x + 1];
	    }
	}
    }
  
  if (tid == 0)
    {
      loc_freq = blockIdx.y;

      if(offset_scale[loc_freq].y == 0.0)
	//dbuf_out1[blockIdx.x * gridDim.y + blockIdx.y] = __float2uint_rz(scale_sdata[0]);
	dbuf_out1[blockIdx.x * gridDim.y + gridDim.y - blockIdx.y - 1] = __float2uint_rz(scale_sdata[0]); // Reverse frequency order
      else
	//dbuf_out1[blockIdx.x * gridDim.y + blockIdx.y] = __float2uint_rz((scale_sdata[0] - offset_scale[loc_freq].x) / offset_scale[loc_freq].y + OFFS_UINT8);
	dbuf_out1[blockIdx.x * gridDim.y + gridDim.y - blockIdx.y - 1] = __float2uint_rz((scale_sdata[0] - offset_scale[loc_freq].x) / offset_scale[loc_freq].y + OFFS_UINT8); // Reverse frequency order
      
      for(j = 0; j < NDATA_PER_SAMP_RT; j++)	
	dbuf_out2[blockIdx.x * gridDim.y + blockIdx.y + j*offset_out] = scale_sdata[j*blockDim.x];
    }
}

template <unsigned int blockSize>
__global__ void taccumulate_float_kernel(float *dbuf_in, float *dbuf_out, uint64_t offset_in, uint64_t offset_out, uint64_t naccumulate)
{  
  extern volatile __shared__ float float_sdata[];
  int j;
  uint64_t i   = threadIdx.x;
  uint64_t tid = i;
  uint64_t loc;

  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
    float_sdata[tid + j*blockDim.x] = 0;
  
  while (i < naccumulate)
    { 
      loc = blockIdx.x*naccumulate + i;
      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	float_sdata[tid + j*blockDim.x] += dbuf_in[loc + j * offset_in];
      
      i += blockSize;
    }
  __syncthreads();
  
  if (blockSize >= 1024)
    {
      if (tid < 512)
	{
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    float_sdata[tid + j*blockDim.x] += float_sdata[tid + j*blockDim.x + 512];
	}
    }
  __syncthreads();
  
  if (blockSize >= 512)
    {
      if (tid < 256)
	{
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    float_sdata[tid + j*blockDim.x] += float_sdata[tid + j*blockDim.x + 256];
	}
    }
  __syncthreads();

  if (blockSize >= 256)
    {
      if (tid < 128)
	{
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    float_sdata[tid + j*blockDim.x] += float_sdata[tid + j*blockDim.x + 128];
	}
    }
  __syncthreads();

  if (blockSize >= 128)
    {
      if (tid < 64)
	{
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    float_sdata[tid + j*blockDim.x] += float_sdata[tid + j*blockDim.x + 64];
	}
    }
  __syncthreads();
    
  if (tid < 32)
    {
      if (blockSize >= 64)
	{
	  if (tid < 32)
	    {
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		float_sdata[tid + j*blockDim.x] += float_sdata[tid + j*blockDim.x + 32];
	    }
	}
      if (blockSize >= 32)
	{
	  if (tid < 16)
	    {
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		float_sdata[tid + j*blockDim.x] += float_sdata[tid + j*blockDim.x + 16];
	    }
	}
      if (blockSize >= 16)
	{
	  if (tid < 8)
	    {
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		float_sdata[tid + j*blockDim.x] += float_sdata[tid + j*blockDim.x + 8];
	    }
	}
      if (blockSize >= 8)
	{
	  if (tid < 4)
	    {
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		float_sdata[tid + j*blockDim.x] += float_sdata[tid + j*blockDim.x + 4];
	    }
	}
      if (blockSize >= 4)
	{
	  if (tid < 2)
	    {
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		float_sdata[tid + j*blockDim.x] += float_sdata[tid + j*blockDim.x + 2];
	    }
	}
      if (blockSize >= 2)
	{
	  if (tid < 1)
	    {
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		float_sdata[tid + j*blockDim.x] += float_sdata[tid + j*blockDim.x + 1];
	    }
	}
    }
  
  if (tid == 0)
    {
      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	dbuf_out[blockIdx.x + j*offset_out] = float_sdata[j*blockDim.x];
    }
}

/*
  This kernel will detect data, accumulate it in frequency;
  The accumulation here is different from the normal accumulation as we need to put two polarisation togethere here;
 */
template <unsigned int blockSize>
__global__ void detect_faccumulate_pad_transpose1_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t offset_in, uint64_t naccumulate)
{
  extern volatile __shared__ float scale_sdata[];
  uint64_t i   = threadIdx.x;
  uint64_t tid = i;
  uint64_t loc;

  scale_sdata[tid] = 0;
  while (i < naccumulate)
    {
      loc = blockIdx.x*gridDim.y*naccumulate + blockIdx.y*naccumulate + i;
      scale_sdata[tid] += (dbuf_in[loc].x*dbuf_in[loc].x +
			   dbuf_in[loc].y*dbuf_in[loc].y +
			   dbuf_in[loc + offset_in].x*dbuf_in[loc + offset_in].x +
			   dbuf_in[loc + offset_in].y*dbuf_in[loc + offset_in].y);			   
      i += blockSize;
    }  
  __syncthreads();

  if (blockSize >= 1024)
    {
      if (tid < 512)
	scale_sdata[tid] += scale_sdata[tid + 512];
    }
  __syncthreads();
  
  if (blockSize >= 512)
    {
      if (tid < 256)
	scale_sdata[tid] += scale_sdata[tid + 256];
    }
  __syncthreads();

  if (blockSize >= 256)
    {
      if (tid < 128)
	scale_sdata[tid] += scale_sdata[tid + 128];
    }
  __syncthreads();

  if (blockSize >= 128)
    {
      if (tid < 64)
	scale_sdata[tid] += scale_sdata[tid + 64];
    }
  __syncthreads();
    
  if (tid < 32)
    {
      if (blockSize >= 64)
	{
	  if (tid < 32)
	    scale_sdata[tid] += scale_sdata[tid + 32];
	}
      if (blockSize >= 32)
	{
	  if (tid < 16)
	    scale_sdata[tid] += scale_sdata[tid + 16];
	}
      if (blockSize >= 16)
	{
	  if (tid < 8)
	    scale_sdata[tid] += scale_sdata[tid + 8];
	}
      if (blockSize >= 8)
	{
	  if (tid < 4)
	    scale_sdata[tid] += scale_sdata[tid + 4];
	}
      if (blockSize >= 4)
	{
	  if (tid < 2)
	    scale_sdata[tid] += scale_sdata[tid + 2];
	}
      if (blockSize >= 2)
	{
	  if (tid < 1)
	    scale_sdata[tid] += scale_sdata[tid + 1];
	}
    }
  
  if (tid == 0)
    {      
      dbuf_out[blockIdx.y * gridDim.x + blockIdx.x].x = scale_sdata[tid];
      dbuf_out[blockIdx.y * gridDim.x + blockIdx.x].y = scale_sdata[tid]*scale_sdata[tid];
    }
}

/*
  This kernel take PFT order data with cufftComplex,
  calculate the spctrum according of all types;
  accumulate it in T and the final output will be PFT;
*/
template <unsigned int blockSize>
__global__ void spectral_taccumulate_kernel(cufftComplex *dbuf_in, float *dbuf_out, uint64_t offset_in, uint64_t offset_out, uint64_t naccumulate)
{
  extern volatile __shared__ float spectral_sdata[];
  uint64_t i = threadIdx.x, j;
  uint64_t tid = i;
  uint64_t loc;
  float aa, bb, u, v;
  
  for(j = 0 ; j < NDATA_PER_SAMP_RT; j ++)
    spectral_sdata[tid + j*blockDim.x] = 0;

  while (i < naccumulate)
    {
      loc = blockIdx.x*gridDim.y*naccumulate + blockIdx.y*naccumulate + i;
      aa = dbuf_in[loc].x*dbuf_in[loc].x + dbuf_in[loc].y*dbuf_in[loc].y;
      bb = dbuf_in[loc + offset_in].x*dbuf_in[loc + offset_in].x + dbuf_in[loc + offset_in].y*dbuf_in[loc + offset_in].y;
      
      u = 2 * (dbuf_in[loc].x * dbuf_in[loc + offset_in].x + dbuf_in[loc].y * dbuf_in[loc + offset_in].y);
      v = 2 * (dbuf_in[loc].x * dbuf_in[loc + offset_in].y - dbuf_in[loc + offset_in].x * dbuf_in[loc].y);
      // v = 2 * (dbuf_in[loc + offset_in].x * dbuf_in[loc].y - dbuf_in[loc].x * dbuf_in[loc + offset_in].y);
      
      spectral_sdata[tid] += (aa + bb);
      spectral_sdata[tid + blockDim.x] += (aa - bb);
      spectral_sdata[tid + blockDim.x*2] += u;
      spectral_sdata[tid + blockDim.x*3] += v;
      spectral_sdata[tid + blockDim.x*4] += aa;
      spectral_sdata[tid + blockDim.x*5] += bb;
  
      i += blockSize;
    }  
  __syncthreads();
  
  if (blockSize >= 1024)
    {
      if (tid < 512)
	{
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 512];
	}
    }
  __syncthreads();
  
  if (blockSize >= 512)
    {
      if (tid < 256)
	{	  
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 256];
	}
    }
  __syncthreads();

  if (blockSize >= 256)
    {
      if (tid < 128)
	{
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 128];
	}
    }
  __syncthreads();

  if (blockSize >= 128)
    {
      if (tid < 64)
	{
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 64];
	}
    }
  __syncthreads();
    
  if (tid < 32)
    {
      if (blockSize >= 64)
	{
	  if (tid < 32)
	    {	      
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 32];
	    }
	}
      if (blockSize >= 32)
	{
	  if (tid < 16)
	    {	      
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 16];
	    }
	}
      if (blockSize >= 16)
	{
	  if (tid < 8)
	    {	      
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 8];
	    }
	}
      if (blockSize >= 8)
	{
	  if (tid < 4)
	    {     
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 4];
	    }
	}
      if (blockSize >= 4)
	{
	  if (tid < 2)
	    {	      
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 2];
	    }
	}
      if (blockSize >= 2)
	{
	  if (tid < 1)
	    {  
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 1];
	    }
	}
    }
  
  if (tid == 0)
    {
      for(j = 0; j < NDATA_PER_SAMP_RT; j++)	
	dbuf_out[blockIdx.x * gridDim.y + blockIdx.y + j*offset_out] += spectral_sdata[j*blockDim.x];
    }
}

/*
  This kernel take PFT order data with cufftComplex,
  calculate the spctrum according of all types;
  accumulate it in T and the final output will be PFT;
the same as spectral_taccumulate_kernel, but it only accumulate with current execution
*/
template <unsigned int blockSize>
__global__ void spectral_taccumulate_fold_kernel(cufftComplex *dbuf_in, float *dbuf_out, uint64_t offset_in, uint64_t offset_out, uint64_t naccumulate)
{
  extern volatile __shared__ float spectral_sdata[];
  uint64_t i = threadIdx.x, j;
  uint64_t tid = i;
  uint64_t loc;
  float aa, bb, u, v;
  
  for(j = 0 ; j < NDATA_PER_SAMP_RT; j ++)
    spectral_sdata[tid + j*blockDim.x] = 0;

  while (i < naccumulate)
    {
      loc = blockIdx.x*gridDim.y*naccumulate + blockIdx.y*naccumulate + i;
      aa = dbuf_in[loc].x*dbuf_in[loc].x + dbuf_in[loc].y*dbuf_in[loc].y;
      bb = dbuf_in[loc + offset_in].x*dbuf_in[loc + offset_in].x + dbuf_in[loc + offset_in].y*dbuf_in[loc + offset_in].y;
      
      u = 2 * (dbuf_in[loc].x * dbuf_in[loc + offset_in].x + dbuf_in[loc].y * dbuf_in[loc + offset_in].y);
      v = 2 * (dbuf_in[loc].x * dbuf_in[loc + offset_in].y - dbuf_in[loc + offset_in].x * dbuf_in[loc].y);
      // v = 2 * (dbuf_in[loc + offset_in].x * dbuf_in[loc].y - dbuf_in[loc].x * dbuf_in[loc + offset_in].y);
      
      spectral_sdata[tid] += (aa + bb);
      spectral_sdata[tid + blockDim.x] += (aa - bb);
      spectral_sdata[tid + blockDim.x*2] += u;
      spectral_sdata[tid + blockDim.x*3] += v;
      spectral_sdata[tid + blockDim.x*4] += aa;
      spectral_sdata[tid + blockDim.x*5] += bb;
  
      i += blockSize;
    }  
  __syncthreads();
  
  if (blockSize >= 1024)
    {
      if (tid < 512)
	{
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 512];
	}
    }
  __syncthreads();
  
  if (blockSize >= 512)
    {
      if (tid < 256)
	{	  
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 256];
	}
    }
  __syncthreads();

  if (blockSize >= 256)
    {
      if (tid < 128)
	{
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 128];
	}
    }
  __syncthreads();

  if (blockSize >= 128)
    {
      if (tid < 64)
	{
	  for(j = 0; j < NDATA_PER_SAMP_RT; j++)
	    spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 64];
	}
    }
  __syncthreads();
    
  if (tid < 32)
    {
      if (blockSize >= 64)
	{
	  if (tid < 32)
	    {	      
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 32];
	    }
	}
      if (blockSize >= 32)
	{
	  if (tid < 16)
	    {	      
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 16];
	    }
	}
      if (blockSize >= 16)
	{
	  if (tid < 8)
	    {	      
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 8];
	    }
	}
      if (blockSize >= 8)
	{
	  if (tid < 4)
	    {     
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 4];
	    }
	}
      if (blockSize >= 4)
	{
	  if (tid < 2)
	    {	      
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 2];
	    }
	}
      if (blockSize >= 2)
	{
	  if (tid < 1)
	    {  
	      for(j = 0; j < NDATA_PER_SAMP_RT; j++)
		spectral_sdata[tid + j*blockDim.x] += spectral_sdata[tid + j*blockDim.x + 1];
	    }
	}
    }
  
  if (tid == 0)
    {
      for(j = 0; j < NDATA_PER_SAMP_RT; j++)	
	dbuf_out[blockIdx.x * gridDim.y + blockIdx.y + j*offset_out] = spectral_sdata[j*blockDim.x];
    }
}

// reduce9 + cufftComplex output + unblock
template <unsigned int blockSize>
__global__ void reduce10_kernel(cufftComplex *dbuf_in, cufftComplex *offset_scale, uint64_t naccumulate, float scl_ndim)
{
  extern volatile __shared__ cufftComplex complex_sdata[];
  uint64_t i   = threadIdx.x;
  uint64_t tid = i;
  uint64_t loc; 

  complex_sdata[tid].x = 0;
  complex_sdata[tid].y = 0;
  
  while (i < naccumulate)
    {
      loc = blockIdx.x*gridDim.y*naccumulate + blockIdx.y*naccumulate + i;
      complex_sdata[tid].x += dbuf_in[loc].x;
      complex_sdata[tid].y += dbuf_in[loc].y;
      i += blockSize;
    }
  __syncthreads();
  
  if (blockSize >= 1024)
    {
      if (tid < 512)
	{
	  complex_sdata[tid].x += complex_sdata[tid + 512].x;
	  complex_sdata[tid].y += complex_sdata[tid + 512].y;
	}
    }
  __syncthreads();
  
  if (blockSize >= 512)
    {
      if (tid < 256)
	{
	  complex_sdata[tid].x += complex_sdata[tid + 256].x;
	  complex_sdata[tid].y += complex_sdata[tid + 256].y;
	}
    }
  __syncthreads();

  if (blockSize >= 256)
    {
      if (tid < 128)
	{
	  complex_sdata[tid].x += complex_sdata[tid + 128].x;
	  complex_sdata[tid].y += complex_sdata[tid + 128].y;
	}
    }
  __syncthreads();

  if (blockSize >= 128)
    {
      if (tid < 64)
	{
	  complex_sdata[tid].x += complex_sdata[tid + 64].x;
	  complex_sdata[tid].y += complex_sdata[tid + 64].y;
	}
    }
  __syncthreads();
    
  if (tid < 32)
    {
      if (blockSize >= 64)
	{
	  if (tid < 32)
	    {
	      complex_sdata[tid].x += complex_sdata[tid + 32].x;
	      complex_sdata[tid].y += complex_sdata[tid + 32].y;
	    }
	}
      if (blockSize >= 32)
	{
	  if (tid < 16)
	    {
	      complex_sdata[tid].x += complex_sdata[tid + 16].x;
	      complex_sdata[tid].y += complex_sdata[tid + 16].y;
	    }
	}
      if (blockSize >= 16)
	{
	  if (tid < 8)
	    {
	      complex_sdata[tid].x += complex_sdata[tid + 8].x;
	      complex_sdata[tid].y += complex_sdata[tid + 8].y;
	    }
	}
      if (blockSize >= 8)
	{
	  if (tid < 4)
	    {
	      complex_sdata[tid].x += complex_sdata[tid + 4].x;
	      complex_sdata[tid].y += complex_sdata[tid + 4].y;
	    }
	}
      if (blockSize >= 4)
	{
	  if (tid < 2)
	    {
	      complex_sdata[tid].x += complex_sdata[tid + 2].x;
	      complex_sdata[tid].y += complex_sdata[tid + 2].y;
	    }
	}
      if (blockSize >= 2)
	{
	  if (tid < 1)
	    {
	      complex_sdata[tid].x += complex_sdata[tid + 1].x;
	      complex_sdata[tid].y += complex_sdata[tid + 1].y;
	    }
	}
    }
  
  if (tid == 0)
    {
      offset_scale[blockIdx.x * gridDim.y + blockIdx.y].x += complex_sdata[0].x/scl_ndim;
      offset_scale[blockIdx.x * gridDim.y + blockIdx.y].y += complex_sdata[0].y/scl_ndim;
    }
}

#endif