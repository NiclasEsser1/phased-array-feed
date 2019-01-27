#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <inttypes.h>

/* For raw data unpack to get ready for forward FFT */
__global__ void unpack_kernel(int64_t *dbuf_in,  cufftComplex *dbuf_out, uint64_t offset_out);

/* Use after forward FFT to get ready for further steps */
__global__ void swap_select_transpose_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t offset_in, uint64_t offset_out, int cufft_nx, int cufft_mod, int nchan_keep_chan, int nchan_keep_band, int nchan_edge);

/* The following 4 kernels are for scale calculation */
__global__ void accumulate_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out);  // Share between fold and search mode
__global__ void mean_kernel(cufftComplex *buf_in, uint64_t offset_in, float *ddat_offs, float *dsquare_mean, int nstream, float scl_ndim); // Share between fold and search mode
//__global__ void scale_kernel(float *ddat_offs, float *dsquare_mean, float *ddat_scl); // Share between fold and search mode
__global__ void scale_kernel(float *ddat_offs, float *dsquare_mean, float *ddat_scl, float scl_nsig, float scl_uint8);
__global__ void scale1_kernel(cufftComplex *mean, float *ddat_scl, float scl_nsig, float scl_uint8);
__global__ void scale2_kernel(cufftComplex *mean_scale, float scl_nsig, float scl_uint8);

/* The following are only for search mode */
__global__ void detect_faccumulate_scale_kernel(cufftComplex *dbuf_in, uint8_t *dbuf_out, uint64_t offset_in, float *ddat_offs, float *ddat_scl);
__global__ void detect_faccumulate_scale_kernel1(cufftComplex *dbuf_in, uint8_t *dbuf_out, uint64_t offset_in, cufftComplex *mean_scale);
__global__ void detect_faccumulate_pad_transpose_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t offset_in);

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
__global__ void reduce6_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t n_accumulate)
{
  extern volatile __shared__ cufftComplex sdata[];
  uint64_t i   = threadIdx.x;
  uint64_t tid = i;
  uint64_t loc; 

  sdata[tid].x = 0;
  sdata[tid].y = 0;
  
  while (i < n_accumulate)
    { 
      loc = blockIdx.x*gridDim.y*n_accumulate + blockIdx.y*n_accumulate + i;
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
__global__ void reduce7_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t offset_in, uint64_t n_accumulate, int nstream)
{
  extern volatile __shared__ cufftComplex sdata[];
  uint64_t i   = threadIdx.x, j;
  uint64_t tid = i;
  uint64_t loc; 

  sdata[tid].x = 0;
  sdata[tid].y = 0;
  
  while (i < n_accumulate)
    {
      for(j = 0; j < nstream; j++)
	{
	  loc = blockIdx.x*gridDim.y*n_accumulate + blockIdx.y*n_accumulate + j * offset_in + i;
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
__global__ void reduce8_kernel(cufftComplex *dbuf_in, float *ddat_offs, float *dsquare_mean, uint64_t offset_in, uint64_t n_accumulate, int nstream, float scl_ndim)
{
  extern volatile __shared__ cufftComplex sdata[];
  int j;
  uint64_t i   = threadIdx.x;
  uint64_t tid = i;
  uint64_t loc; 

  sdata[tid].x = 0;
  sdata[tid].y = 0;
  
  while (i < n_accumulate)
    {
      for(j = 0; j < nstream; j++)
	{
	  loc = blockIdx.x*gridDim.y*n_accumulate + blockIdx.y*n_accumulate + j * offset_in + i;
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
      ddat_offs[blockIdx.x * gridDim.y + blockIdx.y]    += sdata[0].x/scl_ndim;
      dsquare_mean[blockIdx.x * gridDim.y + blockIdx.y] += sdata[0].y/scl_ndim;
    }
}

// reduce8 + cufftComplex output
template <unsigned int blockSize>
__global__ void reduce9_kernel(cufftComplex *dbuf_in, cufftComplex *mean, uint64_t offset_in, uint64_t n_accumulate, int nstream, float scl_ndim)
{
  extern volatile __shared__ cufftComplex sdata[];
  int j;
  uint64_t i   = threadIdx.x;
  uint64_t tid = i;
  uint64_t loc; 

  sdata[tid].x = 0;
  sdata[tid].y = 0;
  
  while (i < n_accumulate)
    {
      for(j = 0; j < nstream; j++)
	{
	  loc = blockIdx.x*gridDim.y*n_accumulate + blockIdx.y*n_accumulate + j * offset_in + i;
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
      mean[blockIdx.x * gridDim.y + blockIdx.y].x += sdata[0].x/scl_ndim;
      mean[blockIdx.x * gridDim.y + blockIdx.y].y += sdata[0].y/scl_ndim;
    }
}

#endif