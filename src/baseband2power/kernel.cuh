#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>

__global__ void unpack_detect_kernel(int64_t *dbuf_in,  float *dbuf_rt1);
__global__ void sum_kernel(float *dbuf_rt1, float *dbuf_rt2);  
#endif