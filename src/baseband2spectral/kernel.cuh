#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>

__global__ void unpack_kernel(int64_t *dbuf_in,  cufftComplex *dbuf_rtc, uint64_t offset_rtc);
__global__ void swap_select_transpose_detect_kernel(cufftComplex *dbuf_rtc, float *dbuf_rtf1, uint64_t offset_rtc);
__global__ void sum_kernel(float *dbuf_rtf1, float *dbuf_rtf2);
#endif