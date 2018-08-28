#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>

/* For raw data unpack to get ready for forward FFT */
__global__ void unpack_kernel(int64_t *dbuf_in,  cufftComplex *dbuf_rt1, size_t offset_rt1);

/* Use after forward FFT to get ready for inverse FFT */
__global__ void swap_select_transpose_swap_kernel(cufftComplex *dbuf_rt1, cufftComplex *dbuf_rt2, size_t offset_rt1, size_t offset_rt2);

/* Use after the inverse FFT to get final data product in 8-bits */
__global__ void transpose_scale_kernel(cufftComplex *dbuf_rt2, int8_t *dbuf_out, size_t offset_rt2, float *ddat_offs, float *ddat_scl);

/* Use for float data recording */
__global__ void transpose_float_kernel(cufftComplex *dbuf_rt2, float *dbuf_out, size_t offset_rt2);

/* The following 4 kernels are for scale calculation */
__global__ void transpose_pad_kernel(cufftComplex *dbuf_rt2, size_t offset_rt2, cufftComplex *dbuf_rt1);
__global__ void sum_kernel(cufftComplex *dbuf_rt1, cufftComplex *dbuf_rt2);  
__global__ void mean_kernel(cufftComplex *buf_rt1, size_t offset_rt1, float *ddat_offs, float *dsquare_mean, int nstream, float scl_ndim); 
__global__ void scale_kernel(float *ddat_offs, float *dsquare_mean, float *ddat_scl);

#endif