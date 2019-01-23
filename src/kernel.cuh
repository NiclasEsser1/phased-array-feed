#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>

/* For raw data unpack to get ready for forward FFT */
__global__ void unpack_kernel(int64_t *dbuf_in,  cufftComplex *dbuf_out, uint64_t offset_out);

/* Use after forward FFT to get ready for further steps */
__global__ void swap_select_transpose_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t offset_in, uint64_t offset_out, int cufft_nx, int cufft_mod, int nchan_keep_chan, int nchan_keep_band, int nchan_edge);

/* The following 4 kernels are for scale calculation */
__global__ void accumulate_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out);  // Share between fold and search mode
__global__ void mean_kernel(cufftComplex *buf_in, uint64_t offset_in, float *ddat_offs, float *dsquare_mean, int nstream, float scl_ndim); // Share between fold and search mode
__global__ void scale_kernel(float *ddat_offs, float *dsquare_mean, float *ddat_scl); // Share between fold and search mode

/* The following are only for search mode */
__global__ void detect_faccumulate_scale_kernel(cufftComplex *dbuf_in, uint8_t *dbuf_out, uint64_t offset_in, float *ddat_offs, float *ddat_scl);
__global__ void detect_faccumulate_pad_transpose_kernel(cufftComplex *dbuf_in, cufftComplex *dbuf_out, uint64_t offset_in);
#endif