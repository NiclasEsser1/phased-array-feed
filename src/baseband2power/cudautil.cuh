#ifndef _CUDAUTIL_CUH
#define _CUDAUTIL_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <stdio.h>

#define CUDA_ERROR_CHECK
#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaSynchronizeCall()  __cudaSynchronizeCall(__FILE__, __LINE__)
#define CufftSafeCall(err) __cufftSafeCall(err, __FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char *file, const int line);
inline void __cudaSynchronizeCall(const char *file, const int line);


// Define this to turn on error checking
/*
  3.2.9. Error Checking
  All runtime functions return an error code, but for an asynchronous function (see Asynchronous Concurrent Execution), this error code cannot possibly report any of the asynchronous errors that could occur on the device since the function returns before the device has completed the task; the error code only reports errors that occur on the host prior to executing the task, typically related to parameter validation; if an asynchronous error occurs, it will be reported by some subsequent unrelated runtime function call.
  The only way to check for asynchronous errors just after some asynchronous function call is therefore to synchronize just after the call by calling cudaDeviceSynchronize() (or by using any other synchronization mechanisms described in Asynchronous Concurrent Execution) and checking the error code returned by cudaDeviceSynchronize().
  The runtime maintains an error variable for each host thread that is initialized to cudaSuccess and is overwritten by the error code every time an error occurs (be it a parameter validation error or an asynchronous error). cudaPeekAtLastError() returns this variable. cudaGetLastError() returns this variable and resets it to cudaSuccess.
  Kernel launches do not return any error code, so cudaPeekAtLastError() or cudaGetLastError() must be called just after the kernel launch to retrieve any pre-launch errors. To ensure that any error returned by cudaPeekAtLastError() or cudaGetLastError() does not originate from calls prior to the kernel launch, one has to make sure that the runtime error variable is set to cudaSuccess just before the kernel launch, for example, by calling cudaGetLastError() just before the kernel launch. Kernel launches are asynchronous, so to check for asynchronous errors, the application must synchronize in-between the kernel launch and the call to cudaPeekAtLastError() or cudaGetLastError().
  Note that cudaErrorNotReady that may be returned by cudaStreamQuery() and cudaEventQuery() is not considered an error and is therefore not reported by cudaPeekAtLastError() or cudaGetLastError().
  Read more at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ixzz56FXWocQh 
  Follow us: @GPUComputing on Twitter | NVIDIA on Facebook																												*/

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err)
    {
      fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
	      file, line, cudaGetErrorString(err));
      exit(-1);
    }
#endif
  
  return;
}

inline void __cudaSynchronizeCall(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err)
    {
      fprintf(stderr, "cudaSynchronizeCall() failed at %s:%i : %s\n",
	      file, line, cudaGetErrorString(err));
      exit(-1);
    }
  
  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  err = cudaDeviceSynchronize();
  if(cudaSuccess != err)
    {
      fprintf(stderr, "cudaSynchronizeCall() with sync failed at %s:%i : %s\n",
	       file, line, cudaGetErrorString(err));
      exit(-1);
    }
#endif
  
  return;
}

static const char *_cudaGetErrorEnum(cufftResult error)
{
  switch (error)
    {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";
      
    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";
      
    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";
      
    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";
      
    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";
      
    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";
      
    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";
      
    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";
      
    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";
      
    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
    }
  
  return "<unknown>";
}

inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
  if( CUFFT_SUCCESS != err)
    {
      fprintf(stderr, "CUFFT error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, 
	      _cudaGetErrorEnum(err));					
      cudaDeviceReset();
    }
  #endif
}

#define BSWAP_64(x)     (((uint64_t)(x) << 56) |                        \
                         (((uint64_t)(x) << 40) & 0xff000000000000ULL) | \
                         (((uint64_t)(x) << 24) & 0xff0000000000ULL) |  \
                         (((uint64_t)(x) << 8)  & 0xff00000000ULL) |    \
                         (((uint64_t)(x) >> 8)  & 0xff000000ULL) |      \
                         (((uint64_t)(x) >> 24) & 0xff0000ULL) |        \
                         (((uint64_t)(x) >> 40) & 0xff00ULL) |          \
                         ((uint64_t)(x)  >> 56))

#endif
