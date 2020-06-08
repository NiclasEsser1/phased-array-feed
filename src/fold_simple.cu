#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

#define CUDA_ERROR_CHECK
#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaSynchronizeCall()  __cudaSynchronizeCall(__FILE__, __LINE__)
#define CudaSafeKernelLaunch()  __CudaSafeKernelLaunch(__FILE__, __LINE__)

inline void __cudaSynchronizeCall(const char *file, const int line);
inline void __cudaSafeKernelLaunch(const char *file, const int line);
inline void __cudaSafeCall(cudaError err, const char *file, const int line);

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
  if (cudaSuccess != err)
    {
      fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
	      file, line, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#endif
  
  return;
}

inline void __cudaSynchronizeCall(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  err = cudaDeviceSynchronize();
  if(cudaSuccess != err)
    {
      fprintf(stderr, "cudaSynchronizeCall() with sync failed at %s:%i : %s\n",
	      file, line, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#endif
  
  return;
}

// Macro to catch CUDA errors in kernel launches
inline void __CudaSafeKernelLaunch(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
  
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err)
    {
      fprintf(stderr, "cudaSynchronizeCall() failed at %s:%i : %s\n",
	      file, line, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
#endif
}

#define NCHAN        256 // Will be threadIdx.x
#define NBIN         256 // blockIdx.x
#define NSMP_BIN     8
#define NPROFILE     1024

__global__ void kernel(float *input, float *profile)
{
  uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  profile[idx] = 0;

  uint64_t i = 0;
  uint64_t j = 0;
  while(i<NPROFILE){
    while(j<NSMP_BIN){
      profile[idx] += input[threadIdx.x +
			    j*blockDim.x +
			    blockIdx.x*NSMP_BIN*blockDim.x +
			    i*gridDim.x*NSMP_BIN*blockDim.x];
      j++;
    }
    i++;
  }
}

int fold(float *input, float *profile){
  for(uint64_t i1=0; i1<NBIN; i1++){
    for(uint64_t j1=0; j1<NCHAN; j1++){
      profile[i1*NCHAN+j1] = 0;
      
      uint64_t i = 0;
      uint64_t j = 0;
      while(i<NPROFILE){
	while(j<NSMP_BIN){
	  profile[i1*NCHAN+j1] += input[j1 +
					j*NCHAN +
					i1*NSMP_BIN*NCHAN +
					i*NBIN*NSMP_BIN*NCHAN];
	  j++;
	}
	i++;
      }
    }
  }
  
  return EXIT_SUCCESS;
}

int main(int argc, char *argv[]){
  uint64_t input_nsmp = NCHAN*NBIN*NSMP_BIN*NPROFILE;
  
  // Get required memory both in host and device
  float *input_h = NULL;
  float *input_d = NULL;
  float *profile_h = NULL;
  float *profile_d = NULL;
  uint64_t input_size   = input_nsmp*sizeof(float);
  uint64_t profile_size = NCHAN*NBIN*sizeof(float);

  CudaSafeCall(cudaMallocHost((void **)&input_h,   input_size));  // Memory on host for a
  CudaSafeCall(cudaMallocHost((void **)&profile_h, profile_size));  // Memory on host for b
  CudaSafeCall(cudaMemset((void *)input_h,   0, input_size));  // To be safe, set to zeros
  CudaSafeCall(cudaMemset((void *)profile_h, 0, profile_size));  // To be safe, set to zeros
  
  CudaSafeCall(cudaMalloc((void **)&input_d,   input_size));  // Memory on host for a
  CudaSafeCall(cudaMalloc((void **)&profile_d, profile_size));  // Memory on host for b
  CudaSafeCall(cudaMemset((void *)input_d,   0, input_size));  // To be safe, set to zeros
  CudaSafeCall(cudaMemset((void *)profile_d, 0, profile_size));  // To be safe, set to zeros
  
  // Prepare initial input 
  for(uint64_t i = 0; i < input_nsmp; i++){
    input_h[i] = 1.0;
  }

  // Do the work
  struct timespec start, stop;
  double elapsed_time;  
  clock_gettime(CLOCK_REALTIME, &start);

  CudaSafeCall(cudaMemcpy(input_d, input_h, input_size, cudaMemcpyHostToDevice));  // Copy data into device
  dim3 block_size = {NCHAN, 1, 1};
  dim3 grid_size = {NBIN, 1, 1};
  kernel<<<grid_size, block_size>>>(input_d, profile_d);  
  CudaSafeKernelLaunch(); 
  CudaSafeCall(cudaMemcpy(profile_h, profile_d, profile_size, cudaMemcpyDeviceToHost)); // Copy data from device, 
  
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "Elapse time for GPU version is %f seconds\n", elapsed_time);

  // interesting to see how long the CPU code will take
  clock_gettime(CLOCK_REALTIME, &start);
  fold(input_h, profile_h);
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "Elapse time for CPU version is %f seconds\n", elapsed_time);
  
  // Free the memory space
  CudaSafeCall(cudaFreeHost(input_h));
  CudaSafeCall(cudaFreeHost(profile_h));
  CudaSafeCall(cudaFree(input_d));
  CudaSafeCall(cudaFree(profile_d));
  
  return EXIT_SUCCESS;
}
