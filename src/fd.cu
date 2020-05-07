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

#define MATRIX_SIZE 10000 // Size of the one dimensional of the matrix
#define NSMP        (MATRIX_SIZE*MATRIX_SIZE)

// y and z in grid to represent i in the fortran code
// x in block and x in grid to represent j in the fortran code
// "good" number picked here so that i do not need to check the boundary in kernel
#define BLOCK_X     100   // proposed block size x, better to be power 2
#define BLOCK_Y     1     // proposed block size y
#define BLOCK_Z     1     // proposed block size y
#define GRID_X      100    // proposed grif size x, better to be power 2
#define GRID_Y      100    // proposed grif size y, better to be power 2
#define GRID_Z      100    // proposed grif size y, better to be power 2

#define NITERATION  100

__global__ void kernel(double *a, double *b)
{
  int i;
  int j;

  // y and z in grid to represent i in the fortran code 
  i = blockIdx.y * gridDim.z +
    blockIdx.z;
  
  // x in block and x in grid to represent j in the fortran code
  j = blockIdx.x * blockDim.x +
    threadIdx.x;  
  
  // Quick and dirty code here
  // The memory access here may not be fully optimized
  if((0 < i)&&(i < (MATRIX_SIZE-1)) && (0 < j)&&(j < (MATRIX_SIZE-1))){    
    b[i*MATRIX_SIZE+j] = a[i*MATRIX_SIZE+j]/2.0 +
      a[(i+1)*MATRIX_SIZE+j]/8.0 +
      a[(i-1)*MATRIX_SIZE+j]/8.0 +
      a[i*MATRIX_SIZE+j+1]/8.0 +
      a[i*MATRIX_SIZE+j-1]/8.0;
  }
}

int main(int argc, char *argv[]){
  int i;
  int j;
  int k;
  int buffer_size = NSMP*sizeof(double);
  
  dim3 block_size;
  dim3 grid_size;
  
  double *a_host = NULL;
  double *b_host = NULL;
  double *a_device = NULL;
  double *b_device = NULL;

  struct timespec start, stop;
  double elapsed_time;
  
  // The size here is not optimized for sure, use this number just to make sure that I do not need to check the boundary inside the kernel
  // It is better to be power of 2, but again quick and dirty code here
  // The threads in one block can not exceed 512 for most platforms, here to be easier, set it to 100
  block_size.x = BLOCK_X;  
  block_size.y = BLOCK_Y;
  block_size.z = BLOCK_Z;

  // The blocks in on grid can be 65536x65536x65536
  // grid_size.y and grid_size.z to represent i in the fortran code
  // grid_size.x and block_size.x to represent j in the fortran code
  grid_size.x = GRID_X;
  grid_size.y = GRID_Y;
  grid_size.z = GRID_Z;

  // Get required memory both in host and device
  CudaSafeCall(cudaMallocHost((void **)&a_host, buffer_size));  // Memory on host for a
  CudaSafeCall(cudaMallocHost((void **)&b_host, buffer_size));  // Memory on host for b
  CudaSafeCall(cudaMemset((void *)a_host, 0,    buffer_size));  // To be safe, set to zeros
  CudaSafeCall(cudaMemset((void *)b_host, 0,    buffer_size));  // To be safe, set to zeros
  
  CudaSafeCall(cudaMalloc((void **)&a_device, buffer_size));   // Memory on device for a
  CudaSafeCall(cudaMalloc((void **)&b_device, buffer_size));   // Memory on device for b
  CudaSafeCall(cudaMemset((void *)a_device, 0, buffer_size));  // To be safe, set to zeros
  CudaSafeCall(cudaMemset((void *)b_device, 0, buffer_size));  // To be safe, set to zeros

  // Prepare initial input 
  for(i = 0; i < MATRIX_SIZE; i++){
    a_host[i*MATRIX_SIZE]                 = 1.0;
    a_host[i*MATRIX_SIZE+MATRIX_SIZE-1]   = 2.0;
    a_host[i]                             = 3.0;
    a_host[(MATRIX_SIZE-1)*MATRIX_SIZE+i] = 4.0;
  }
  memcpy(b_host, a_host, buffer_size); // Copy a to b

  clock_gettime(CLOCK_REALTIME, &start);
  // Do the iteration
  for(i = 0; i < NITERATION; i++){
    // Most strightforward way to do the iteration
    // Qucik and dirty code, can be improved if time premits
    // 1. Copy a and b host data into device
    // 2. Do the calculation
    // 3. Copy the b back to host
    // 4. Copy b host data to a host and return to 1.
    
    CudaSafeCall(cudaMemcpy(a_device, a_host, buffer_size, cudaMemcpyHostToDevice));  // Copy data into device
    CudaSafeCall(cudaMemcpy(b_device, b_host, buffer_size, cudaMemcpyHostToDevice));  // Copy data into device

    kernel<<<grid_size, block_size>>>(a_device, b_device);  
    CudaSafeKernelLaunch(); // No stream, not sync required

    CudaSafeCall(cudaMemcpy(b_host, b_device, buffer_size, cudaMemcpyDeviceToHost)); // Copy data from device, no need to copy a from device to host here
    memcpy(a_host, b_host, buffer_size); // Copy b to a on host
  }

  // Copy a from device to host in the end, now users are free to use the result on host
  CudaSafeCall(cudaMemcpy(a_host, a_device, buffer_size, cudaMemcpyDeviceToHost));
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "Elapse time for GPU version is %f seconds\n", elapsed_time);
  
  // Code to use a, that depends on users, I do not do this part
  /*
    ......
  */

  // interesting to see how long the CPU code will take
  clock_gettime(CLOCK_REALTIME, &start);
  for(k = 0; k < NITERATION; k++){    
    for(i = 1; i < MATRIX_SIZE-1; i++){
      for(j = 1; j < MATRIX_SIZE-1; j++){
	b_host[i*MATRIX_SIZE+j] = a_host[i*MATRIX_SIZE+j]/2.0 +
	  a_host[(i+1)*MATRIX_SIZE+j]/8.0 +
	  a_host[(i-1)*MATRIX_SIZE+j]/8.0 +
	  a_host[i*MATRIX_SIZE+j+1]/8.0 +
	  a_host[i*MATRIX_SIZE+j-1]/8.0;
      }
    }
    memcpy(a_host, b_host, buffer_size); // Copy b to a on host
  }
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "Elapse time for CPU version is %f seconds\n", elapsed_time);
  
  // Free the memory space
  CudaSafeCall(cudaFreeHost(a_host));
  CudaSafeCall(cudaFreeHost(b_host));
  CudaSafeCall(cudaFree(a_device));
  CudaSafeCall(cudaFree(b_device));
  
  return EXIT_SUCCESS;
}
