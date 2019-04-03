#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda_runtime.h>
#include <cuda.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>
#include <byteswap.h>

#include "cudautil.cuh"
#include "constants.h"

int main(int argc, char *argv[])
{
  int i;
  float *d_buffer=NULL, *h_buffer = NULL;
  int buffer_size = 100;
  CudaSafeCall(cudaMalloc((void **)&d_buffer, buffer_size * sizeof(float)));
  CudaSafeCall(cudaMallocHost((void **)&h_buffer, buffer_size * sizeof(float)));

  CudaSafeCall(cudaMemset((void *)d_buffer,  0,      sizeof(d_buffer)));
  CudaSafeCall(cudaMemset((void *)h_buffer,  0,      sizeof(h_buffer)));

  srand(time(NULL));
  for (i = 0; i < buffer_size; i++)
    {
      h_buffer[i] = rand()*RAND_STD/RAND_MAX;
      fprintf(stdout, "FITST:\t%d\t%f\n", i, h_buffer[i]);
    }
  
  CudaSafeCall(cudaMemcpy(d_buffer, h_buffer, buffer_size * sizeof(float), cudaMemcpyHostToDevice));

  for (i = 0; i < buffer_size; i++)
    {
      h_buffer[i] = 0.0;
      fprintf(stdout, "ZERO:\t%d\t%f\n", i, h_buffer[i]);
    }

  CudaSafeCall(cudaMemcpy(h_buffer, d_buffer, buffer_size * sizeof(float), cudaMemcpyDeviceToHost));

  for (i = 0; i < buffer_size; i++)
    fprintf(stdout, "DEVICE:\t%d\t%f\n", i, h_buffer[i]);

  CudaSafeCall(cudaMemset((void *)d_buffer,  0,      sizeof(d_buffer)));
  CudaSafeCall(cudaMemcpy(h_buffer, d_buffer, buffer_size * sizeof(float), cudaMemcpyDeviceToHost));

  for (i = 0; i < buffer_size; i++)
    fprintf(stdout, "DEVICE RESET:\t%d\t%f\n", i, h_buffer[i]);

  fprintf(stdout, "%d\n", (int)sizeof(d_buffer));
  
  CudaSafeCall(cudaFree(d_buffer));
  CudaSafeCall(cudaFreeHost(h_buffer));
  
  return EXIT_SUCCESS;
}