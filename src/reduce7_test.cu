#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>
#include "cudautil.cuh"
#include "kernel.cuh"

#include "constants.h"

extern "C" void usage ()
{
  fprintf (stdout,
	   "reduce7_test - Test the reduce7 kernel \n"
	   "\n"
	   "Usage: reduce7_test [options]\n"
	   " -a  Grid size in X\n"
	   " -b  Grid size in Y\n"
	   " -c  Block size in X\n"
	   " -d  Number of samples to accumulate in each block\n"
	   " -e  Number of streams in use\n"
	   " -h  show help\n");
}

// ./reduce7_test -a 512 -b 1 -c 512 -d 1024 -e 2
int main(int argc, char *argv[])
{
  int i, j, k;
  int arg, nstream;
  int grid_x, grid_y, block_x;
  uint64_t n_accumulate;
  uint64_t len_in, len_out, idx;
  dim3 gridsize_reduce7, blocksize_reduce7;
  float h_total = 0, g_total = 0;
  cufftComplex *h_result = NULL, *g_result = NULL, *data = NULL, *g_in = NULL, *g_out = NULL;
  
  /* Read in parameters, the arguments here have the same name  */
  while((arg=getopt(argc,argv,"a:b:hc:d:e:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);	  

	case 'a':	  
	  if (sscanf (optarg, "%d", &grid_x) != 1)
	    {
	      fprintf (stderr, "Does not get grid_x, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'b':	  
	  if (sscanf (optarg, "%d", &grid_y) != 1)
	    {
	      fprintf (stderr, "Does not get grid_y, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'c':	  
	  if (sscanf (optarg, "%d", &block_x) != 1)
	    {
	      fprintf (stderr, "Does not get block_x, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'd':	  
	  if (sscanf (optarg, "%"SCNu64"", &n_accumulate) != 1)
	    {
	      fprintf (stderr, "Does not get n_accumulate, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  fprintf(stdout, "n_accumulate is %"PRIu64"\n",  n_accumulate);
	  break;
	  
	case 'e':	  
	  if (sscanf (optarg, "%d", &nstream) != 1)
	    {
	      fprintf (stderr, "Does not get nstream, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  fprintf(stdout, "nstream is %d\n",  nstream);
	  break;
	}
    }

  fprintf(stdout, "grid_x is %d, grid_y is %d, block_x is %d, n_accumulate is %"SCNu64" and nstream is %d\n", grid_x, grid_y, block_x, n_accumulate, nstream);
  
  /* Setup size */
  gridsize_reduce7.x  = grid_x;
  gridsize_reduce7.y  = grid_y;
  gridsize_reduce7.z  = 1;
  blocksize_reduce7.x = block_x;
  blocksize_reduce7.y = 1;
  blocksize_reduce7.z = 1;
  len_out             = grid_x*grid_y;
  len_in              = len_out*n_accumulate;

  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&data,     nstream * len_in * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&h_result, len_out * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&g_result, len_out * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMalloc((void **)&g_in,         nstream * len_in * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMalloc((void **)&g_out,        len_out * NBYTE_CUFFT_COMPLEX));

  /* cauculate on CPU */
  srand(time(NULL));
  for(i = 0; i < len_out; i ++)
    {
      h_result[i].x = 0;
      h_result[i].y = 0;
      for(j = 0; j < n_accumulate; j++)
	{
	  idx = i * n_accumulate + j;
	  for(k = 0; k < nstream; k++)
	    {
	      data[idx+k*len_in].x = (float)rand()/(float)(RAND_MAX/(float)MAX_RAND);
	      data[idx+k*len_in].y = (float)rand()/(float)(RAND_MAX/(float)MAX_RAND);
	  
	      h_result[i].x += data[idx+k*len_in].x;
	      h_result[i].y += data[idx+k*len_in].y;
	    }
	}
    }
  
  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(g_in, data, nstream * len_in * NBYTE_CUFFT_COMPLEX, cudaMemcpyHostToDevice));
  switch (blocksize_reduce7.x)
    {
    case 1024:
      reduce7_kernel<1024><<<gridsize_reduce7, blocksize_reduce7, blocksize_reduce7.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, len_in, n_accumulate, nstream);
      break;
      
    case 512:
      reduce7_kernel< 512><<<gridsize_reduce7, blocksize_reduce7, blocksize_reduce7.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, len_in, n_accumulate, nstream);
      break;
      
    case 256:
      reduce7_kernel< 256><<<gridsize_reduce7, blocksize_reduce7, blocksize_reduce7.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, len_in, n_accumulate, nstream);
      break;
      
    case 128:
      reduce7_kernel< 128><<<gridsize_reduce7, blocksize_reduce7, blocksize_reduce7.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, len_in, n_accumulate, nstream);
      break;
      
    case 64:
      reduce7_kernel<  64><<<gridsize_reduce7, blocksize_reduce7, blocksize_reduce7.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, len_in, n_accumulate, nstream);
      break;
      
    case 32:
      reduce7_kernel<  32><<<gridsize_reduce7, blocksize_reduce7, blocksize_reduce7.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, len_in, n_accumulate, nstream);
      break;
      
    case 16:
      reduce7_kernel<  16><<<gridsize_reduce7, blocksize_reduce7, blocksize_reduce7.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, len_in, n_accumulate, nstream);
      break;
      
    case 8:
      reduce7_kernel<   8><<<gridsize_reduce7, blocksize_reduce7, blocksize_reduce7.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, len_in, n_accumulate, nstream);
      break;
      
    case 4:
      reduce7_kernel<   4><<<gridsize_reduce7, blocksize_reduce7, blocksize_reduce7.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, len_in, n_accumulate, nstream);
      break;
      
    case 2:
      reduce7_kernel<   2><<<gridsize_reduce7, blocksize_reduce7, blocksize_reduce7.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, len_in, n_accumulate, nstream);
      break;
      
    case 1:
      reduce7_kernel<   1><<<gridsize_reduce7, blocksize_reduce7, blocksize_reduce7.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, len_in, n_accumulate, nstream);
      break;
    }
  CudaSafeKernelLaunch();
  CudaSafeCall(cudaMemcpy(g_result, g_out, len_out * NBYTE_CUFFT_COMPLEX, cudaMemcpyDeviceToHost));

  /* Check the result */
  for(i = 0; i < len_out; i++)
    {
      h_total += (h_result[i].x + h_result[i].y);
      g_total += (g_result[i].x + g_result[i].y);
    }
  //fprintf(stdout, "%f\t%f\t%E\n", h_total, g_total, (g_total - h_total)/h_total);
  fprintf(stdout, "CPU:\t%f\nGPU:\t%f\n%E\n", h_total, g_total, (g_total - h_total)/h_total);
  
  /* Free buffer */
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  CudaSafeCall(cudaFree(g_in));
  CudaSafeCall(cudaFree(g_out));
  
  return EXIT_SUCCESS;
}