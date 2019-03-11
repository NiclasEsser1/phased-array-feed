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
	   "reduce10_test - Test the reduce10 kernel \n"
	   "\n"
	   "Usage: reduce10_test [options]\n"
	   " -a  Grid size in X\n"
	   " -b  Grid size in Y\n"
	   " -c  Block size in X\n"
	   " -d  Number of samples to accumulate in each block\n"
	   " -e  Number of samples to average\n"
	   " -h  show help\n");
}

// ./reduce10_test -a 512 -b 1 -c 512 -d 1024 -e 1.0
int main(int argc, char *argv[])
{
  int i, j;
  int arg;
  int grid_x, grid_y, block_x;
  uint64_t n_accumulate;
  uint64_t len_in, len_out, idx;
  dim3 gridsize_reduce10, blocksize_reduce10;
  float h_total = 0, g_total = 0;
  cufftComplex *data = NULL, *g_in = NULL, *g_out = NULL, *g_result = NULL, *h_result = NULL;
  float scl_ndim;
  
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
	  if (sscanf (optarg, "%f", &scl_ndim) != 1)
	    {
	      fprintf (stderr, "Does not get scl_ndim, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  fprintf(stdout, "scl_ndim is %f\n",  scl_ndim);
	  break;
	}
    }

  fprintf(stdout, "grid_x is %d, grid_y is %d, block_x is %d, n_accumulate is %"SCNu64" and scl_ndim is %f\n", grid_x, grid_y, block_x, n_accumulate, scl_ndim);
  
  /* Setup size */
  gridsize_reduce10.x  = grid_x;
  gridsize_reduce10.y  = grid_y;
  gridsize_reduce10.z  = 1;
  blocksize_reduce10.x = block_x;
  blocksize_reduce10.y = 1;
  blocksize_reduce10.z = 1;
  len_out             = grid_x*grid_y;
  len_in              = len_out*n_accumulate;

  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&data, len_in * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMalloc((void **)&g_in,     len_in * NBYTE_CUFFT_COMPLEX));
  
  CudaSafeCall(cudaMallocHost((void **)&h_result, len_out * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&g_result, len_out * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMalloc((void **)&g_out,        len_out * NBYTE_CUFFT_COMPLEX));
  
  CudaSafeCall(cudaMemset((void *)g_out,  0,      sizeof(g_out)));
  CudaSafeCall(cudaMemset((void *)h_result, 0,    sizeof(h_result)));
  
  /* cauculate on CPU */
  srand(time(NULL));
  for(i = 0; i < len_out; i ++)
    {
      h_result[i].x = 0;
      h_result[i].y= 0;
      for(j = 0; j < n_accumulate; j++)
	{
	  idx = i * n_accumulate + j;
	  data[idx].x = rand()*RAND_STD/RAND_MAX;
	  data[idx].y = rand()*RAND_STD/RAND_MAX;
	  
	  h_result[i].x += (data[idx].x/scl_ndim);
	  h_result[i].y += (data[idx].y/scl_ndim);
	}
    }
  
  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(g_in, data, len_in * NBYTE_CUFFT_COMPLEX, cudaMemcpyHostToDevice));
  switch (blocksize_reduce10.x)
    {
    case 1024:
      reduce10_kernel<1024><<<gridsize_reduce10, blocksize_reduce10, blocksize_reduce10.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, n_accumulate, scl_ndim);
      break;
      
    case 512:
      reduce10_kernel< 512><<<gridsize_reduce10, blocksize_reduce10, blocksize_reduce10.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, n_accumulate, scl_ndim);
      break;
      
    case 256:
      reduce10_kernel< 256><<<gridsize_reduce10, blocksize_reduce10, blocksize_reduce10.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, n_accumulate, scl_ndim);
      break;
      
    case 128:
      reduce10_kernel< 128><<<gridsize_reduce10, blocksize_reduce10, blocksize_reduce10.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, n_accumulate, scl_ndim);
      break;
      
    case 64:
      reduce10_kernel<  64><<<gridsize_reduce10, blocksize_reduce10, blocksize_reduce10.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, n_accumulate, scl_ndim);
      break;
      
    case 32:
      reduce10_kernel<  32><<<gridsize_reduce10, blocksize_reduce10, blocksize_reduce10.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, n_accumulate, scl_ndim);
      break;
      
    case 16:
      reduce10_kernel<  16><<<gridsize_reduce10, blocksize_reduce10, blocksize_reduce10.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, n_accumulate, scl_ndim);
      break;
      
    case 8:
      reduce10_kernel<   8><<<gridsize_reduce10, blocksize_reduce10, blocksize_reduce10.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, n_accumulate, scl_ndim);
      break;
      
    case 4:
      reduce10_kernel<   4><<<gridsize_reduce10, blocksize_reduce10, blocksize_reduce10.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, n_accumulate, scl_ndim);
      break;
      
    case 2:
      reduce10_kernel<   2><<<gridsize_reduce10, blocksize_reduce10, blocksize_reduce10.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, n_accumulate, scl_ndim);
      break;
      
    case 1:
      reduce10_kernel<   1><<<gridsize_reduce10, blocksize_reduce10, blocksize_reduce10.x * NBYTE_CUFFT_COMPLEX>>>(g_in, g_out, n_accumulate, scl_ndim);
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
  CudaSafeCall(cudaFree(g_in));
  CudaSafeCall(cudaFree(g_out));
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  
  return EXIT_SUCCESS;
}