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
	   "taccumulate_float_test - Test the taccumulate_float kernel \n"
	   "\n"
	   "Usage: taccumulate_float_test [options]\n"
	   " -a  Grid size in X\n"
	   " -b  Block size in X\n"
	   " -c  Number of samples to accumulate in each block\n"
	   " -h  show help\n");
}

// ./taccumulate_float_test -a 512 -b 512 -c 1024
int main(int argc, char *argv[])
{
  int i, j, k, arg;
  int grid_x, block_x;
  int naccumulate;
  uint64_t len_in, len_out, nsamp_in, nsamp_out, idx_in, idx_out;
  dim3 gridsize, blocksize;
  float h_total = 0, g_total = 0;
  float *h_result = NULL, *g_result = NULL, *data = NULL, *g_in = NULL, *g_out = NULL;
  
  /* Read in parameters, the arguments here have the same name  */
  while((arg=getopt(argc,argv,"a:b:hc:d:")) != -1)
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
	  if (sscanf (optarg, "%d", &block_x) != 1)
	    {
	      fprintf (stderr, "Does not get block_x, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'c':	  
	  if (sscanf (optarg, "%d", &naccumulate) != 1)
	    {
	      fprintf (stderr, "Does not get naccumulate, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  fprintf(stdout, "naccumulate is %d\n",  naccumulate);
	  break;
	}
    }
  fprintf(stdout, "grid_x is %d, block_x is %d and naccumulate is %d\n", grid_x, block_x, naccumulate);
  
  /* Setup size */
  gridsize.x  = grid_x;
  gridsize.y  = 1;
  gridsize.z  = 1;
  blocksize.x = block_x;
  blocksize.y = 1;
  blocksize.z = 1;
  nsamp_in  = grid_x * naccumulate;
  nsamp_out = grid_x;
  len_in    = nsamp_in * NDATA_PER_SAMP_RT;
  len_out   = nsamp_out * NDATA_PER_SAMP_RT;
  fprintf(stdout, "nsamp_in is %"PRIu64", nsamp_out is %"PRIu64", len_in is %"PRIu64" and len_out is %"PRIu64"\n", nsamp_in, nsamp_out, len_in, len_out);
  
  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&data,     len_in * NBYTE_FLOAT));
  CudaSafeCall(cudaMallocHost((void **)&h_result, len_out * NBYTE_FLOAT));
  CudaSafeCall(cudaMallocHost((void **)&g_result, len_out * NBYTE_FLOAT));
  CudaSafeCall(cudaMalloc((void **)&g_in,         len_in * NBYTE_FLOAT));
  CudaSafeCall(cudaMalloc((void **)&g_out,        len_out * NBYTE_FLOAT));

  /* cauculate on CPU */
  srand(time(NULL));
  for(i = 0; i < nsamp_out; i ++)
    {
      h_result[i] = 0;
      for(j = 0; j < naccumulate; j++)
	{
	  for (k = 0; k< NDATA_PER_SAMP_RT; k++)
	    {
	      idx_in = i * naccumulate + j + k * nsamp_in;
	      data[idx_in] = (float)rand()/(float)(RAND_MAX/(float)MAX_RAND);

	      idx_out = i + k * nsamp_out;
	      h_result[idx_out] += data[idx_in];
	    }
	}
    }
  
  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(g_in, data, len_in * NBYTE_FLOAT, cudaMemcpyHostToDevice));
  switch (blocksize.x)
    {
    case 1024:
      taccumulate_float_kernel<1024><<<gridsize, blocksize, blocksize.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;															              
      																              
    case 512:															              
      taccumulate_float_kernel< 512><<<gridsize, blocksize, blocksize.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;															              
      																              
    case 256:															              
      taccumulate_float_kernel< 256><<<gridsize, blocksize, blocksize.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;															              
      																              
    case 128:															              
      taccumulate_float_kernel< 128><<<gridsize, blocksize, blocksize.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;															              
      																              
    case 64:															              
      taccumulate_float_kernel<  64><<<gridsize, blocksize, blocksize.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out, nsamp_in, nsamp_in, naccumulate);
      break;															              
      																              
    case 32:															              
      taccumulate_float_kernel<  32><<<gridsize, blocksize, blocksize.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;															              
      																              
    case 16:															              
      taccumulate_float_kernel<  16><<<gridsize, blocksize, blocksize.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;															              
      																              
    case 8:															              
      taccumulate_float_kernel<   8><<<gridsize, blocksize, blocksize.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;															              
      																              
    case 4:															              
      taccumulate_float_kernel<   4><<<gridsize, blocksize, blocksize.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;															              
      																              
    case 2:															              
      taccumulate_float_kernel<   2><<<gridsize, blocksize, blocksize.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;															              
      																              
    case 1:															              
      taccumulate_float_kernel<   1><<<gridsize, blocksize, blocksize.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;
    }
  CudaSafeKernelLaunch();
  CudaSafeCall(cudaMemcpy(g_result, g_out, len_out * NBYTE_FLOAT, cudaMemcpyDeviceToHost));

  /* Check the result */
  for(i = 0; i < len_out; i++)
    {
      h_total += h_result[i];
      g_total += g_result[i];
    }
  fprintf(stdout, "CPU:\t%f\nGPU:\t%f\n%E\n", h_total, g_total, (g_total - h_total)/h_total);
  
  /* Free buffer */
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  CudaSafeCall(cudaFree(g_in));
  CudaSafeCall(cudaFree(g_out));
  
  return EXIT_SUCCESS;
}