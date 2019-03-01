#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <math.h>
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
	   "scale3_test - Test the scale3 kernel \n"
	   "\n"
	   "Usage: scale3_test [options]\n"
	   " -a  Number of channels\n"
	   " -b  Number of streams\n"
	   " -h  show help\n");
}

// ./scale3_test -a 512 -b 2
int main(int argc, char *argv[])
{
  int i, j, arg, nchan, nstream;
  float *h_result = NULL;
  float x, y, temp;
  dim3 gridsize_scale3, blocksize_scale3;
  cufftComplex *g = NULL, *data = NULL, *g_result = NULL;
  
  /* Read in parameters, the arguments here have the same name  */
  while((arg=getopt(argc,argv,"ha:b:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);	  

	case 'a':	  
	  if (sscanf (optarg, "%d", &nchan) != 1)
	    {
	      fprintf (stderr, "Does not get nchan, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'b':	  
	  if (sscanf (optarg, "%d", &nstream) != 1)
	    {
	      fprintf (stderr, "Does not get nstream, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	}
    }
  fprintf(stdout, "nchan is %d and nstream is %d\n", nchan, nstream);
  
  /* Setup size */
  gridsize_scale3.x = 1;
  gridsize_scale3.y = 1;
  gridsize_scale3.z = 1;
  blocksize_scale3.x = nchan;
  blocksize_scale3.y = 1;
  blocksize_scale3.z = 1;
  fprintf(stdout, "configuration for kernel is (%d, %d, %d) and (%d, %d, %d)", gridsize_scale3.x, gridsize_scale3.y, gridsize_scale3.z, blocksize_scale3.x, blocksize_scale3.y, blocksize_scale3.z);

  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&data, nstream * nchan * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&h_result,  nstream * nchan * NBYTE_FLOAT));
  CudaSafeCall(cudaMallocHost((void **)&g_result, nstream * nchan * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMalloc((void **)&g, nstream * nchan * NBYTE_CUFFT_COMPLEX));
  /* prepare data and calculate on CPU*/
  srand(time(NULL));
  for(i = 0; i < nchan; i++)
    {
      x = 0;
      y = 0;
      while (y < x * x)
	{
	  for(j = 0; j < nstream; j++)
	    {
	      data[j*nchan+i].x = rand()*RAND_STD/RAND_MAX;
	      temp = rand()*RAND_STD/RAND_MAX;
	      data[j*nchan+i].y = temp * temp;
	      x += data[j*nchan+i].x;
	      y += data[j*nchan+i].y;
	    }
	  x = 0;
	  y = 0;
	}
      h_result[i]  = SCL_NSIG * sqrt(y - x * x)/SCL_UINT8;
    }

  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(g, data, nstream * nchan * NBYTE_CUFFT_COMPLEX, cudaMemcpyHostToDevice));
  scale3_kernel<<<gridsize_scale3, blocksize_scale3>>>(g, nchan, nstream, SCL_NSIG, SCL_UINT8);
  CudaSafeKernelLaunch();
  
  CudaSafeCall(cudaMemcpy(g_result, g, nstream * nchan * NBYTE_CUFFT_COMPLEX, cudaMemcpyDeviceToHost));

  /* Check the result */
  for(i = 0; i < nchan; i++)
    fprintf(stdout, "%E\t%E\t%E\n", g_result[i].y - h_result[i], g_result[i].y, h_result[i]);
  
  /* Free memory */
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));  
  CudaSafeCall(cudaFree(g));
  
  return EXIT_SUCCESS;
}