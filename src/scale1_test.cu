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

#define SCL_UINT8            64.0f          // uint8_t, detected samples should vary in 0.5 * range(uint8_t) = 127, to be safe, use 0.25
#define SCL_NSIG             3.0f


extern "C" void usage ()
{
  fprintf (stdout,
	   "scale1_test - Test the scale1 kernel \n"
	   "\n"
	   "Usage: scale1_test [options]\n"
	   " -a  Number of channels\n"
	   " -h  show help\n");
}

// ./scale1_test -a 512
int main(int argc, char *argv[])
{
  int i, arg, nchan;
  float *g_result = NULL, *h_result = NULL, *g_out = NULL;
  float temp;
  dim3 gridsize_scale1, blocksize_scale1;
  cufftComplex *g_in = NULL, *data = NULL;
  
  /* Read in parameters, the arguments here have the same name  */
  while((arg=getopt(argc,argv,"ha:")) != -1)
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
	}
    }
  fprintf(stdout, "nchan is %d\n", nchan);
  
  /* Setup size */
  gridsize_scale1.x = 1;
  gridsize_scale1.y = 1;
  gridsize_scale1.z = 1;
  blocksize_scale1.x = nchan;
  blocksize_scale1.y = 1;
  blocksize_scale1.z = 1;
  fprintf(stdout, "configuration for kernel is (%d, %d, %d) and (%d, %d, %d)", gridsize_scale1.x, gridsize_scale1.y, gridsize_scale1.z, blocksize_scale1.x, blocksize_scale1.y, blocksize_scale1.z);

  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&data, nchan * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&h_result,  nchan * NBYTE_FLOAT));
  CudaSafeCall(cudaMallocHost((void **)&g_result, nchan * NBYTE_FLOAT));
  CudaSafeCall(cudaMalloc((void **)&g_in, nchan * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMalloc((void **)&g_out, nchan * NBYTE_CUFFT_COMPLEX));
  /* prepare data and calculate on CPU*/
  srand(time(NULL));
  for(i = 0; i < nchan; i++)
    {
      data[i].x = rand()*RAND_STD/RAND_MAX;
      temp = rand()*RAND_STD/RAND_MAX;
      data[i].y =  temp * temp;
      while (data[i].y < data[i].x * data[i].x)
	{
	  temp = rand()*RAND_STD/RAND_MAX;
	  data[i].y =  temp * temp;
	}
      h_result[i]  = SCL_NSIG * sqrt(data[i].y - data[i].x * data[i].x)/SCL_UINT8;
    }

  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(g_in, data, nchan * NBYTE_CUFFT_COMPLEX, cudaMemcpyHostToDevice));
  scale1_kernel<<<gridsize_scale1, blocksize_scale1>>>(g_in, g_out, SCL_NSIG, SCL_UINT8);
  CudaSafeKernelLaunch();
  
  CudaSafeCall(cudaMemcpy(g_result, g_out, nchan * NBYTE_FLOAT, cudaMemcpyDeviceToHost));

  /* Check the result */
  for(i = 0; i < nchan; i++)
    fprintf(stdout, "%E\t%f\t%f\n", g_result[i] - h_result[i], g_result[i], h_result[i]);
  
  /* Free memory */
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  
  CudaSafeCall(cudaFree(g_in));
  CudaSafeCall(cudaFree(g_out));
  
  return EXIT_SUCCESS;
}