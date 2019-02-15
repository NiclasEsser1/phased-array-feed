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
	   "scale2_test - Test the scale2 kernel \n"
	   "\n"
	   "Usage: scale2_test [options]\n"
	   " -a  Number of channels\n"
	   " -h  show help\n");
}

// ./scale2_test -a 512
int main(int argc, char *argv[])
{
  int i, arg, nchan;
  float *h_result = NULL;
  dim3 gridsize_scale2, blocksize_scale2;
  cufftComplex *g = NULL, *data = NULL, *g_result = NULL;
  
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
  gridsize_scale2.x = 1;
  gridsize_scale2.y = 1;
  gridsize_scale2.z = 1;
  blocksize_scale2.x = nchan;
  blocksize_scale2.y = 1;
  blocksize_scale2.z = 1;
  fprintf(stdout, "configuration for kernel is (%d, %d, %d) and (%d, %d, %d)", gridsize_scale2.x, gridsize_scale2.y, gridsize_scale2.z, blocksize_scale2.x, blocksize_scale2.y, blocksize_scale2.z);

  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&data, nchan * sizeof(cufftComplex)));
  CudaSafeCall(cudaMallocHost((void **)&h_result,  nchan * sizeof(float)));
  CudaSafeCall(cudaMallocHost((void **)&g_result, nchan * sizeof(cufftComplex)));
  CudaSafeCall(cudaMalloc((void **)&g, nchan * sizeof(cufftComplex)));
  /* prepare data and calculate on CPU*/
  srand(time(NULL));
  for(i = 0; i < nchan; i++)
    {
      data[i].x = (float)rand()/(float)(RAND_MAX/(float)MAX_RAND);
      data[i].y = (float)rand()/(float)(RAND_MAX/(float)MAX_RAND) * (float)rand()/(float)(RAND_MAX/(float)MAX_RAND);
      while (data[i].y < data[i].x * data[i].x)
	data[i].y = (float)rand()/(float)(RAND_MAX/(float)MAX_RAND) * (float)rand()/(float)(RAND_MAX/(float)MAX_RAND);
      h_result[i]  = SCL_NSIG * sqrt(data[i].y - data[i].x * data[i].x)/SCL_UINT8;
    }

  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(g, data, nchan * sizeof(cufftComplex), cudaMemcpyHostToDevice));
  scale2_kernel<<<gridsize_scale2, blocksize_scale2>>>(g, SCL_NSIG, SCL_UINT8);
  CudaSafeKernelLaunch();
  
  CudaSafeCall(cudaMemcpy(g_result, g, nchan * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

  /* Check the result */
  for(i = 0; i < nchan; i++)
    fprintf(stdout, "%E\t%f\t%f\n", g_result[i].y - h_result[i], g_result[i].y, h_result[i]);
  
  /* Free memory */
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  
  CudaSafeCall(cudaFree(g));
  
  return EXIT_SUCCESS;
}