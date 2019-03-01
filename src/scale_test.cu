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
	   "scale_test - Test the scale kernel \n"
	   "\n"
	   "Usage: scale_test [options]\n"
	   " -a  Number of channels\n"
	   " -h  show help\n");
}

// ./scale_test -a 512
int main(int argc, char *argv[])
{
  int i, arg, nchan;
  float *g_offs = NULL, *g_mean = NULL, *g_result_scl = NULL, *g_out_scl = NULL, *h_offs = NULL, *h_mean = NULL, *h_scl = NULL;
  float temp;
  dim3 gridsize_scale, blocksize_scale;
  
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
  gridsize_scale.x = 1;
  gridsize_scale.y = 1;
  gridsize_scale.z = 1;
  blocksize_scale.x = nchan;
  blocksize_scale.y = 1;
  blocksize_scale.z = 1;
  fprintf(stdout, "configuration for kernel is (%d, %d, %d) and (%d, %d, %d)", gridsize_scale.x, gridsize_scale.y, gridsize_scale.z, blocksize_scale.x, blocksize_scale.y, blocksize_scale.z);

  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&h_offs, nchan * NBYTE_FLOAT));
  CudaSafeCall(cudaMallocHost((void **)&h_mean, nchan * NBYTE_FLOAT));
  CudaSafeCall(cudaMallocHost((void **)&h_scl,  nchan * NBYTE_FLOAT));
  CudaSafeCall(cudaMallocHost((void **)&g_result_scl, nchan * NBYTE_FLOAT));
  
  CudaSafeCall(cudaMalloc((void **)&g_offs, nchan * NBYTE_FLOAT));
  CudaSafeCall(cudaMalloc((void **)&g_mean, nchan * NBYTE_FLOAT));
  CudaSafeCall(cudaMalloc((void **)&g_out_scl, nchan * NBYTE_FLOAT));

  /* prepare data and calculate on CPU*/
  srand(time(NULL));
  for(i = 0; i < nchan; i++)
    {
      h_offs[i] = rand()*RAND_STD/RAND_MAX;
      temp = rand()*RAND_STD/RAND_MAX;
      h_mean[i] = temp * temp;
      while (h_mean[i] < h_offs[i] * h_offs[i])
	{
	  temp = rand()*RAND_STD/RAND_MAX;
	  h_mean[i] = temp * temp;
	}
      h_scl[i]  = SCL_NSIG * sqrt(h_mean[i] - h_offs[i] * h_offs[i])/SCL_UINT8;
    }

  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(g_offs, h_offs, nchan * NBYTE_FLOAT, cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(g_mean, h_mean, nchan * NBYTE_FLOAT, cudaMemcpyHostToDevice));
  scale_kernel<<<gridsize_scale, blocksize_scale>>>(g_offs, g_mean, g_out_scl, SCL_NSIG, SCL_UINT8);
  CudaSafeKernelLaunch();
  
  CudaSafeCall(cudaMemcpy(g_result_scl, g_out_scl, nchan * NBYTE_FLOAT, cudaMemcpyDeviceToHost));

  /* Check the result */
  for(i = 0; i < nchan; i++)
    fprintf(stdout, "%E\t%f\t%f\n", g_result_scl[i] - h_scl[i], g_result_scl[i], h_scl[i]);
  //fprintf(stdout, "%f\t%f\t%f\n", h_offs[i], h_mean[i], h_scl[i]);
  
  /* Free memory */
  CudaSafeCall(cudaFreeHost(h_offs));
  CudaSafeCall(cudaFreeHost(h_mean));
  CudaSafeCall(cudaFreeHost(h_scl));
  CudaSafeCall(cudaFreeHost(g_result_scl));
  
  CudaSafeCall(cudaFree(g_offs));
  CudaSafeCall(cudaFree(g_mean));
  CudaSafeCall(cudaFree(g_out_scl));
  
  return EXIT_SUCCESS;
}