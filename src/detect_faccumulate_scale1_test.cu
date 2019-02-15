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

#define NBYTE   4

extern "C" void usage ()
{
  fprintf (stdout,
	   "detect_faccumulate_scale1_test - Test the detect_faccumulate_scale1 kernel \n"
	   "\n"
	   "Usage: detect_faccumulate_scale_test [options]\n"
	   " -a  Grid size in X, which is number of samples in time\n"
	   " -b  Grid size in Y, which is number of channels\n"
	   " -c  Block size in X\n"
	   " -h  show help\n");
}

// ./detect_faccumulate_scale1_test -a 512 -b 1 -c 512
int main(int argc, char *argv[])
{
  int i, j,l, k, arg;
  int grid_x, grid_y, block_x;
  uint64_t n_accumulate, idx;
  uint64_t nsamp, npol, nout, nchan;
  dim3 gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale;
  float power;
  cufftComplex *mean_scale_h = NULL, *mean_scale_d = NULL, *g_in = NULL, *data = NULL;
  uint8_t *g_out = NULL, *g_result = NULL;
  float *h_result = NULL;
  
  /* Read in parameters, the arguments here have the same name  */
  while((arg=getopt(argc,argv,"a:b:hc:")) != -1)
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
	}
    }
  n_accumulate = 2 * block_x;
  fprintf(stdout, "grid_x is %d, grid_y is %d, block_x is %d and n_accumulate is %"SCNu64"\n", grid_x, grid_y, block_x, n_accumulate);
  
  /* Setup size */
  gridsize_detect_faccumulate_scale.x  = grid_x;
  gridsize_detect_faccumulate_scale.y  = grid_y;
  gridsize_detect_faccumulate_scale.z  = 1;
  blocksize_detect_faccumulate_scale.x = block_x;
  blocksize_detect_faccumulate_scale.y = 1;
  blocksize_detect_faccumulate_scale.z = 1;
  nout                                 = grid_x*grid_y;
  nsamp                                = nout*n_accumulate;
  npol                                 = NPOL_IN * nsamp;
  nchan                                = grid_y;
  
  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&mean_scale_h, nchan * sizeof(cufftComplex)));
  CudaSafeCall(cudaMallocHost((void **)&data,  npol * sizeof(cufftComplex)));
  CudaSafeCall(cudaMallocHost((void **)&h_result,  nout * sizeof(float)));
  CudaSafeCall(cudaMallocHost((void **)&g_result,  nout * sizeof(uint8_t)));
  CudaSafeCall(cudaMalloc((void **)&g_out,  nout * sizeof(uint8_t)));
  CudaSafeCall(cudaMalloc((void **)&mean_scale_d, nchan * sizeof(cufftComplex)));
  CudaSafeCall(cudaMalloc((void **)&g_in,  npol * sizeof(cufftComplex)));  
    
  /* cauculate on CPU */
  srand(time(NULL));
  for(i = 0; i < nchan; i ++) // Prepare the scale
    {
      mean_scale_h[i].x = fabs((float)rand()/(float)(RAND_MAX/(float)MAX_RAND)) + 1 ;
      mean_scale_h[i].y = fabs((float)rand()/(float)(RAND_MAX/(float)MAX_RAND)) * 50000.0;
    }
  for(i = 0; i < grid_x; i ++) // Prepare the input data
    {
      for(j = 0; j < grid_y; j ++)
	{
	  power = 0;
	  h_result[i*grid_y+grid_y-j-1] = 0;
	  for(k = 0; k < n_accumulate; k++)
	    {
	      idx = (i*grid_y + j) * n_accumulate + k;
	      for(l = 0; l < NPOL_IN; l++)
		{
		  data[idx+l*nsamp].x = fabs((float)rand()/(float)(RAND_MAX/(float)MAX_RAND));
		  data[idx+l*nsamp].y = fabs((float)rand()/(float)(RAND_MAX/(float)MAX_RAND));
		  
		  power += (data[idx+l*nsamp].x*data[idx+l*nsamp].x + data[idx+l*nsamp].y*data[idx+l*nsamp].y);
		}
	    }
	  if(mean_scale_h[j].y == 0 )
	    h_result[i*grid_y+grid_y-j-1] = (power);
	  else
	    h_result[i*grid_y+grid_y-j-1] = ((power - mean_scale_h[j].x) / mean_scale_h[j].y + OFFS_UINT8); // Reverse frequency order
	}
    }
    
  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(mean_scale_d, mean_scale_h, nchan * sizeof(cufftComplex), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(g_in, data, npol * sizeof(cufftComplex), cudaMemcpyHostToDevice));
  detect_faccumulate_scale1_kernel<<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * NBYTE>>>(g_in, g_out, nsamp, mean_scale_d);
  CudaSafeKernelLaunch();
  CudaSafeCall(cudaMemcpy(g_result, g_out, nout * sizeof(uint8_t), cudaMemcpyDeviceToHost));
 
  /* Check the result */
  for(i = 0; i < nout; i++)
    fprintf(stdout, "CPU:\t%f\tGPU:\t%d\t%f\n", h_result[i], g_result[i], g_result[i] - floor(h_result[i]));
  
  /* Free buffer */  
  CudaSafeCall(cudaFreeHost(mean_scale_h));
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFree(g_out));
  CudaSafeCall(cudaFree(mean_scale_d));
  CudaSafeCall(cudaFree(g_in));
  
  return EXIT_SUCCESS;
}