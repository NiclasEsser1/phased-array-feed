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
	   "detect_faccumulate_pad_transpose_test - Test the detect_faccumulate_pad_transpose kernel \n"
	   "\n"
	   "Usage: detect_faccumulate_pad_transpose_test [options]\n"
	   " -a  Grid size in X, which is number of samples in time\n"
	   " -b  Grid size in Y, which is number of channels\n"
	   " -c  Block size in X\n"
	   " -h  show help\n");
}

// ./detect_faccumulate_pad_transpose_test -a 512 -b 1 -c 512
int main(int argc, char *argv[])
{
  int i, j,l, k, arg;
  int grid_x, grid_y, block_x;
  uint64_t n_accumulate, idx;
  uint64_t nsamp, npol, nout;
  dim3 gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose;
  cufftComplex *g_in = NULL, *data = NULL, *g_out = NULL, *g_result = NULL, *h_result = NULL;
  float accumulate;
  
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
  gridsize_detect_faccumulate_pad_transpose.x  = grid_x;
  gridsize_detect_faccumulate_pad_transpose.y  = grid_y;
  gridsize_detect_faccumulate_pad_transpose.z  = 1;
  blocksize_detect_faccumulate_pad_transpose.x = block_x;
  blocksize_detect_faccumulate_pad_transpose.y = 1;
  blocksize_detect_faccumulate_pad_transpose.z = 1;
  nout                                         = grid_x*grid_y;
  nsamp                                        = nout*n_accumulate;
  npol                                         = NPOL_IN * nsamp;
   
  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&data, npol * sizeof(cufftComplex)));
  CudaSafeCall(cudaMallocHost((void **)&h_result, nout * sizeof(cufftComplex)));
  CudaSafeCall(cudaMallocHost((void **)&g_result, nout * sizeof(cufftComplex)));
  CudaSafeCall(cudaMalloc((void **)&g_out, nout * sizeof(cufftComplex)));
  CudaSafeCall(cudaMalloc((void **)&g_in, npol * sizeof(cufftComplex)));
  
  CudaSafeCall(cudaMemset((void *)h_result, 0, nout * sizeof(cufftComplex)));
  
  /* cauculate on CPU */
  srand(time(NULL));
  for(i = 0; i < grid_x; i ++) // Prepare the input data
    {
      for(j = 0; j < grid_y; j ++)
	{
	  accumulate = 0;
	  for(k = 0; k < n_accumulate; k++)
	    {
	      idx = (i*grid_y + j) * n_accumulate + k;
	      for(l = 0; l < NPOL_IN; l++)
		{
		  data[idx+l*nsamp].x = fabs((float)rand()/(float)(RAND_MAX/(float)MAX_RAND))/100.;
		  data[idx+l*nsamp].y = fabs((float)rand()/(float)(RAND_MAX/(float)MAX_RAND))/100.;
		  accumulate += (data[idx+l*nsamp].x*data[idx+l*nsamp].x + data[idx+l*nsamp].y*data[idx+l*nsamp].y);
		}
 	    }
	  h_result[j*grid_x+i].x += accumulate; 
	  h_result[j*grid_x+i].y += (accumulate*accumulate);
	}
    }
    
  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(g_in, data, npol * sizeof(cufftComplex), cudaMemcpyHostToDevice));
  detect_faccumulate_pad_transpose_kernel<<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE>>>(g_in, g_out, nsamp);
  CHECK_LAUNCH_ERROR();
  CudaSafeCall(cudaMemcpy(g_result, g_out, nout * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
 
  /* Check the result */
  for(i = 0; i < nout; i++)
    fprintf(stdout, "CPU:\t%f\t%f\tGPU:\t%f\t%f\tDifference\t%E\t%E\n", h_result[i].x, h_result[i].y, g_result[i].x, g_result[i].y, (g_result[i].x - h_result[i].x)/h_result[i].x, (g_result[i].y - h_result[i].y)/h_result[i].y);
  
  /* Free buffer */  
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFree(g_out));
  CudaSafeCall(cudaFree(g_in));
  
  return EXIT_SUCCESS;
}