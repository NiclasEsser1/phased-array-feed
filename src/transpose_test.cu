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
#include <byteswap.h>

#include "cudautil.cuh"
#include "kernel.cuh"
#include "constants.h"


extern "C" void usage ()
{
  fprintf (stdout,
	   "transpose_test - Test the transpose kernel \n"
	   "\n"
	   "Usage: transpose_test [options]\n"
	   " -a  Number of time stamps\n"
	   " -b  Number of frequency channels\n"
	   " -h  show help\n");
}

// ./transpose_test -a 1024 -b 512
int main(int argc, char *argv[])
{
  int nchan, ntime, arg, m, n, i, j, k;
  dim3 blockSize, gridSize;
  float *g_out = NULL, *g_in = NULL;
  float *data = NULL, *g_result = NULL, *h_result = NULL;
  uint64_t ndata, nsamp, idx_in, idx_out;
  
  /* read in parameters */
  while((arg=getopt(argc,argv,"a:hb:")) != -1)
    {      
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);	  

	case 'a':	  
	  if (sscanf (optarg, "%d", &ntime) != 1)
	    {
	      fprintf (stderr, "Does not get ntime, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'b':	  
	  if (sscanf (optarg, "%d", &nchan) != 1)
	    {
	      fprintf (stderr, "Does not get nchan, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	}
    }
  m = ntime;
  n = nchan;
  nsamp = ntime * nchan;
  ndata = nsamp * NDATA_PER_SAMP_RT;
  fprintf(stdout, "ntime is %d, nchan is %d, m is %d, n is %d, nsamp is %"PRIu64" and ndata is %"PRIu64"\n", ntime, nchan, m, n, nsamp, ndata);

  gridSize.x = ceil(n / (double)TILE_DIM);
  gridSize.y = ceil(m / (double)TILE_DIM);
  gridSize.z = 1;
  blockSize.x = TILE_DIM;
  blockSize.y = NROWBLOCK_TRANS;
  blockSize.z = 1;
  fprintf(stdout, "The configuration of the kernel is (%d, %d, %d) and (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);

  /* Create buffers */
  CudaSafeCall(cudaMallocHost((void **)&h_result, ndata * NBYTE_FLOAT));
  CudaSafeCall(cudaMallocHost((void **)&g_result, ndata * NBYTE_FLOAT));
  CudaSafeCall(cudaMallocHost((void **)&data, ndata * NBYTE_FLOAT));
  CudaSafeCall(cudaMalloc((void **)&g_in, ndata * NBYTE_FLOAT));
  CudaSafeCall(cudaMalloc((void **)&g_out, ndata * NBYTE_FLOAT));

  /* Create data on host */
  srand(time(NULL));
  for(i = 0; i < ndata; i ++)
    data[i] = (float)rand()/(float)(RAND_MAX/(float)MAX_RAND);

  /* Transpose with CPU */
  for(i = 0; i < m; i ++)
    {
      for(j = 0; j < n; j++)
	{
	  for(k = 0; k < NDATA_PER_SAMP_RT; k++)
	    {
	      idx_in  = i * n + j + k * nsamp;
	      idx_out = j * m + i + k * nsamp;
	      h_result[idx_out] = data[idx_in];
	    }
	}
    }
  
  /* Tranpose with GPU */  
  CudaSafeCall(cudaMemcpy(g_in, data, ndata * NBYTE_FLOAT, cudaMemcpyHostToDevice));
  transpose_kernel<<<gridSize, blockSize>>>(g_in, g_out, nsamp, n, m);
  CudaSafeKernelLaunch();
  CudaSafeCall(cudaMemcpy(g_result, g_out, ndata * NBYTE_FLOAT, cudaMemcpyDeviceToHost));

  for(i = 0; i < ndata; i++)
    {
      if(h_result[i] != g_result[i])
  	{
  	  fprintf(stdout, "%f\t%f\t%f\n", h_result[i], g_result[i], h_result[i] - g_result[i]);
  	  fflush(stdout);
  	}
    }
  
  /* Delete buffers */
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFree(g_in));
  CudaSafeCall(cudaFree(g_out));
  
  return EXIT_SUCCESS;
}