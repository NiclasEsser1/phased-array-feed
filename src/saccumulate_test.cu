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
#include "baseband2spectral.cuh"

// ./saccumulate_test -a 48 -b 1024 -c 2
// ./saccumulate_test -a 33 -b 1024 -c 2

extern "C" void usage ()
{
  fprintf (stdout,
	   "saccumulate_test - Test the saccumulate kernel \n"
	   "\n"
	   "Usage: saccumulate_test [options]\n"
	   " -a  Number of input frequency chunks\n"
	   " -b  Number of FFT points\n"
	   " -c  Number of streams\n"
	   " -h  show help\n");
}

int main(int argc, char *argv[])
{
  int i, j, k, l;
  int arg, nchk_in, cufft_nx, nchan_in, nstream;
  dim3 grid_size, block_size;
  uint64_t nsamp_in, nsamp_out, npol_in, npol_out, idx_in, idx_out;
  float *data = NULL, *g_in = NULL, *h_result = NULL, *g_result = NULL, *g_out = NULL;
  
  /* Read in parameters */
  while((arg=getopt(argc,argv,"a:b:hc:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);
	  
	case 'a':	  
	  if (sscanf (optarg, "%d", &nchk_in) != 1)
	    {
	      fprintf (stderr, "Could not get nchk_in, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'b':	  
	  if (sscanf (optarg, "%d", &cufft_nx) != 1)
	    {
	      fprintf (stderr, "Could not get cufft_nx, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  	
	case 'c':	  
	  if (sscanf (optarg, "%d", &nstream) != 1)
	    {
	      fprintf (stderr, "Could not get nstream, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	    
	}
    }
  fprintf(stdout, "nchk_in is %d, cufft_nx is %d and nstream is %d\n", nchk_in, cufft_nx, nstream);

  /* Setup size */
  nchan_in = nchk_in * NCHAN_PER_CHUNK;
  grid_size.x = NDATA_PER_SAMP_FULL;
  grid_size.y = cufft_nx / OVER_SAMP_RATE;
  grid_size.z = 1;
  block_size.x = nchan_in;
  block_size.y = 1;
  block_size.z = 1;
  fprintf(stdout, "kernel configuration is (%d, %d, %d) and (%d, %d, %d)\n", grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z);
  
  nsamp_in  = nchan_in * cufft_nx / OVER_SAMP_RATE * nstream;
  nsamp_out = nchan_in * cufft_nx / OVER_SAMP_RATE;  
  npol_in   = nsamp_in * NDATA_PER_SAMP_FULL;
  npol_out  = nsamp_out * NDATA_PER_SAMP_FULL;
  fprintf(stdout, "%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\n", nsamp_in, nsamp_out, npol_in, npol_out);
  
  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&data,     npol_in * NBYTE_FLOAT));
  CudaSafeCall(cudaMallocHost((void **)&h_result, npol_out * NBYTE_FLOAT));
  CudaSafeCall(cudaMallocHost((void **)&g_result, npol_out * NBYTE_FLOAT));
  CudaSafeCall(cudaMalloc((void **)&g_in,         npol_in * NBYTE_FLOAT));
  CudaSafeCall(cudaMemset((void *)g_result, 0,    npol_out * NBYTE_FLOAT));
  //CudaSafeCall(cudaMallocHost((void **)&g_result, npol_out * NBYTE_FLOAT));
  
  /* Prepare the data and calculate on CPU */
  srand(time(NULL));
  for(i = 0; i < NDATA_PER_SAMP_FULL; i++)
    {
      for(j = 0; j < nchan_in; j++)
	{
	  for(k = 0; k < cufft_nx / OVER_SAMP_RATE; k++)
	    {
	      idx_out = i * nchan_in * cufft_nx / OVER_SAMP_RATE + j * cufft_nx/OVER_SAMP_RATE + k;
	      for (l = 0; l < nstream; l++)
		{
		  idx_in = idx_out + l * npol_out;	      
		  data[idx_in] = (float)rand()/(float)(RAND_MAX/(float)MAX_RAND);
		  h_result[idx_out] += data[idx_in];
		}
	    }
	}
    }
  //fprintf(stdout, "%"PRIu64"\t%"PRIu64"\n", idx_out, idx_in);
  
  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(g_in, data, npol_in * NBYTE_FLOAT, cudaMemcpyHostToDevice));
  saccumulate_kernel<<<grid_size, block_size>>>(g_in, npol_out, nstream);  
  CudaSafeKernelLaunch();
  CudaSafeCall(cudaMemcpy(g_result, g_in, npol_out * NBYTE_FLOAT, cudaMemcpyDeviceToHost));

  /* Check the result */
  for(i = 0; i < npol_out; i++)
    {
      if(fabs((h_result[i] - g_result[i])/g_result[i])>1E-3)
  	{
  	  fprintf(stdout, "%d\t%f\t%f\t%E\t", i, h_result[i], g_result[i], (h_result[i] - g_result[i])/g_result[i]);
	  fprintf(stdout, "%d\t%f\t", i, h_result[i] - data[i]);
  	  fprintf(stdout, "\n");
  	}
    }

  /* Free buffer */
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  CudaSafeCall(cudaFree(g_in));
  CudaSafeCall(cudaFree(g_out));
  
  return EXIT_SUCCESS;
}