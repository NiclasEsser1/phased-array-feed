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

// ./transpose_complex_test -a 48 -b 1024 -c 64
// ./transpose_complex_test -a 33 -b 1024 -c 64

extern "C" void usage ()
{
  fprintf (stdout,
	   "transpose_complex_test - Test the transpose_complex kernel \n"
	   "\n"
	   "Usage: transpose_complex_test [options]\n"
	   " -a  Number of input frequency chunks\n"
	   " -b  Number of packets of each stream per frequency chunk\n"
	   " -c  Number of FFT points\n"
	   " -h  show help\n");
}

int main(int argc, char *argv[])
{
  int arg;
  int i, j, k;
  int nchk_in, nchan_in;
  int stream_ndf_chk, cufft_nx, cufft_mod;
  int nchan_keep_chan;
  dim3 grid_size, block_size;
  uint64_t nsamp_in, nsamp_out, npol_in, npol_out, idx_in, idx_out;
  cufftComplex *data = NULL, *h_result = NULL, *g_result = NULL, *g_in = NULL, *g_out = NULL;
  
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
	  if (sscanf (optarg, "%d", &stream_ndf_chk) != 1)
	    {
	      fprintf (stderr, "Could not get stream_ndf_chk, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'c':	  
	  if (sscanf (optarg, "%d", &cufft_nx) != 1)
	    {
	      fprintf (stderr, "Could not get cufft_nx, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	}
    }
  fprintf(stdout, "nchk_in is %d, stream_ndf_chk is %d and cufft_nx is %d\n", nchk_in, stream_ndf_chk, cufft_nx);

  /* Setup size */
  nchan_in        = nchk_in * NCHAN_PER_CHUNK;
  nchan_keep_chan = cufft_nx / OVER_SAMP_RATE;
  cufft_mod       = 0.5 * nchan_keep_chan;
  fprintf(stdout, "nchan_in is %d, nchan_keep_chan is %d and cufft_mod is %d\n", nchan_in, nchan_keep_chan, cufft_mod);
    
  grid_size.x = stream_ndf_chk * NSAMP_DF / cufft_nx;
  grid_size.y = nchan_in;
  grid_size.z = 1;  
  block_size.x = nchan_keep_chan;
  block_size.y = 1;
  block_size.z = 1;
  fprintf(stdout, "kernel configuration is (%d, %d, %d) and (%d, %d, %d)\n", grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z);

  nsamp_in  = stream_ndf_chk * nchan_in * NSAMP_DF / OVER_SAMP_RATE;
  nsamp_out = nsamp_in;
  npol_in   = nsamp_in * NPOL_BASEBAND;
  npol_out  = nsamp_out * NPOL_BASEBAND;
  
  fprintf(stdout, "%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\n", nsamp_in, nsamp_out, npol_in, npol_out);

  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&data,     npol_in * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&h_result, npol_out * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&g_result, npol_out * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMalloc((void **)&g_in,         npol_in * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMalloc((void **)&g_out,        npol_out * NBYTE_CUFFT_COMPLEX));

  /* Prepare the data */
  srand(time(NULL));
  for(i = 0; i < grid_size.x; i++)
    {
      for(j = 0; j < grid_size.y; j++)
	{
	  for(k = 0; k < block_size.x; k++)
	    {
	      idx_in = i * grid_size.y * block_size.x + j * block_size.x + k;
	      data[idx_in].x = rand() * RAND_STD/RAND_MAX;
	      data[idx_in].y = rand() * RAND_STD/RAND_MAX;
	      data[idx_in+nsamp_in].x = rand() * RAND_STD/RAND_MAX;
	      data[idx_in+nsamp_in].y = rand() * RAND_STD/RAND_MAX;

	      idx_out = j * grid_size.x * block_size.x + i * block_size.x + k;
	      
	      h_result[idx_out].x   = data[idx_in].x;
	      h_result[idx_out].y   = data[idx_in].y;
	      h_result[idx_out+nsamp_out].x = data[idx_in+nsamp_in].x;
	      h_result[idx_out+nsamp_out].y = data[idx_in+nsamp_in].y;
	    }
	}
    }

  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(g_in, data, npol_in * NBYTE_CUFFT_COMPLEX, cudaMemcpyHostToDevice));
  transpose_complex_kernel<<<grid_size, block_size>>>(g_in, nsamp_in, g_out);
  CudaSafeKernelLaunch();
  CudaSafeCall(cudaMemcpy(g_result, g_out, npol_out * NBYTE_CUFFT_COMPLEX, cudaMemcpyDeviceToHost));

  /* Check the result */
  for(i = 0; i < nsamp_out; i++)
    {      
      if(fabs((h_result[i].x - g_result[i].x)/g_result[i].x) > 1.0E-3 || fabs((h_result[i].y - g_result[i].y)/g_result[i].y) > 1.0E-3)
	fprintf(stdout, "%f\t%f\t%f\t%f\t%f\t%f\n", h_result[i].x, g_result[i].x, h_result[i].x - g_result[i].x, h_result[i].y, g_result[i].y, h_result[i].y - g_result[i].y);
    }

  /* Free buffer */
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  CudaSafeCall(cudaFree(g_in));
  CudaSafeCall(cudaFree(g_out));
  
  return EXIT_SUCCESS;
}