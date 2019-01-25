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

// ./swap_select_transpose_test -a 48 -b 1024 -c 128 -d 32768
// ./swap_select_transpose_test -a 33 -b 1024 -c 128 -d 24576

extern "C" void usage ()
{
  fprintf (stdout,
	   "swap_select_transpose_test_test - Test the swap_select_transpose_test kernel \n"
	   "\n"
	   "Usage: swap_select_transpose_test_test [options]\n"
	   " -a  Number of input frequency chunks\n"
	   " -b  Number of packets of each stream per frequency chunk\n"
	   " -c  Number of FFT points\n"
	   " -d  Number of fine frequency channels keep for the band\n"
	   " -h  show help\n");
}

int main(int argc, char *argv[])
{
  int arg;
  int i, j, k;
  int remainder;
  int nchk_in, nchan_in;
  int stream_ndf_chk, cufft_nx, cufft_mod;
  int nchan_keep_band, nchan_keep_chan, nchan_edge;
  dim3 grid_size, block_size;
  uint64_t nsamp_in, nsamp_out, npol_in, npol_out, idx_in, idx_out, loc;
  cufftComplex *data = NULL, *h_result = NULL, *g_result = NULL, *g_in = NULL, *g_out = NULL;
  
  /* Read in parameters */
  while((arg=getopt(argc,argv,"a:b:c:hd:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  fprintf (stderr, "no input, which happens at \"%s\", line [%d].\n",  __FILE__, __LINE__);
	  exit(EXIT_FAILURE);
	  
	case 'a':	  
	  if (sscanf (optarg, "%d", &nchk_in) != 1)
	    {
	      fprintf (stderr, "Could not get nchk_in, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'b':	  
	  if (sscanf (optarg, "%d", &stream_ndf_chk) != 1)
	    {
	      fprintf (stderr, "Could not get stream_ndf_chk, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'c':	  
	  if (sscanf (optarg, "%d", &cufft_nx) != 1)
	    {
	      fprintf (stderr, "Could not get cufft_nx, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'd':	  
	  if (sscanf (optarg, "%d", &nchan_keep_band) != 1)
	    {
	      fprintf (stderr, "Could not get nchan_keep_band, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	}
    }
  fprintf(stdout, "nchk_in is %d, stream_ndf_chk is %d, cufft_nx is %d and nchan_keep_band is %d\n", nchk_in, stream_ndf_chk, cufft_nx, nchan_keep_band);

  /* Setup size */
  nchan_in        = nchk_in * NCHAN_CHK;
  nchan_keep_chan = cufft_nx * OSAMP_RATEI;
  nchan_edge      = 0.5 * (nchan_in * nchan_keep_chan - nchan_keep_band);
  cufft_mod       = 0.5 * nchan_keep_chan;
  fprintf(stdout, "nchan_in is %d, nchan_keep_chan is %d, cufft_mod is %d and nchan_edge is %d\n", nchan_in, nchan_keep_chan, cufft_mod, nchan_edge);
    
  grid_size.x = nchan_in;
  grid_size.y = stream_ndf_chk * NSAMP_DF / cufft_nx;
  grid_size.z = 1;  
  block_size.x = cufft_nx;
  block_size.y = 1;
  block_size.z = 1;
  fprintf(stdout, "kernel configuration is (%d, %d, %d) and (%d, %d, %d)\n", grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z);

  nsamp_in  = stream_ndf_chk * nchan_in * NSAMP_DF;
  nsamp_out = stream_ndf_chk * NSAMP_DF / cufft_nx * nchan_keep_band;
  npol_in   = nsamp_in * NPOL_IN;
  npol_out  = nsamp_out * NPOL_IN;
  
  fprintf(stdout, "%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\n", nsamp_in, nsamp_out, npol_in, npol_out);

  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&data,     npol_in * sizeof(cufftComplex)));
  CudaSafeCall(cudaMallocHost((void **)&h_result, npol_out * sizeof(cufftComplex)));
  CudaSafeCall(cudaMallocHost((void **)&g_result, npol_out * sizeof(cufftComplex)));
  CudaSafeCall(cudaMalloc((void **)&g_in,         npol_in * sizeof(cufftComplex)));
  CudaSafeCall(cudaMalloc((void **)&g_out,        npol_out * sizeof(cufftComplex)));

  /* Prepare the data */
  for(i = 0; i < nchan_in; i++)
    {
      for(j = 0; j < stream_ndf_chk * NSAMP_DF / cufft_nx; j++)
	{
	  for(k = 0; k < cufft_nx; k++)
	    {
	      idx_in      = i * stream_ndf_chk * NSAMP_DF + j * cufft_nx + k;
	      data[idx_in].x = (float)rand()/(float)(RAND_MAX/(float)MAX_RAND);
	      data[idx_in].y = (float)rand()/(float)(RAND_MAX/(float)MAX_RAND);
	      data[idx_in+nsamp_in].x = (float)rand()/(float)(RAND_MAX/(float)MAX_RAND);
	      data[idx_in+nsamp_in].y = (float)rand()/(float)(RAND_MAX/(float)MAX_RAND);
	    }
	}
    }

  /* Calculate on CPU */
  for(i = 0; i < nchan_in; i++)
    {
      for(j = 0; j < stream_ndf_chk * NSAMP_DF / cufft_nx; j++)
	{
	  for(k = 0; k < cufft_nx; k++)
	    {
	      remainder = (k + cufft_mod)%cufft_nx;
	      if (remainder<nchan_keep_chan)
		{
		  loc = i * nchan_keep_chan + remainder - nchan_edge;
		  
		  if((loc >= 0) && (loc < nchan_keep_band))
		    {
		      idx_in      = i * stream_ndf_chk * NSAMP_DF + j * cufft_nx + k;
		      idx_out     = j * nchan_keep_band + loc;
		      		      
		      h_result[idx_out].x           = data[idx_in].x;
		      h_result[idx_out].y           = data[idx_in].y;
		      h_result[idx_out+nsamp_out].x = data[idx_in+nsamp_in].x;
		      h_result[idx_out+nsamp_out].y = data[idx_in+nsamp_in].y;
		    }
		}
	    }
	}
    }
  
  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(g_in, data, npol_in * sizeof(cufftComplex), cudaMemcpyHostToDevice));
  swap_select_transpose_kernel<<<grid_size, block_size>>>(g_in, g_out, nsamp_in, nsamp_out, cufft_nx, cufft_mod, nchan_keep_chan, nchan_keep_band, nchan_edge);
  CHECK_LAUNCH_ERROR();
  CudaSafeCall(cudaMemcpy(g_result, g_out, npol_out * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

  /* Check the result */
  for(i = 0; i < nsamp_out; i++)
    {      
      if((h_result[i].x - g_result[i].x) !=0 || (h_result[i].y - g_result[i].y) != 0)
	fprintf(stdout, "%f\t%f\t%f\t%f\t%f\t%f\n", h_result[i].x, g_result[i].x, h_result[i].x - g_result[i].x, h_result[i].y, g_result[i].y, h_result[i].y - g_result[i].y);
      if((h_result[i+nsamp_out].x - g_result[i+nsamp_out].x) !=0 || (h_result[i+nsamp_out].y - g_result[i+nsamp_out].y) !=0)
	fprintf(stdout, "%f\t%f\t%f\t%f\t%f\t%f\n", h_result[i+nsamp_out].x, g_result[i+nsamp_out].x, h_result[i+nsamp_out].x - g_result[i+nsamp_out].x, h_result[i+nsamp_out].y, g_result[i+nsamp_out].y, h_result[i+nsamp_out].y - g_result[i+nsamp_out].y);
    }

  /* Free buffer */
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  CudaSafeCall(cudaFree(g_in));
  CudaSafeCall(cudaFree(g_out));
  
  return EXIT_SUCCESS;
}