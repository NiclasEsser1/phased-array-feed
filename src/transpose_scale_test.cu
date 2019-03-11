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

// ./transpose_scale_test -a 48 -b 1024 -c 64
// ./transpose_scale_test -a 33 -b 1024 -c 64

extern "C" void usage ()
{
  fprintf (stdout,
	   "transpose_scale_test - Test the transpose_scale kernel \n"
	   "\n"
	   "Usage: transpose_scale_test [options]\n"
	   " -a  Number of input frequency chunks\n"
	   " -b  Number of packets of each stream per frequency chunk\n"
	   " -c  Number of FFT points\n"
	   " -h  show help\n");
}

int main(int argc, char *argv[])
{
  int arg;
  int i, j, k;
  int nchunk, nchan, nchan_keep_chan;
  int stream_ndf_chk, cufft_nx;
  dim3 grid_size, block_size;
  uint64_t nsamp_in, nsamp_out, npol_in, npol_out, ndim_in, ndim_out, idx_in, idx_out;
  cufftComplex *data = NULL, *g_in = NULL;
  cufftComplex *mean_scale_h = NULL, *mean_scale_d = NULL;
  int8_t *h_result = NULL, *g_result = NULL,  *g_out = NULL;
  
  /* Read in parameters */
  while((arg=getopt(argc,argv,"a:b:hc:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);
	  
	case 'a':	  
	  if (sscanf (optarg, "%d", &nchunk) != 1)
	    {
	      fprintf (stderr, "Could not get nchunk, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
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
  fprintf(stdout, "nchunk is %d, stream_ndf_chk is %d and cufft_nx is %d\n", nchunk, stream_ndf_chk, cufft_nx);

  /* Setup size */
  nchan        = nchunk * NCHAN_PER_CHUNK;
  nchan_keep_chan = cufft_nx / OVER_SAMP_RATE;
  fprintf(stdout, "nchan is %d\n", nchan);
  
  grid_size.x = ceil(nchan_keep_chan / (double)TILE_DIM);  
  grid_size.y = ceil(nchan / (double)TILE_DIM);
  grid_size.z = stream_ndf_chk * NSAMP_DF / cufft_nx;
  block_size.x = TILE_DIM;
  block_size.y = NROWBLOCK_TRANS;
  block_size.z = 1;
  fprintf(stdout, "kernel configuration is (%d, %d, %d) and (%d, %d, %d)\n", grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z);
  
  nsamp_in  = stream_ndf_chk * nchan * NSAMP_DF / OVER_SAMP_RATE;
  nsamp_out = nsamp_in;
  npol_in   = nsamp_in * NPOL_BASEBAND;
  npol_out  = nsamp_out * NPOL_BASEBAND;
  ndim_in   = npol_in * NDIM_BASEBAND;
  ndim_out  = npol_out * NDIM_BASEBAND;
  fprintf(stdout, "%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\n", nsamp_in, nsamp_out, npol_in, npol_out, ndim_in, ndim_out);

  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&mean_scale_h, nchan * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&data,     npol_in * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&h_result, ndim_out * NBYTE_FOLD));
  CudaSafeCall(cudaMallocHost((void **)&g_result, ndim_out * NBYTE_FOLD));  
  CudaSafeCall(cudaMalloc((void **)&g_in,         npol_in * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMalloc((void **)&mean_scale_d, nchan * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMalloc((void **)&g_out,        ndim_out * NBYTE_FOLD));
  
  /* Prepare the data */
  srand(time(NULL));
  for(i = 0; i < nchan; i ++) // Prepare the scale
    {
      mean_scale_h[i].x = fabs((float)RAND_MAX * rand()/(float)(RAND_MAX));
      mean_scale_h[i].y = fabs((float)RAND_MAX * rand()/(float)(RAND_MAX));

      //mean_scale_h[i].x = 1.0;
      //mean_scale_h[i].y = 1.0;
    }
  for(i = 0; i < grid_size.z; i++)
    {
      for(j = 0; j < nchan; j++)
  	{
  	  for(k = 0; k < nchan_keep_chan; k++)
  	    {
  	      idx_in = i * nchan * nchan_keep_chan + j * nchan_keep_chan + k;
  	      data[idx_in].x = rand()*RAND_STD/RAND_MAX;
  	      data[idx_in].y = rand()*RAND_STD/RAND_MAX;
  	      data[idx_in+nsamp_in].x = rand()*RAND_STD/RAND_MAX;
  	      data[idx_in+nsamp_in].y = rand()*RAND_STD/RAND_MAX;
	      
	      idx_out = NPOL_BASEBAND * NDIM_BASEBAND * (i * nchan * nchan_keep_chan + k * nchan + j);
	      if(mean_scale_h[j].y == 0)
		{
		  h_result[idx_out] = truncf(data[idx_in].x);
		  h_result[idx_out + 1] = truncf(data[idx_in].y);
		  h_result[idx_out + 2] = truncf(data[idx_in+nsamp_in].x);
		  h_result[idx_out + 3] = truncf(data[idx_in+nsamp_in].y);
		}
	      else
		{		  
		  h_result[idx_out] = truncf((data[idx_in].x - mean_scale_h[j].x) / mean_scale_h[j].y);
		  h_result[idx_out + 1] = truncf((data[idx_in].y - mean_scale_h[j].x) / mean_scale_h[j].y);
		  h_result[idx_out + 2] = truncf((data[idx_in+nsamp_in].x - mean_scale_h[j].x) / mean_scale_h[j].y);
		  h_result[idx_out + 3] = truncf((data[idx_in+nsamp_in].y - mean_scale_h[j].x) / mean_scale_h[j].y);
		}
	    }
  	}
    }
  
  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(mean_scale_d, mean_scale_h, nchan * NBYTE_CUFFT_COMPLEX, cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(g_in, data, npol_in * NBYTE_CUFFT_COMPLEX, cudaMemcpyHostToDevice));
  transpose_scale_kernel<<<grid_size, block_size>>>(g_in, g_out, nchan_keep_chan, nchan, nsamp_in, mean_scale_d);
  CudaSafeKernelLaunch();  
  CudaSafeCall(cudaMemcpy(g_result, g_out, ndim_out * NBYTE_FOLD, cudaMemcpyDeviceToHost));
  
  /* Check the result */
  for(i = 0; i < ndim_out; i++)
    {      
      if(h_result[i] != g_result[i])
  	{
	  fprintf(stdout, "The result is not right!\n");
	  break;
  	  //fprintf(stdout, "%d\t%d\t%d\n", h_result[i], g_result[i], (h_result[i] - g_result[i]));
  	  //fflush(stdout);
  	}
    }

  for(i = 0; i < 100; i++)
    fprintf(stdout, "%d\t%d\t%d\n", h_result[i], g_result[i], (h_result[i] - g_result[i]));
  
  /* Free buffer */
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFreeHost(mean_scale_h));
  CudaSafeCall(cudaFree(mean_scale_d));
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  CudaSafeCall(cudaFree(g_in));
  CudaSafeCall(cudaFree(g_out));
  
  return EXIT_SUCCESS;
}