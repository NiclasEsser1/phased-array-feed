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

#define NBYTE 4

// ./spectral_taccumulate_test -a 48 -b 1024 -c 1024
// ./spectral_taccumulate_test -a 33 -b 1024 -c 1024

extern "C" void usage ()
{
  fprintf (stdout,
	   "spectral_taccumulate_test - Test the spectral_taccumulate kernel \n"
	   "\n"
	   "Usage: spectral_taccumulate_test [options]\n"
	   " -a  Number of input frequency chunks\n"
	   " -b  Number of packets of each stream per frequency chunk\n"
	   " -c  Number of FFT points\n"
	   " -h  show help\n");
}

int main(int argc, char *argv[])
{
  int i, j, k, ndim = 6;
  int arg, nchk_in, stream_ndf_chk, cufft_nx, nchan_in, naccumulate, naccumulate_pow2;
  dim3 grid_size, block_size;
  uint64_t nsamp_in, nsamp_out, npol_in, npol_out, idx_in, idx_out;
  cufftComplex *data = NULL, *g_in = NULL;
  float *h_result = NULL, *g_result = NULL, *g_out = NULL, aa, bb, u, v;
  
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
  nchan_in = nchk_in * NCHAN_PER_CHUNK;
  naccumulate = stream_ndf_chk * NSAMP_DF / cufft_nx;
  naccumulate_pow2 = (int)pow(2.0, floor(log2((double)naccumulate)));
  grid_size.x = nchan_in;
  grid_size.y = cufft_nx / OVER_SAMP_RATE;
  grid_size.z = 1;
  block_size.x = (naccumulate_pow2<1024)?naccumulate_pow2:1024;
  block_size.y = 1;
  block_size.z = 1;
  fprintf(stdout, "kernel configuration is (%d, %d, %d) and (%d, %d, %d), naccumulate is %d\n", grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, naccumulate);
  nsamp_in  = nchan_in * stream_ndf_chk * NSAMP_DF / OVER_SAMP_RATE;
  nsamp_out = nchan_in * cufft_nx/OVER_SAMP_RATE;  
  npol_in   = nsamp_in * NPOL_BASEBAND;
  npol_out  = nsamp_out * ndim;
  fprintf(stdout, "%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\n", nsamp_in, nsamp_out, npol_in, npol_out);
  
  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&data,     npol_in * sizeof(cufftComplex)));
  CudaSafeCall(cudaMallocHost((void **)&h_result, npol_out * sizeof(float)));
  CudaSafeCall(cudaMallocHost((void **)&g_result, npol_out * sizeof(float)));
  CudaSafeCall(cudaMalloc((void **)&g_in,         npol_in * sizeof(cufftComplex)));
  CudaSafeCall(cudaMalloc((void **)&g_out,        npol_out * sizeof(float)));
  CudaSafeCall(cudaMemset((void *)g_out,  0,      npol_out * sizeof(float)));
  CudaSafeCall(cudaMemset((void *)h_result, 0,    npol_out * sizeof(float)));
  
  /* Prepare the data */
  srand(time(NULL));
  for(i = 0; i < nchan_in; i++)
    {
      for(j = 0; j < cufft_nx / OVER_SAMP_RATE; j++)
	{
	  idx_out = i * cufft_nx / OVER_SAMP_RATE + j;
	  for(k = 0; k < stream_ndf_chk * NSAMP_DF / cufft_nx; k++)
	    {
	      idx_in      = i * stream_ndf_chk * NSAMP_DF/OVER_SAMP_RATE + j * stream_ndf_chk * NSAMP_DF / cufft_nx + k;
	      
  	      data[idx_in].x = rand()*RAND_STD/RAND_MAX;
  	      data[idx_in].y = rand()*RAND_STD/RAND_MAX;
  	      data[idx_in+nsamp_in].x = rand()*RAND_STD/RAND_MAX;
  	      data[idx_in+nsamp_in].y = rand()*RAND_STD/RAND_MAX;
	      
	      aa = data[idx_in].x * data[idx_in].x + data[idx_in].y * data[idx_in].y;
	      bb = data[idx_in+nsamp_in].x * data[idx_in+nsamp_in].x + data[idx_in+nsamp_in].y * data[idx_in+nsamp_in].y;
	      u = 2 * (data[idx_in].x * data[idx_in+nsamp_in].x + data[idx_in].y * data[idx_in+nsamp_in].y);
	      v = 2 * (data[idx_in].x * data[idx_in+nsamp_in].y - data[idx_in].y * data[idx_in+nsamp_in].x);
	      	      
	      h_result[idx_out] += (aa + bb);
	      h_result[idx_out + nsamp_out] += (aa - bb);
	      h_result[idx_out + nsamp_out*2] += u;
	      h_result[idx_out + nsamp_out*3] += v;
	      h_result[idx_out + nsamp_out*4] += aa;
	      h_result[idx_out + nsamp_out*5] += bb;
	    }
	}
    }
  
  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(g_in, data, npol_in * sizeof(cufftComplex), cudaMemcpyHostToDevice));

  switch (block_size.x)
    {
    case 1024:
      spectral_taccumulate_kernel<1024><<<grid_size, block_size, ndim * block_size.x * NBYTE>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;
      
    case 512:
      spectral_taccumulate_kernel< 512><<<grid_size, block_size, ndim * block_size.x * NBYTE>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;
      
    case 256:
      spectral_taccumulate_kernel< 256><<<grid_size, block_size, ndim * block_size.x * NBYTE>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;
      
    case 128:
      spectral_taccumulate_kernel< 128><<<grid_size, block_size, ndim * block_size.x * NBYTE>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;
      
    case 64:
      spectral_taccumulate_kernel<  64><<<grid_size, block_size, ndim * block_size.x * NBYTE>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;
      
    case 32:
      spectral_taccumulate_kernel<  32><<<grid_size, block_size, ndim * block_size.x * NBYTE>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;
      
    case 16:
      spectral_taccumulate_kernel<  16><<<grid_size, block_size, ndim * block_size.x * NBYTE>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;
      
    case 8:
      spectral_taccumulate_kernel<   8><<<grid_size, block_size, ndim * block_size.x * NBYTE>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;
      
    case 4:
      spectral_taccumulate_kernel<   4><<<grid_size, block_size, ndim * block_size.x * NBYTE>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;
      
    case 2:
      spectral_taccumulate_kernel<   2><<<grid_size, block_size, ndim * block_size.x * NBYTE>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;
      
    case 1:
      spectral_taccumulate_kernel<   1><<<grid_size, block_size, ndim * block_size.x * NBYTE>>>(g_in, g_out, nsamp_in, nsamp_out, naccumulate);
      break;
    }
  CudaSafeKernelLaunch();
  CudaSafeCall(cudaMemcpy(g_result, g_out, npol_out * sizeof(float), cudaMemcpyDeviceToHost));

  /* Check the result */
  for(i = 0; i < nsamp_out; i++)
    {
      for(j = 0; j < ndim; j++)
	if(fabs((h_result[i*ndim + j] - g_result[i*ndim + j])/g_result[i*ndim + j])>1E-3)
	  {
	    fprintf(stdout, "%d\t%d\t%f\t%f\t%E\t", i, j, h_result[i*ndim + j], g_result[i*ndim + j], (h_result[i*ndim + j] - g_result[i*ndim + j])/g_result[i*ndim + j]);
	    fprintf(stdout, "\n");
	  }
      //fprintf(stdout, "\n");
    }

  /* Free buffer */
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  CudaSafeCall(cudaFree(g_in));
  CudaSafeCall(cudaFree(g_out));
  
  return EXIT_SUCCESS;
}