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

extern "C" void usage ()
{
  fprintf (stdout,
	   "detect_faccumulate_scale2_spectral_faccumulate_test - Test the detect_faccumulate_scale kernel \n"
	   "\n"
	   "Usage: detect_faccumulate_scale_test [options]\n"
	   " -a  Grid size in X, which is number of samples in time\n"
	   " -b  Grid size in Y, which is number of channels\n"
	   " -c  Block size in X\n"
	   " -d  Number of samples to accumulate in each block\n"
	   " -h  show help\n");
}

// ./detect_faccumulate_scale2_spectral_faccumulate_test -a 512 -b 1 -c 512 -d 1024 
int main(int argc, char *argv[])
{
  int i, j,l, k, arg;
  int grid_x, grid_y, block_x;
  uint64_t n_accumulate, idx;
  uint64_t nsamp, npol, nout, nchan;
  dim3 gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale;
  float power, aa, bb, u, v;
  cufftComplex *mean_scale_h = NULL, *mean_scale_d = NULL, *g_in = NULL, *data = NULL;
  uint8_t *g_out1 = NULL, *g_result1 = NULL;
  float *g_out2 = NULL, *g_result2 = NULL;
  float *h_result1 = NULL, *h_result2 = NULL;
  
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
	  
	case 'd':	  
	  if (sscanf (optarg, "%"SCNu64"", &n_accumulate) != 1)
	    {
	      fprintf (stderr, "Does not get n_accumulate, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  fprintf(stdout, "n_accumulate is %"PRIu64"\n",  n_accumulate);
	  break;
	}
    }

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
  npol                                 = NPOL_BASEBAND * nsamp;
  nchan                                = grid_y;
  
  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&mean_scale_h, nchan * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&data,  npol * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&h_result1,  nout * NBYTE_FLOAT));
  CudaSafeCall(cudaMallocHost((void **)&h_result2,  nout * NDATA_PER_SAMP_RT * NBYTE_FLOAT));
  CudaSafeCall(cudaMallocHost((void **)&g_result1,  nout * sizeof(uint8_t)));
  CudaSafeCall(cudaMallocHost((void **)&g_result2,  nout * NDATA_PER_SAMP_RT * NBYTE_FLOAT));
  CudaSafeCall(cudaMalloc((void **)&g_out1,  nout * sizeof(uint8_t)));
  CudaSafeCall(cudaMalloc((void **)&g_out2,  nout * NDATA_PER_SAMP_RT * NBYTE_FLOAT));
  CudaSafeCall(cudaMalloc((void **)&mean_scale_d, nchan * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMalloc((void **)&g_in,  npol * NBYTE_CUFFT_COMPLEX));  
    
  /* cauculate on CPU */
  srand(time(NULL));
  for(i = 0; i < nchan; i ++) // Prepare the scale
    {
      mean_scale_h[i].x = fabs(rand()*RAND_STD/RAND_MAX) + 1 ;
      mean_scale_h[i].y = fabs(rand()*RAND_STD/RAND_MAX) * 50000.0;
    }
  for(i = 0; i < grid_x; i ++) // Prepare the input data
    {
      for(j = 0; j < grid_y; j ++)
	{
	  power = 0;
	  aa = 0;
	  bb = 0;
	  u = 0;
	  v = 0;
	  h_result1[i*grid_y+grid_y-j-1] = 0;
	  for (k = 0; k < NDATA_PER_SAMP_RT; k++)
	    h_result2[i*grid_y+grid_y-j-1 + k * nout] = 0;
	  
	  for(k = 0; k < n_accumulate; k++)
	    {
	      idx = (i*grid_y + j) * n_accumulate + k;
	      for(l = 0; l < NPOL_BASEBAND; l++)
		{
		  data[idx+l*nsamp].x = fabs(rand()*RAND_STD/RAND_MAX);
		  data[idx+l*nsamp].y = fabs(rand()*RAND_STD/RAND_MAX);
		  
		  power += (data[idx+l*nsamp].x*data[idx+l*nsamp].x + data[idx+l*nsamp].y*data[idx+l*nsamp].y);
		}
	      aa += (data[idx].x * data[idx].x + data[idx].y * data[idx].y);
	      bb += (data[idx + nsamp].x * data[idx + nsamp].x + data[idx + nsamp].y * data[idx + nsamp].y);;
	      u += 2 * (data[idx].x * data[idx + nsamp].x + data[idx].y * data[idx + nsamp].y);
	      v += 2 * (data[idx].x * data[idx + nsamp].y - data[idx + nsamp].x * data[idx].y);
	    }
	  if(mean_scale_h[j].y == 0 )
	    h_result1[i*grid_y+grid_y-j-1] = (power);
	  else
	    h_result1[i*grid_y+grid_y-j-1] = ((power - mean_scale_h[j].x) / mean_scale_h[j].y + OFFS_UINT8); // Reverse frequency order

	  h_result2[i*grid_y+grid_y-j-1 + 0 * nout] += (aa + bb);
	  h_result2[i*grid_y+grid_y-j-1 + 1 * nout] += (aa - bb);
	  h_result2[i*grid_y+grid_y-j-1 + 2 * nout] += u;
	  h_result2[i*grid_y+grid_y-j-1 + 3 * nout] += v;
	  h_result2[i*grid_y+grid_y-j-1 + 4 * nout] += aa;
	  h_result2[i*grid_y+grid_y-j-1 + 5 * nout] += bb;
	}
    }
    
  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(mean_scale_d, mean_scale_h, nchan * NBYTE_CUFFT_COMPLEX, cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(g_in, data, npol * NBYTE_CUFFT_COMPLEX, cudaMemcpyHostToDevice));
    
  switch (blocksize_detect_faccumulate_scale.x)
    {
    case 1024:
      detect_faccumulate_scale2_spectral_faccumulate_kernel<1024><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out1, g_out2, nsamp, nout, n_accumulate, mean_scale_d);
      break;

    case 512:
      detect_faccumulate_scale2_spectral_faccumulate_kernel< 512><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out1, g_out2, nsamp, nout, n_accumulate, mean_scale_d);
      break;

    case 256:
      detect_faccumulate_scale2_spectral_faccumulate_kernel< 256><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out1, g_out2, nsamp, nout, n_accumulate, mean_scale_d);
      break;

    case 128:
      detect_faccumulate_scale2_spectral_faccumulate_kernel< 128><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out1, g_out2, nsamp, nout, n_accumulate, mean_scale_d);
      break;

    case 64:
      detect_faccumulate_scale2_spectral_faccumulate_kernel<  64><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out1, g_out2, nsamp, nout, n_accumulate, mean_scale_d);
      break;

    case 32:
      detect_faccumulate_scale2_spectral_faccumulate_kernel<  32><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out1, g_out2, nsamp, nout, n_accumulate, mean_scale_d);
      break;

    case 16:
      detect_faccumulate_scale2_spectral_faccumulate_kernel<  16><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out1, g_out2, nsamp, nout, n_accumulate, mean_scale_d);
      break;

    case 8:
      detect_faccumulate_scale2_spectral_faccumulate_kernel<   8><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out1, g_out2, nsamp, nout, n_accumulate, mean_scale_d);
      break;
      
    case 4:
      detect_faccumulate_scale2_spectral_faccumulate_kernel<   4><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out1, g_out2, nsamp, nout, n_accumulate, mean_scale_d);
      break;
      
    case 2:
      detect_faccumulate_scale2_spectral_faccumulate_kernel<   2><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out1, g_out2, nsamp, nout, n_accumulate, mean_scale_d);
      break;
      
    case 1:
      detect_faccumulate_scale2_spectral_faccumulate_kernel<   1><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT>>>(g_in, g_out1, g_out2, nsamp, nout, n_accumulate, mean_scale_d);
      break;
    }
  CudaSafeKernelLaunch();    
  CudaSafeCall(cudaMemcpy(g_result1, g_out1, nout * sizeof(uint8_t), cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(g_result2, g_out2, nout * NDATA_PER_SAMP_RT * NBYTE_FLOAT, cudaMemcpyDeviceToHost));
 
  /* Check the result */
  for(i = 0; i < nout; i++)
    fprintf(stdout, "CPU:\t%f\tGPU:\t%d\t%f\n", h_result1[i], g_result1[i], g_result1[i] - floor(h_result1[i]));

  for(i = 0; i < NDATA_PER_SAMP_RT * nout; i++)
    if(abs((g_result2[i] - h_result2[i])/h_result2[i])>1.0E-4)
      fprintf(stdout, "CPU:\t%f\tGPU:\t%f\t%f\t%E\n", h_result2[i], g_result2[i], g_result2[i] - h_result2[i], (g_result2[i] - h_result2[i])/h_result2[i]);
  
  /* Free buffer */  
  CudaSafeCall(cudaFreeHost(mean_scale_h));
  CudaSafeCall(cudaFreeHost(h_result1));
  CudaSafeCall(cudaFreeHost(h_result2));
  CudaSafeCall(cudaFreeHost(g_result1));
  CudaSafeCall(cudaFreeHost(g_result2));
  CudaSafeCall(cudaFreeHost(data));
  CudaSafeCall(cudaFree(g_out1));
  CudaSafeCall(cudaFree(g_out2));
  CudaSafeCall(cudaFree(mean_scale_d));
  CudaSafeCall(cudaFree(g_in));
  
  return EXIT_SUCCESS;
}