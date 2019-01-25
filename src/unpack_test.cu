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

#define NBYTE_RT  8
#define MAX_RAND       1000

extern "C" void usage ()
{
  fprintf (stdout,
	   "unpack_test - Test the unpack kernel \n"
	   "\n"
	   "Usage: unpack_test [options]\n"
	   " -a  Number of packets per frequency chunk of each stream\n"
	   " -b  Number of input chunks\n"
	   " -h  show help\n");
}

// ./unpack_test -a 1024 -b 33 
int main(int argc, char *argv[])
{
  int i, j, k, l;
  int arg, stream_ndf_chk, nchk_in;
  dim3 gridsize_unpack, blocksize_unpack;
  uint64_t nsamp, npol, ndata, idx_in, idx_out;
  cufftComplex *g_out = NULL, *h_result = NULL, *g_result = NULL;
  int64_t *g_in = NULL, tmp, *data_int64 = NULL;
  int16_t *data_int16 = NULL;
  
  /* read in parameters */
  while((arg=getopt(argc,argv,"a:hb:")) != -1)
    {      
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);	  

	case 'a':	  
	  if (sscanf (optarg, "%d", &stream_ndf_chk) != 1)
	    {
	      fprintf (stderr, "Does not get stream_ndf_chk, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'b':	  
	  if (sscanf (optarg, "%d", &nchk_in) != 1)
	    {
	      fprintf (stderr, "Does not get nchk_in, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	}
    }
  fprintf(stdout, "stream_ndf_chk is %d and nchk_in is %d\n", stream_ndf_chk, nchk_in);

  /* Setup size */
  gridsize_unpack.x = stream_ndf_chk;
  gridsize_unpack.y = nchk_in;
  gridsize_unpack.z = 1;
  blocksize_unpack.x = NSAMP_DF; 
  blocksize_unpack.y = NCHAN_CHK;
  blocksize_unpack.z = 1;
  fprintf(stdout, "The configuration of the kernel is (%d, %d, %d) and (%d, %d, %d)\n", gridsize_unpack.x, gridsize_unpack.y, gridsize_unpack.z, blocksize_unpack.x, blocksize_unpack.y, blocksize_unpack.z);

  nsamp  = stream_ndf_chk * nchk_in * NSAMP_DF * NCHAN_CHK;
  npol   = nsamp * NPOL_IN;
  ndata  = npol * NDIM_IN;
  fprintf(stdout, "nsamp is %"PRIu64", %"PRIu64", %"PRIu64"\n", nsamp, npol, ndata);
  
  /* Create buffer */
  //CudaSafeCall(cudaMallocHost((void **)&data,     nsamp * sizeof(int64_t)));
  CudaSafeCall(cudaMallocHost((void **)&data_int16, ndata * sizeof(int16_t)));
  CudaSafeCall(cudaMalloc((void **)&g_in,           nsamp * sizeof(int64_t)));
  CudaSafeCall(cudaMallocHost((void **)&h_result,   npol * sizeof(cufftComplex)));
  CudaSafeCall(cudaMallocHost((void **)&g_result,   npol * sizeof(cufftComplex)));
  CudaSafeCall(cudaMalloc((void **)&g_out,          npol * sizeof(cufftComplex)));

  /* Prepare data */ 
  for(i = 0; i < ndata; i++)
    data_int16[i] = (int16_t)(rand()/(float)(RAND_MAX/(float)MAX_RAND));//tmp_f;
  data_int64 = (int64_t *)data_int16;
  
  /* calculate on CPU*/
  for(i = 0; i < stream_ndf_chk; i ++)
    {
      for(j = 0 ; j < nchk_in; j++)
	{
	  for(k = 0 ; k < NSAMP_DF; k++)
	    {
	      for(l = 0 ; l < NCHAN_CHK; l++)
		{
		  idx_in = i*nchk_in*NSAMP_DF*NCHAN_CHK +
		    j*NSAMP_DF*NCHAN_CHK + k*NCHAN_CHK + l;

		  idx_out = j*NCHAN_CHK*stream_ndf_chk*NSAMP_DF +
		    l*stream_ndf_chk*NSAMP_DF + i*NSAMP_DF + k;
		  
		  tmp = bswap_64(data_int64[idx_in]);
		  
		  h_result[idx_out].x = (int16_t)((tmp & 0x000000000000ffffULL));
		  h_result[idx_out].y = (int16_t)((tmp & 0x00000000ffff0000ULL) >> 16);
		  h_result[idx_out + nsamp].x = (int16_t)((tmp & 0x0000ffff00000000ULL) >> 32);
		  h_result[idx_out + nsamp].y = (int16_t)((tmp & 0xffff000000000000ULL) >> 48);
		}
	    }
	}
    }
    
  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(g_in, data_int16, ndata * sizeof(int16_t), cudaMemcpyHostToDevice));
  unpack_kernel<<<gridsize_unpack, blocksize_unpack>>>(g_in, g_out, nsamp);
  CHECK_LAUNCH_ERROR();
  CudaSafeCall(cudaMemcpy(g_result, g_out, npol * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

  /* Check the result */
  for(i = 0; i < nsamp; i++)
    {
      if((h_result[i].x - g_result[i].x) !=0 || (h_result[i].y - g_result[i].y) != 0)
	fprintf(stdout, "%f\t%f\t%f\t%f\t%f\t%f\n", h_result[i].x, g_result[i].x, h_result[i].x - g_result[i].x, h_result[i].y, g_result[i].y, h_result[i].y - g_result[i].y);
      if((h_result[i+nsamp].x - g_result[i+nsamp].x) !=0 || (h_result[i+nsamp].y - g_result[i+nsamp].y) !=0)
	fprintf(stdout, "%f\t%f\t%f\t%f\t%f\t%f\n", h_result[i+nsamp].x, g_result[i+nsamp].x, h_result[i+nsamp].x - g_result[i+nsamp].x, h_result[i+nsamp].y, g_result[i+nsamp].y, h_result[i+nsamp].y - g_result[i+nsamp].y);
    }
  /* Free buffer */
  CudaSafeCall(cudaFreeHost(data_int16));
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  CudaSafeCall(cudaFree(g_in));
  CudaSafeCall(cudaFree(g_out));
  
  return EXIT_SUCCESS;
}