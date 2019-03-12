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
	   "unpack_test - Test the unpack kernel \n"
	   "\n"
	   "Usage: unpack_test [options]\n"
	   " -a  Number of packets per frequency chunk of each stream\n"
	   " -b  Number of input chunks\n"
	   " -c  Start frequency chunks\n"
	   " -d  Number of frequency chunks to zoom in\n"
	   " -h  show help\n");
}

// ./unpack_test -a 1024 -b 48 -c 20 -d 2
int main(int argc, char *argv[])
{
  int i, j, k, l;
  int arg, stream_ndf_chunk, nchunk, nchunk_zoom, start_chunk;
  dim3 gridsize_unpack, blocksize_unpack;
  uint64_t nsamp, npol, ndata, idx_in, idx_out, nsamp_zoom, npol_zoom, ndata_zoom, idx_out_zoom;
  cufftComplex *g_out = NULL, *h_result = NULL, *g_result = NULL;
  cufftComplex *g_out_zoom = NULL, *h_result_zoom = NULL, *g_result_zoom = NULL;
  int64_t *g_in = NULL, tmp, *data_int64 = NULL;
  int16_t *data_int16 = NULL;
  
  /* read in parameters */
  while((arg=getopt(argc,argv,"a:hb:c:d:")) != -1)
    {      
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);	  

	case 'a':	  
	  if (sscanf (optarg, "%d", &stream_ndf_chunk) != 1)
	    {
	      fprintf (stderr, "Does not get stream_ndf_chunk, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'b':	  
	  if (sscanf (optarg, "%d", &nchunk) != 1)
	    {
	      fprintf (stderr, "Does not get nchunk, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'c':	  
	  if (sscanf (optarg, "%d", &start_chunk) != 1)
	    {
	      fprintf (stderr, "Does not get start_chunk, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'd':	  
	  if (sscanf (optarg, "%d", &nchunk_zoom) != 1)
	    {
	      fprintf (stderr, "Does not get nchunk, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	}
    }
  fprintf(stdout, "stream_ndf_chunk is %d, nchunk is %d, start_chunk is %d and nchunk_zoom is %d\n", stream_ndf_chunk, nchunk, start_chunk, nchunk_zoom);

  /* Setup size */
  gridsize_unpack.x = stream_ndf_chunk;
  gridsize_unpack.y = nchunk;
  gridsize_unpack.z = 1;
  blocksize_unpack.x = NSAMP_DF; 
  blocksize_unpack.y = NCHAN_PER_CHUNK;
  blocksize_unpack.z = 1;
  fprintf(stdout, "The configuration of the kernel is (%d, %d, %d) and (%d, %d, %d)\n", gridsize_unpack.x, gridsize_unpack.y, gridsize_unpack.z, blocksize_unpack.x, blocksize_unpack.y, blocksize_unpack.z);

  nsamp  = stream_ndf_chunk * nchunk * NSAMP_DF * NCHAN_PER_CHUNK;
  npol   = nsamp * NPOL_BASEBAND;
  ndata  = npol * NDIM_BASEBAND;
  
  nsamp_zoom  = stream_ndf_chunk * nchunk_zoom * NSAMP_DF * NCHAN_PER_CHUNK;
  npol_zoom   = nsamp_zoom * NPOL_BASEBAND;
  ndata_zoom  = npol_zoom * NDIM_BASEBAND;
  fprintf(stdout, "nsamp is %"PRIu64", npol is %"PRIu64", ndata is %"PRIu64", nsamp_zoom is %"PRIu64", npol_zoom is %"PRIu64" and ndata_zoom is %"PRIu64"\n", nsamp, npol, ndata, nsamp_zoom, npol_zoom, ndata_zoom);
  
  /* Create buffer */
  CudaSafeCall(cudaMallocHost((void **)&data_int16, ndata * sizeof(int16_t)));
  CudaSafeCall(cudaMalloc((void **)&g_in,           nsamp * sizeof(int64_t)));
  CudaSafeCall(cudaMallocHost((void **)&h_result,   npol * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&g_result,   npol * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMalloc((void **)&g_out,          npol * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&h_result_zoom,   npol_zoom * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&g_result_zoom,   npol_zoom * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMalloc((void **)&g_out_zoom,          npol_zoom * NBYTE_CUFFT_COMPLEX));

  /* Prepare data */
  srand(time(NULL));
  for(i = 0; i < ndata; i++)
    data_int16[i] = (int16_t)(rand() * RAND_STD/RAND_MAX);
  data_int64 = (int64_t *)data_int16;
  
  /* calculate on CPU*/
  for(i = 0; i < stream_ndf_chunk; i ++)
    {
      for(j = 0 ; j < nchunk; j++)
	{
	  for(k = 0 ; k < NSAMP_DF; k++)
	    {
	      for(l = 0 ; l < NCHAN_PER_CHUNK; l++)
		{
		  idx_in = i*nchunk*NSAMP_DF*NCHAN_PER_CHUNK +
		    j*NSAMP_DF*NCHAN_PER_CHUNK + k*NCHAN_PER_CHUNK + l;

		  idx_out = j*NCHAN_PER_CHUNK*stream_ndf_chunk*NSAMP_DF +
		    l*stream_ndf_chunk*NSAMP_DF + i*NSAMP_DF + k;
		  
		  tmp = bswap_64(data_int64[idx_in]);
		  
		  h_result[idx_out].x = (int16_t)((tmp & 0x000000000000ffffULL));
		  h_result[idx_out].y = (int16_t)((tmp & 0x00000000ffff0000ULL) >> 16);
		  h_result[idx_out + nsamp].x = (int16_t)((tmp & 0x0000ffff00000000ULL) >> 32);
		  h_result[idx_out + nsamp].y = (int16_t)((tmp & 0xffff000000000000ULL) >> 48);

		  if((j>=start_chunk) && (j<(start_chunk + nchunk_zoom)))
		    {
		      idx_out_zoom = (j - start_chunk)*NCHAN_PER_CHUNK*stream_ndf_chunk*NSAMP_DF +
			l*stream_ndf_chunk*NSAMP_DF + i*NSAMP_DF + k;
		      
		      h_result_zoom[idx_out_zoom].x = (int16_t)((tmp & 0x000000000000ffffULL));
		      h_result_zoom[idx_out_zoom].y = (int16_t)((tmp & 0x00000000ffff0000ULL) >> 16);
		      h_result_zoom[idx_out_zoom + nsamp_zoom].x = (int16_t)((tmp & 0x0000ffff00000000ULL) >> 32);
		      h_result_zoom[idx_out_zoom + nsamp_zoom].y = (int16_t)((tmp & 0xffff000000000000ULL) >> 48);
		    }
		}
	    }
	}
    }
    
  /* Calculate on GPU */
  CudaSafeCall(cudaMemcpy(g_in, data_int16, ndata * sizeof(int16_t), cudaMemcpyHostToDevice));
  unpack1_kernel<<<gridsize_unpack, blocksize_unpack>>>(g_in, g_out, nsamp, g_out_zoom, nsamp_zoom, start_chunk, nchunk_zoom);
  CudaSafeKernelLaunch();

  CudaSafeCall(cudaMemcpy(g_result, g_out, npol * NBYTE_CUFFT_COMPLEX, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(g_result_zoom, g_out_zoom, npol_zoom * NBYTE_CUFFT_COMPLEX, cudaMemcpyDeviceToHost));

  /* Check the result */
  for(i = 0; i < npol; i++)
    {
      if((h_result[i].x - g_result[i].x) !=0 || (h_result[i].y - g_result[i].y) != 0)
	fprintf(stdout, "ORIGINAL:\t%f\t%f\t%f\t%f\t%f\t%f\n", h_result[i].x, g_result[i].x, h_result[i].x - g_result[i].x, h_result[i].y, g_result[i].y, h_result[i].y - g_result[i].y);
    }
  for(i = 0; i < npol_zoom; i++)
    {
      if((h_result_zoom[i].x - g_result_zoom[i].x) !=0 || (h_result_zoom[i].y - g_result_zoom[i].y) != 0)
	fprintf(stdout, "ZOOM:\t%f\t%f\t%f\t%f\t%f\t%f\n", h_result_zoom[i].x, g_result_zoom[i].x, h_result_zoom[i].x - g_result_zoom[i].x, h_result_zoom[i].y, g_result_zoom[i].y, h_result_zoom[i].y - g_result_zoom[i].y);
    }
  
  /* Free buffer */
  data_int64 = NULL;
  CudaSafeCall(cudaFreeHost(data_int16));
  CudaSafeCall(cudaFreeHost(h_result));
  CudaSafeCall(cudaFreeHost(g_result));
  CudaSafeCall(cudaFree(g_in));
  CudaSafeCall(cudaFree(g_out));
  CudaSafeCall(cudaFreeHost(h_result_zoom));
  CudaSafeCall(cudaFreeHost(g_result_zoom));
  CudaSafeCall(cudaFree(g_out_zoom));
  
  return EXIT_SUCCESS;
}