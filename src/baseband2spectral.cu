#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>

#include "baseband2spectral.cuh"
#include "cudautil.cuh"
#include "kernel.cuh"
#include "log.h"
#include "constants.h"

extern pthread_mutex_t log_mutex;

int default_arguments(conf_t *conf)
{
  memset(conf->dir, 0x00, sizeof(conf->dir));
  sprintf(conf->dir, "unset"); // Default with "unset"
  memset(conf->ip, 0x00, sizeof(conf->ip));
  sprintf(conf->ip, "unset"); // Default with "unset"
  
  conf->ndf_per_chunk_rbufin = 0; // Default with an impossible value
  conf->nstream              = -1;// Default with an impossible value
  conf->ndf_per_chunk_stream = 0; // Default with an impossible value
  conf->sod = -1;                 // Default no SOD at the beginning
  conf->nchunk_in = -1;
  conf->port = -1;
  conf->cufft_nx = -1;
  conf->output_network = -1;
  conf->p_type = -1;
  conf->ndim_out = -1;
  conf->npol_out = -1;
  
  return EXIT_SUCCESS;
}

int initialize_baseband2spectral(conf_t *conf)
{
  int i;
  int iembed, istride, idist, oembed, ostride, odist, batch, nx;
  int naccumulate_pow2;
  
  /* Prepare parameters */
  conf->naccumulate     = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx;
  conf->nrepeat_per_blk = conf->ndf_per_chunk_rbufin / (conf->ndf_per_chunk_stream * conf->nstream);
  conf->nchan_in        = conf->nchunk_in * NCHAN_PER_CHUNK;
  conf->nchan_keep_chan = (int)(conf->cufft_nx / OVER_SAMP_RATE);
  conf->nchan_out       = conf->nchan_in * conf->nchan_keep_chan;
  conf->cufft_mod       = (int)(0.5 * conf->cufft_nx / OVER_SAMP_RATE);
  conf->scale_dtsz      = NBYTE_SPECTRAL * NDATA_PER_SAMP_FULL * conf->nchan_in * conf->nchan_keep_chan / (double)(NBYTE_BASEBAND * NPOL_BASEBAND * NDIM_BASEBAND * conf->ndf_per_chunk_rbufin * conf->nchan_in * NSAMP_DF); // replace NDATA_PER_SAMP_FULL with conf->p_type if we do not fill 0 for other pols

  log_add(conf->log_file, "INFO", 1, log_mutex, "We have %d channels input", conf->nchan_in);
  log_add(conf->log_file, "INFO", 1, log_mutex, "The mod to reduce oversampling is %d", conf->cufft_mod);
  log_add(conf->log_file, "INFO", 1, log_mutex, "We will keep %d fine channels for each input channel after FFT", conf->nchan_keep_chan);
  log_add(conf->log_file, "INFO", 1, log_mutex, "The data size rate between spectral and baseband data is %E", conf->scale_dtsz);
  log_add(conf->log_file, "INFO", 1, log_mutex, "%d run to finish one ring buffer block", conf->nrepeat_per_blk);
  
  /* Prepare buffer, stream and fft plan for process */
  conf->nsamp1      = conf->ndf_per_chunk_stream * conf->nchan_in * NSAMP_DF;
  conf->npol1       = conf->nsamp1 * NPOL_BASEBAND;
  conf->ndata1      = conf->npol1  * NDIM_BASEBAND;
  
  conf->nsamp2      = conf->nsamp1 / OVER_SAMP_RATE;
  conf->npol2       = conf->nsamp2 * NPOL_BASEBAND;
  conf->ndata2      = conf->npol2  * NDIM_BASEBAND;

  conf->nsamp3      = conf->nsamp2 / conf->naccumulate;
  conf->ndata3      = conf->nsamp3  * NDATA_PER_SAMP_RT;
  
  nx        = conf->cufft_nx;
  batch     = conf->npol1 / conf->cufft_nx;
  
  iembed    = nx;
  istride   = 1;
  idist     = nx;
  
  oembed    = nx;
  ostride   = 1;
  odist     = nx;
  
  conf->streams = (cudaStream_t *)malloc(conf->nstream * sizeof(cudaStream_t));
  conf->fft_plans = (cufftHandle *)malloc(conf->nstream * sizeof(cufftHandle));
  for(i = 0; i < conf->nstream; i ++)
    {
      CudaSafeCall(cudaStreamCreate(&conf->streams[i]));
      CufftSafeCall(cufftPlanMany(&conf->fft_plans[i], CUFFT_RANK, &nx, &iembed, istride, idist, &oembed, ostride, odist, CUFFT_C2C, batch));
      CufftSafeCall(cufftSetStream(conf->fft_plans[i], conf->streams[i]));
    }
  
  conf->sbufin_size  = conf->ndata1 * NBYTE_BASEBAND;
  conf->sbufout_size = conf->ndata3 * NBYTE_SPECTRAL;
  
  conf->bufin_size   = conf->nstream * conf->sbufin_size;
  conf->bufout_size  = conf->nstream * conf->sbufout_size;
  
  conf->sbufrt1_size = conf->npol1 * NBYTE_RT;
  conf->sbufrt2_size = conf->npol2 * NBYTE_RT;
  conf->bufrt1_size  = conf->nstream * conf->sbufrt1_size;
  conf->bufrt2_size  = conf->nstream * conf->sbufrt2_size;
    
  conf->hbufin_offset = conf->sbufin_size;
  conf->dbufin_offset = conf->sbufin_size / (NBYTE_BASEBAND * NPOL_BASEBAND * NDIM_BASEBAND);
  conf->bufrt1_offset = conf->sbufrt1_size / NBYTE_RT;
  conf->bufrt2_offset = conf->sbufrt2_size / NBYTE_RT;
  
  conf->dbufout_offset = conf->sbufout_size / NBYTE_SPECTRAL;

  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_in, conf->bufin_size));  
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out, conf->bufout_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt1, conf->bufrt1_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt2, conf->bufrt2_size));

  /* Prepare the setup of kernels */
  conf->gridsize_unpack.x = conf->ndf_per_chunk_stream;
  conf->gridsize_unpack.y = conf->nchunk_in;
  conf->gridsize_unpack.z = 1;
  conf->blocksize_unpack.x = NSAMP_DF; 
  conf->blocksize_unpack.y = NCHAN_PER_CHUNK;
  conf->blocksize_unpack.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of unpack kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_unpack.x, conf->gridsize_unpack.y, conf->gridsize_unpack.z,
	  conf->blocksize_unpack.x, conf->blocksize_unpack.y, conf->blocksize_unpack.z);
  
  conf->gridsize_swap_select_transpose_pft1.x = ceil(conf->cufft_nx / (double)TILE_DIM);  
  conf->gridsize_swap_select_transpose_pft1.y = ceil(conf->ndf_per_chunk_stream * NSAMP_DF / (double) (conf->cufft_nx * TILE_DIM));
  conf->gridsize_swap_select_transpose_pft1.z = conf->nchan_in;
  conf->blocksize_swap_select_transpose_pft1.x = TILE_DIM;
  conf->blocksize_swap_select_transpose_pft1.y = NROWBLOCK_TRANS;
  conf->blocksize_swap_select_transpose_pft1.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of swap_select_transpose_pft1 kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_swap_select_transpose_pft1.x, conf->gridsize_swap_select_transpose_pft1.y, conf->gridsize_swap_select_transpose_pft1.z,
	  conf->blocksize_swap_select_transpose_pft1.x, conf->blocksize_swap_select_transpose_pft1.y, conf->blocksize_swap_select_transpose_pft1.z);        
  naccumulate_pow2 = (int)pow(2.0, floor(log2((double)conf->naccumulate)));
  conf->gridsize_spectral_taccumulate.x = conf->nchan_in;
  conf->gridsize_spectral_taccumulate.y = conf->cufft_nx / OVER_SAMP_RATE;
  conf->gridsize_spectral_taccumulate.z = 1;
  conf->blocksize_spectral_taccumulate.x = (naccumulate_pow2<1024)?naccumulate_pow2:1024;
  conf->blocksize_spectral_taccumulate.y = 1;
  conf->blocksize_spectral_taccumulate.z = 1; 
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of spectral_taccumulate kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_spectral_taccumulate.x, conf->gridsize_spectral_taccumulate.y, conf->gridsize_spectral_taccumulate.z,
	  conf->blocksize_spectral_taccumulate.x, conf->blocksize_spectral_taccumulate.y, conf->blocksize_spectral_taccumulate.z);        

  conf->gridsize_spectral_saccumulate.x = NDATA_PER_SAMP_RT;
  conf->gridsize_spectral_saccumulate.y = conf->cufft_nx / OVER_SAMP_RATE;
  conf->gridsize_spectral_saccumulate.z = 1;
  conf->blocksize_spectral_saccumulate.x = conf->nchan_in;
  conf->blocksize_spectral_saccumulate.y = 1;
  conf->blocksize_spectral_saccumulate.z = 1; 
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of spectral_saccumulate kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_spectral_saccumulate.x, conf->gridsize_spectral_saccumulate.y, conf->gridsize_spectral_saccumulate.z,
	  conf->blocksize_spectral_saccumulate.x, conf->blocksize_spectral_saccumulate.y, conf->blocksize_spectral_saccumulate.z);        

  /* attach to input ring buffer */
  conf->hdu_in = dada_hdu_create(NULL);
  dada_hdu_set_key(conf->hdu_in, conf->key_in);
  if(dada_hdu_connect(conf->hdu_in) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }  
  conf->db_in = (ipcbuf_t *) conf->hdu_in->data_block;
  conf->rbufin_size = ipcbuf_get_bufsz(conf->db_in);
  if((conf->rbufin_size % conf->bufin_size != 0) || (conf->rbufin_size/conf->bufin_size)!= conf->nrepeat_per_blk)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }

  struct timespec start, stop;
  double elapsed_time;

  clock_gettime(CLOCK_REALTIME, &start);
  /* registers the existing host memory range for use by CUDA */
  dada_cuda_dbregister(conf->hdu_in); // To put this into capture does not improve the memcpy!!!
  
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "elapse_time for spectral for dbregister is %f\n", elapsed_time);
  fflush(stdout);
  
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_in->header_block);  
  if(conf->hdrsz != DADA_HDRSZ)    // This number should match
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }
  
  /* make ourselves the read client */
  if(dada_hdu_lock_read(conf->hdu_in) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error locking HDU, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }

  /* Prepare output ring buffer */
  if(conf->output_network==0)
    {
      conf->hdu_out = dada_hdu_create(NULL);
      dada_hdu_set_key(conf->hdu_out, conf->key_out);
      if(dada_hdu_connect(conf->hdu_out) < 0)
	{
	  log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  
	  destroy_baseband2spectral(*conf);
	  fclose(conf->log_file);
	  exit(EXIT_FAILURE);    
	}
      conf->db_out = (ipcbuf_t *) conf->hdu_out->data_block;
      conf->rbufout_size = ipcbuf_get_bufsz(conf->db_out);
      
      if(conf->rbufout_size != conf->nsamp3 * NDATA_PER_SAMP_FULL * NBYTE_SPECTRAL)
	{
	  // replace NDATA_PER_SAMP_FULL with conf->p_type if we do not fill 0 for other pols
	  log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, %"PRIu64" vs %"PRIu64", which happens at \"%s\", line [%d].", conf->rbufout_size, conf->nsamp3 * NDATA_PER_SAMP_FULL * NBYTE_SPECTRAL, __FILE__, __LINE__);
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Buffer size mismatch, %"PRIu64" vs %"PRIu64", which happens at \"%s\", line [%d].\n", conf->rbufout_size, conf->nsamp3 * NDATA_PER_SAMP_FULL * NBYTE_SPECTRAL, __FILE__, __LINE__);
	  
	  destroy_baseband2spectral(*conf);
	  fclose(conf->log_file);
	  exit(EXIT_FAILURE);    
	}
      
      conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_out->header_block);  
      if(conf->hdrsz != DADA_HDRSZ)    // This number should match
	{
	  log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  
	  destroy_baseband2spectral(*conf);
	  fclose(conf->log_file);
	  exit(EXIT_FAILURE);    
	}
      
      /* make ourselves the write client */
      if(dada_hdu_lock_write(conf->hdu_out) < 0)
	{
	  log_add(conf->log_file, "ERR", 1, log_mutex, "Error locking HDU, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  
	  destroy_baseband2spectral(*conf);
	  fclose(conf->log_file);
	  exit(EXIT_FAILURE);
	}
      
      if(!(conf->sod == 1))
	{
	  if(ipcbuf_disable_sod(conf->db_out) < 0)
	    {
	      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.", __FILE__, __LINE__);
	      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      
	      destroy_baseband2spectral(*conf);
	      fclose(conf->log_file);
	      exit(EXIT_FAILURE);
	    }
	}
    }
  
  return EXIT_SUCCESS;
}

int baseband2spectral(conf_t conf)
{
  uint64_t i, j;
  uint64_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset, dbufout_offset;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_pft1, blocksize_swap_select_transpose_pft1;
  dim3 gridsize_spectral_taccumulate, blocksize_spectral_taccumulate;
  dim3 gridsize_spectral_saccumulate, blocksize_spectral_saccumulate;
  uint64_t cbufsz;
  double time_res_blk, elapsed_time = 0;
  float *h_stokes_i = (float *)malloc(conf.nchan_out * sizeof(float));
  
  gridsize_unpack                      = conf.gridsize_unpack;
  blocksize_unpack                     = conf.blocksize_unpack;
  gridsize_swap_select_transpose_pft1  = conf.gridsize_swap_select_transpose_pft1;   
  blocksize_swap_select_transpose_pft1 = conf.blocksize_swap_select_transpose_pft1;
  gridsize_spectral_taccumulate        = conf.gridsize_spectral_taccumulate; 
  blocksize_spectral_taccumulate       = conf.blocksize_spectral_taccumulate;
  gridsize_spectral_saccumulate        = conf.gridsize_spectral_saccumulate; 
  blocksize_spectral_saccumulate       = conf.blocksize_spectral_saccumulate;
  
  fprintf(stdout, "BASEBAND2SPECTRAL_READY\n");  // Ready to take data from ring buffer, just before the header thing
  fflush(stdout);
  log_add(conf.log_file, "INFO", 1, log_mutex, "BASEBAND2SPECTRAL_READY");
  
  if(register_header(&conf))
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "header register failed, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2spectral(conf);
      free(h_stokes_i);
      fclose(conf.log_file);
      exit(EXIT_FAILURE);
    }
  time_res_blk = conf.tsamp_in * conf.ndf_per_chunk_rbufin * NSAMP_DF / 1.0E6; // This has to be after register_header, in seconds
  log_add(conf.log_file, "INFO", 1, log_mutex, "register_header done");
  
  while(!ipcbuf_eod(conf.db_in))
    {
      CudaSafeCall(cudaMemset((void *)conf.dbuf_out, 0, conf.bufout_size));// We have to clear the memory for this parameter
      
      log_add(conf.log_file, "INFO", 1, log_mutex, "before getting new buffer block");
      conf.cbuf_in  = ipcbuf_get_next_read(conf.db_in, &cbufsz);
      conf.cbuf_out = ipcbuf_get_next_write(conf.db_out);
      log_add(conf.log_file, "INFO", 1, log_mutex, "after getting new buffer block");
      
      for(i = 0; i < conf.nrepeat_per_blk; i ++)
	{
	  for(j = 0; j < conf.nstream; j++)
	    {
	      hbufin_offset = j * conf.hbufin_offset + i * conf.bufin_size;
	      dbufin_offset = j * conf.dbufin_offset; 
	      bufrt1_offset = j * conf.bufrt1_offset;
	      bufrt2_offset = j * conf.bufrt2_offset;
	      dbufout_offset = j * conf.dbufout_offset;
	      	      
	      /* Copy data into device */
	      CudaSafeCall(cudaMemcpyAsync(&conf.dbuf_in[dbufin_offset], &conf.cbuf_in[hbufin_offset], conf.sbufin_size, cudaMemcpyHostToDevice, conf.streams[j]));
	      
	      /* Unpack raw data into cufftComplex array */
	      unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, conf.streams[j]>>>(&conf.dbuf_in[dbufin_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp1);
	      CudaSafeKernelLaunch();
	      
	      /* Do forward FFT */
	      CufftSafeCall(cufftExecC2C(conf.fft_plans[j], &conf.buf_rt1[bufrt1_offset], &conf.buf_rt1[bufrt1_offset], CUFFT_FORWARD));

	      /* from PFTF order to PFT order, also remove channel edge */
	      swap_select_transpose_pft1_kernel<<<gridsize_swap_select_transpose_pft1, blocksize_swap_select_transpose_pft1, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.cufft_nx, NSAMP_DF, conf.nsamp1, conf.nsamp2, conf.cufft_nx, conf.cufft_mod, conf.nchan_keep_chan);
	      CudaSafeKernelLaunch();
	      
	      /* Convert to required pol and accumulate in time */
	      switch(blocksize_spectral_taccumulate.x)
	      	{
	      	case 1024:
	      	  spectral_taccumulate_kernel<1024><<<gridsize_spectral_taccumulate, blocksize_spectral_taccumulate, blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL, conf.streams[j]>>>(&conf.buf_rt2[conf.bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.nsamp3, conf.naccumulate);
		  break;
		  
	      	case 512:
	      	  spectral_taccumulate_kernel< 512><<<gridsize_spectral_taccumulate, blocksize_spectral_taccumulate, blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL, conf.streams[j]>>>(&conf.buf_rt2[conf.bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.nsamp3, conf.naccumulate);
		  break;
		  
	      	case 256:
	      	  spectral_taccumulate_kernel< 256><<<gridsize_spectral_taccumulate, blocksize_spectral_taccumulate, blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL, conf.streams[j]>>>(&conf.buf_rt2[conf.bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.nsamp3, conf.naccumulate);
		  break;
		  
	      	case 128:
	      	  spectral_taccumulate_kernel< 128><<<gridsize_spectral_taccumulate, blocksize_spectral_taccumulate, blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL, conf.streams[j]>>>(&conf.buf_rt2[conf.bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.nsamp3, conf.naccumulate);
		  break;
		  
	      	case  64:
	      	  spectral_taccumulate_kernel<  64><<<gridsize_spectral_taccumulate, blocksize_spectral_taccumulate, blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL, conf.streams[j]>>>(&conf.buf_rt2[conf.bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.nsamp3, conf.naccumulate);
		  break;
		  
	      	case  32:
	      	  spectral_taccumulate_kernel<  32><<<gridsize_spectral_taccumulate, blocksize_spectral_taccumulate, blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL, conf.streams[j]>>>(&conf.buf_rt2[conf.bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.nsamp3, conf.naccumulate);
		  break;
		  
	      	case  16:
	      	  spectral_taccumulate_kernel<  16><<<gridsize_spectral_taccumulate, blocksize_spectral_taccumulate, blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL, conf.streams[j]>>>(&conf.buf_rt2[conf.bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.nsamp3, conf.naccumulate);
		  break;
		  
	      	case  8:
	      	  spectral_taccumulate_kernel<   8><<<gridsize_spectral_taccumulate, blocksize_spectral_taccumulate, blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL, conf.streams[j]>>>(&conf.buf_rt2[conf.bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.nsamp3, conf.naccumulate);
		  break;
		  
	      	case  4:
	      	  spectral_taccumulate_kernel<   4><<<gridsize_spectral_taccumulate, blocksize_spectral_taccumulate, blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL, conf.streams[j]>>>(&conf.buf_rt2[conf.bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.nsamp3, conf.naccumulate);
		  break;
		  
	      	case  2:
	      	  spectral_taccumulate_kernel<   2><<<gridsize_spectral_taccumulate, blocksize_spectral_taccumulate, blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL, conf.streams[j]>>>(&conf.buf_rt2[conf.bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.nsamp3, conf.naccumulate);
		  break;
		  
	      	case  1:
	      	  spectral_taccumulate_kernel<   1><<<gridsize_spectral_taccumulate, blocksize_spectral_taccumulate, blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL, conf.streams[j]>>>(&conf.buf_rt2[conf.bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.nsamp3, conf.naccumulate);
		  break;
	      	}
	      CudaSafeKernelLaunch();
	    }
	}
      CudaSynchronizeCall(); // Sync here is for multiple streams

      spectral_saccumulate_kernel<<<conf.gridsize_spectral_saccumulate, conf.blocksize_spectral_saccumulate>>>(conf.dbuf_out, conf.ndata3, conf.nstream);  
      CudaSafeKernelLaunch();
      if(conf.p_type == 2)
	CudaSafeCall(cudaMemcpy(conf.cbuf_out, &conf.dbuf_out[conf.nsamp3  * NDATA_PER_SAMP_FULL], 2 * conf.nsamp3 * NBYTE_SPECTRAL, cudaMemcpyDeviceToHost));
      else
	CudaSafeCall(cudaMemcpy(conf.cbuf_out, conf.dbuf_out, conf.nsamp3  * conf.p_type * NBYTE_SPECTRAL, cudaMemcpyDeviceToHost));

      // Always copy Stokes I to host for later use, make a figure with it or record it to a file
      // We can also copy full stokes to host
      CudaSafeCall(cudaMemcpy(h_stokes_i, conf.dbuf_out, conf.nchan_out * NBYTE_SPECTRAL, cudaMemcpyDeviceToHost));
      
      log_add(conf.log_file, "INFO", 1, log_mutex, "before closing old buffer block");
      ipcbuf_mark_filled(conf.db_out, (uint64_t)(cbufsz * conf.scale_dtsz));
      ipcbuf_mark_cleared(conf.db_in);
      log_add(conf.log_file, "INFO", 1, log_mutex, "after closing old buffer block");

      elapsed_time += time_res_blk;
      fprintf(stdout, "BASEBAND2SPECTRAL, finished %f seconds data\n", elapsed_time);
      fflush(stdout);
    }

  log_add(conf.log_file, "INFO", 1, log_mutex, "FINISH the process");

  return EXIT_SUCCESS;
}

int destroy_baseband2spectral(conf_t conf)
{
  int i;
  for (i = 0; i < conf.nstream; i++)
    {
      if(conf.streams[i])
	CudaSafeCall(cudaStreamDestroy(conf.streams[i]));
      if(conf.fft_plans[i])
      CufftSafeCall(cufftDestroy(conf.fft_plans[i]));
    }
  if(conf.streams)
    free(conf.streams);
  if(conf.fft_plans)
    free(conf.fft_plans);
  log_add(conf.log_file, "INFO", 1, log_mutex, "destroy fft plan and stream done");

  if(conf.dbuf_in)
    cudaFree(conf.dbuf_in);
  if(conf.dbuf_out)
    cudaFree(conf.dbuf_out);
  log_add(conf.log_file, "INFO", 1, log_mutex, "Free cuda memory done");

  if(conf.db_in)
    {
      dada_cuda_dbunregister(conf.hdu_in);
      dada_hdu_unlock_read(conf.hdu_in);
      dada_hdu_destroy(conf.hdu_in);
    }
  if(conf.db_out)
    {
      dada_hdu_unlock_write(conf.hdu_out);
      dada_hdu_destroy(conf.hdu_out);
    }  
  log_add(conf.log_file, "INFO", 1, log_mutex, "destory hdu done");  

  return EXIT_SUCCESS;
}

int register_header(conf_t *conf)
{
  uint64_t hdrsz;
  char *hdrbuf_in = NULL, *hdrbuf_out = NULL;
  uint64_t file_size, bytes_per_seconds;
  
  hdrbuf_in  = ipcbuf_get_next_read(conf->hdu_in->header_block, &hdrsz);  
  if (!hdrbuf_in)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting header_buf, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if(hdrsz != DADA_HDRSZ)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Header size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Header size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }

  hdrbuf_out = ipcbuf_get_next_write(conf->hdu_out->header_block);
  if (!hdrbuf_out)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting header_buf, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }  
  if (ascii_header_get(hdrbuf_in, "FILE_SIZE", "%"PRIu64"", &file_size) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting FILE_SIZE, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }   
  if (ascii_header_get(hdrbuf_in, "BYTES_PER_SECOND", "%"PRIu64"", &bytes_per_seconds) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }  
  if (ascii_header_get(hdrbuf_in, "TSAMP", "%lf", &conf->tsamp_in) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting TSAMP, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }   
  /* Get utc_start from hdrin */
  if (ascii_header_get(hdrbuf_in, "UTC_START", "%s", conf->utc_start) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting UTC_START, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header
  
  file_size = (uint64_t)(file_size * conf->scale_dtsz);
  bytes_per_seconds = (uint64_t)(bytes_per_seconds * conf->scale_dtsz);
  
  if (ascii_header_set(hdrbuf_out, "NCHAN", "%d", conf->nchan_out) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting NCHAN, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error setting NCHAN, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  conf->tsamp_out = conf->tsamp_in * conf->ndf_per_chunk_rbufin * NSAMP_DF;
  if (ascii_header_set(hdrbuf_out, "TSAMP", "%lf", conf->tsamp_out) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting TSAMP, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error setting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }

  if (ascii_header_set(hdrbuf_out, "NBIT", "%d", NBIT_SPECTRAL) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error setting NBIT, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "NDIM", "%d", conf->ndim_out) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting NDIM, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error setting NDIM, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "NPOL", "%d", conf->npol_out) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting NPOL, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error setting NPOL, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "FILE_SIZE", "%"PRIu64"", file_size) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: BASEBAND2SPECTRAL_ERROR:\tError setting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "BYTES_PER_SECOND", "%"PRIu64"", bytes_per_seconds) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
    
  if(ipcbuf_mark_cleared (conf->hdu_in->header_block))  // We are the only one reader, so that we can clear it after read;
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error header_clear, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error header_clear, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  /* donot set header parameters anymore - acqn. doesn't start */
  if (ipcbuf_mark_filled (conf->hdu_out->header_block, conf->hdrsz) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error header_fill, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }

  return EXIT_SUCCESS;
}

int examine_record_arguments(conf_t conf, char **argv, int argc)
{
  int i;
  char command_line[MSTR_LEN] = {'\0'};
  
  /* Log the input */
  strcpy(command_line, argv[0]);
  for(i = 1; i < argc; i++)
    {
      strcat(command_line, " ");
      strcat(command_line, argv[i]);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "The command line is \"%s\"", command_line);
  log_add(conf.log_file, "INFO", 1, log_mutex, "The input ring buffer key is %x", conf.key_in); 
  log_add(conf.log_file, "INFO", 1, log_mutex, "The output ring buffer key is %x", conf.key_out);

  if(conf.ndf_per_chunk_rbufin == 0)
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: ndf_per_chunk_rbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_rbufin, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "ndf_per_chunk_rbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort", conf.ndf_per_chunk_rbufin, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "Each input ring buffer block has %"PRIu64" packets per frequency chunk", conf.ndf_per_chunk_rbufin); 

  if(conf.nstream <= 0)
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: nstream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.nstream, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "nstream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.nstream, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "%d streams run on GPU", conf.nstream);
  
  if(conf.ndf_per_chunk_stream == 0)
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: ndf_per_chunk_stream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_stream, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "ndf_per_chunk_stream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.ndf_per_chunk_stream, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "Each stream process %d packets per frequency chunk", conf.ndf_per_chunk_stream);

  log_add(conf.log_file, "INFO", 1, log_mutex, "The runtime information is %s", conf.dir);  // Checked already

  if(conf.output_network == 0)
    {
      log_add(conf.log_file, "INFO", 1, log_mutex, "We will send spectral data with ring buffer");
      if(conf.sod == 1)
	log_add(conf.log_file, "INFO", 1, log_mutex, "The spectral data is enabled at the beginning");
      if(conf.sod == 0)
	log_add(conf.log_file, "INFO", 1, log_mutex, "The spectral data is NOT enabled at the beginning");
      if(conf.sod == -1)
	{
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: The sod should be 0 or 1 when we use ring buffer to send spectral data, but it is -1, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1, log_mutex, "The sod should be 0 or 1 when we use ring buffer to send spectral data, but it is -1, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      else
	log_add(conf.log_file, "INFO", 1, log_mutex, "The key for the spectral ring buffer is %x", conf.key_out);  
    }
  if(conf.output_network == 1)
    {
      log_add(conf.log_file, "INFO", 1, log_mutex, "We will send spectral data with network interface");
      if(strstr(conf.ip, "unset"))
	{
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: We are going to send spectral data with network interface, but no ip is given, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1, log_mutex, "We are going to send spectral data with network interface, but no ip is given, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      if(conf.port == -1)
	{
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: We are going to send spectral data with network interface, but no port is given, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1, log_mutex, "We are going to send spectral data with network interface, but no port is given, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      else
	log_add(conf.log_file, "INFO", 1, log_mutex, "The network interface for the spectral data is %s_%d", conf.ip, conf.port);  
    }
  if(conf.output_network == -1)
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: The method to send spectral data is not configured, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "The method to send spectral data is not configured, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  
  if(conf.nchunk_in<=0 || conf.nchunk_in>NCHUNK_MAX)    
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: nchunk_in shoule be in (0 %d], but it is %d, which happens at \"%s\", line [%d], has to abort\n", NCHUNK_MAX, conf.nchunk_in, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "nchunk_in shoule be in (0 %d], but it is %d, which happens at \"%s\", line [%d], has to abort", NCHUNK_MAX, conf.nchunk_in, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }  
  log_add(conf.log_file, "INFO", 1, log_mutex, "%d chunks of input data", conf.nchunk_in);

  if(conf.cufft_nx<=0)    
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: cufft_nx shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.cufft_nx, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "cufft_nx shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.cufft_nx, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "We use %d points FFT", conf.cufft_nx);

  if(conf.p_type == -1)
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: p_type should be 1, 2 or 4, but it is -1, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "p_type should be 1, 2 or 4, but it is -1, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "p_type is %d", conf.p_type, __FILE__, __LINE__);
  log_add(conf.log_file, "INFO", 1, log_mutex, "npol_out is %d", conf.npol_out, __FILE__, __LINE__);
  log_add(conf.log_file, "INFO", 1, log_mutex, "ndim_out is %d", conf.ndim_out, __FILE__, __LINE__);
  
  return EXIT_SUCCESS;
}