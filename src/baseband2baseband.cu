#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <cuda_profiler_api.h>

#include "log.h"
#include "baseband2baseband.cuh"
#include "cudautil.cuh"
#include "kernel.cuh"
#include "constants.h"

extern pthread_mutex_t log_mutex;

int default_arguments(conf_t *conf)
{
  memset(conf->dir, 0x00, sizeof(conf->dir));
  sprintf(conf->dir, "unset"); // Default with "unset"  
  conf->ndf_per_chunk_rbufin = 0; // Default with an impossible value
  conf->nstream          = -1; // Default with an impossible value
  conf->ndf_per_chunk_stream = 0; // Default with an impossible value
  conf->nchunk = -1;
  conf->cufft_nx = -1;
  conf->sod = -1;
  
  return EXIT_SUCCESS;
}

int initialize_baseband2baseband(conf_t *conf)
{
  int i;
  int iembed1, istride1, idist1, oembed1, ostride1, odist1, batch1, nx1;
  int iembed2, istride2, idist2, oembed2, ostride2, odist2, batch2, nx2;
  uint64_t hdrsz;
  int naccumulate_pow2;

  conf->nrepeat_per_blk = conf->ndf_per_chunk_rbufin / (conf->ndf_per_chunk_stream * conf->nstream);
  conf->nchan = conf->nchunk * NCHAN_PER_CHUNK;
  conf->nchan_keep_chan = conf->cufft_nx / OVER_SAMP_RATE;
  conf->cufft_mod = (int)(0.5 * conf->nchan_keep_chan);
  
  log_add(conf->log_file, "INFO", 1, log_mutex, "We have %d channels input", conf->nchan);
  log_add(conf->log_file, "INFO", 1, log_mutex, "The mod to reduce oversampling is %d", conf->cufft_mod);
  log_add(conf->log_file, "INFO", 1, log_mutex, "We will keep %d fine channels for each input channel after FFT", conf->nchan_keep_chan);
  log_add(conf->log_file, "INFO", 1, log_mutex, "%d run to finish one ring buffer block", conf->nrepeat_per_blk);
  
  /* Prepare buffer, stream and fft plan for process */
  conf->ndim_scale = conf->ndf_per_chunk_rbufin * NSAMP_DF * NPOL_BASEBAND * NDIM_BASEBAND; // Only works when two polarisations has similar power level
  conf->scale_dtsz = NBYTE_FOLD * NDIM_BASEBAND/((double)NBYTE_CUFFT_COMPLEX * OVER_SAMP_RATE);
  log_add(conf->log_file, "INFO", 1, log_mutex, "The data size rate is %f", conf->scale_dtsz);

  conf->nsamp1  = conf->ndf_per_chunk_stream * conf->nchan * NSAMP_DF;  // For each stream
  conf->npol1   = conf->nsamp1 * NPOL_BASEBAND;
  conf->ndata1  = conf->npol1  * NDIM_BASEBAND;

  conf->nsamp2  = conf->nsamp1 / OVER_SAMP_RATE;
  conf->npol2   = conf->nsamp2 * NPOL_BASEBAND;
  conf->ndata2  = conf->npol2  * NDIM_BASEBAND;
  nx1        = conf->cufft_nx;
  batch1     = conf->npol1 / conf->cufft_nx;
  
  iembed1    = nx1;
  istride1   = 1;
  idist1     = nx1;
  
  oembed1    = nx1;
  ostride1   = 1;
  odist1     = nx1;
  
  nx2        = conf->nchan_keep_chan;
  batch2     = conf->npol2 / conf->nchan_keep_chan;
  
  iembed2    = nx2;
  istride2   = 1;
  idist2     = nx2;
  
  oembed2    = nx2;
  ostride2   = 1;
  odist2     = nx2;

  conf->streams = (cudaStream_t *)malloc(conf->nstream * sizeof(cudaStream_t));
  conf->fft_plans1 = (cufftHandle *)malloc(conf->nstream * sizeof(cufftHandle));
  conf->fft_plans2 = (cufftHandle *)malloc(conf->nstream * sizeof(cufftHandle));
  for(i = 0; i < conf->nstream; i ++)
    {
      CudaSafeCall(cudaStreamCreate(&conf->streams[i]));
      CufftSafeCall(cufftPlanMany(&conf->fft_plans1[i], CUFFT_RANK, &nx1, &iembed1, istride1, idist1, &oembed1, ostride1, odist1, CUFFT_C2C, batch1));
      CufftSafeCall(cufftPlanMany(&conf->fft_plans2[i], CUFFT_RANK, &nx2, &iembed2, istride2, idist2, &oembed2, ostride2, odist2, CUFFT_C2C, batch2));
      
      CufftSafeCall(cufftSetStream(conf->fft_plans1[i], conf->streams[i]));
      CufftSafeCall(cufftSetStream(conf->fft_plans2[i], conf->streams[i]));
    }
  
  conf->sbufin_size    = conf->ndata1 * NBYTE_BASEBAND;
  conf->sbufout_size   = conf->ndata2 * NBYTE_FOLD;
  
  conf->bufin_size     = conf->nstream * conf->sbufin_size;
  conf->bufout_size    = conf->nstream * conf->sbufout_size;
  
  conf->sbufrt1_size = conf->npol1 * NBYTE_CUFFT_COMPLEX;
  conf->sbufrt2_size = conf->npol2 * NBYTE_CUFFT_COMPLEX;
  conf->bufrt1_size  = conf->nstream * conf->sbufrt1_size;
  conf->bufrt2_size  = conf->nstream * conf->sbufrt2_size;
    
  conf->hbufin_offset = conf->sbufin_size / NBYTE_CHAR;
  conf->dbufin_offset = conf->sbufin_size / (NBYTE_BASEBAND * NPOL_BASEBAND * NDIM_BASEBAND);
  conf->bufrt1_offset = conf->sbufrt1_size / NBYTE_CUFFT_COMPLEX;
  conf->bufrt2_offset = conf->sbufrt2_size / NBYTE_CUFFT_COMPLEX;
  
  conf->dbufout_offset   = conf->sbufout_size / NBYTE_FOLD;
  conf->hbufout_offset   = conf->sbufout_size;

  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_in, conf->bufin_size));  
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out, conf->bufout_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt1, conf->bufrt1_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt2, conf->bufrt2_size)); 

  CudaSafeCall(cudaMalloc((void **)&conf->offset_scale_d, conf->nstream * conf->nchan * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&conf->offset_scale_h, conf->nchan * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMemset((void *)conf->offset_scale_d, 0, conf->nstream * conf->nchan * NBYTE_CUFFT_COMPLEX));// We have to clear the memory for this parameter
  /* Prepare the setup of kernels */
  conf->gridsize_unpack.x = conf->ndf_per_chunk_stream;
  conf->gridsize_unpack.y = conf->nchunk;
  conf->gridsize_unpack.z = 1;
  conf->blocksize_unpack.x = NSAMP_DF; 
  conf->blocksize_unpack.y = NCHAN_PER_CHUNK;
  conf->blocksize_unpack.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of unpack kernel is (%d, %d, %d) and (%d, %d, %d)",
	      conf->gridsize_unpack.x, conf->gridsize_unpack.y, conf->gridsize_unpack.z,
	      conf->blocksize_unpack.x, conf->blocksize_unpack.y, conf->blocksize_unpack.z);
  
  conf->naccumulate = conf->ndf_per_chunk_stream * NSAMP_DF;
  naccumulate_pow2  = (int)pow(2.0, floor(log2((double)conf->naccumulate)));
  conf->gridsize_taccumulate.x = conf->nchan;
  conf->gridsize_taccumulate.y = 1;
  conf->gridsize_taccumulate.z = 1;
  conf->blocksize_taccumulate.x = (naccumulate_pow2<1024)?naccumulate_pow2:1024;
  conf->blocksize_taccumulate.y = 1;
  conf->blocksize_taccumulate.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of taccumulate kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_taccumulate.x, conf->gridsize_taccumulate.y, conf->gridsize_taccumulate.z,
	  conf->blocksize_taccumulate.x, conf->blocksize_taccumulate.y, conf->blocksize_taccumulate.z);
  
  conf->gridsize_scale.x = 1;
  conf->gridsize_scale.y = 1;
  conf->gridsize_scale.z = 1;
  conf->blocksize_scale.x = conf->nchan;
  conf->blocksize_scale.y = 1;
  conf->blocksize_scale.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of scale kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_scale.x, conf->gridsize_scale.y, conf->gridsize_scale.z,
	  conf->blocksize_scale.x, conf->blocksize_scale.y, conf->blocksize_scale.z);
  
  conf->gridsize_swap_select_transpose_swap.x = conf->nchan;
  conf->gridsize_swap_select_transpose_swap.y = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx;
  conf->gridsize_swap_select_transpose_swap.z = 1;  
  conf->blocksize_swap_select_transpose_swap.x = conf->cufft_nx;
  conf->blocksize_swap_select_transpose_swap.y = 1;
  conf->blocksize_swap_select_transpose_swap.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of swap_select_transpose_swap kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_swap_select_transpose_swap.x, conf->gridsize_swap_select_transpose_swap.y, conf->gridsize_swap_select_transpose_swap.z,
	  conf->blocksize_swap_select_transpose_swap.x, conf->blocksize_swap_select_transpose_swap.y, conf->blocksize_swap_select_transpose_swap.z);
  
  conf->gridsize_transpose_pad.x = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx; 
  conf->gridsize_transpose_pad.y = conf->nchan;
  conf->gridsize_transpose_pad.z = 1;
  conf->blocksize_transpose_pad.x = conf->nchan_keep_chan;
  conf->blocksize_transpose_pad.y = 1;
  conf->blocksize_transpose_pad.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of transpose_pad kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_transpose_pad.x, conf->gridsize_transpose_pad.y, conf->gridsize_transpose_pad.z,
	  conf->blocksize_transpose_pad.x, conf->blocksize_transpose_pad.y, conf->blocksize_transpose_pad.z);
  
  conf->gridsize_transpose_scale.x = ceil(conf->nchan_keep_chan / (double)TILE_DIM);  
  conf->gridsize_transpose_scale.y = ceil(conf->nchan / (double)TILE_DIM);
  conf->gridsize_transpose_scale.z = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx; 
  conf->blocksize_transpose_scale.x = TILE_DIM;
  conf->blocksize_transpose_scale.y = NROWBLOCK_TRANS;
  conf->blocksize_transpose_scale.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of transpose_scale kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_transpose_scale.x, conf->gridsize_transpose_scale.y, conf->gridsize_transpose_scale.z,
	  conf->blocksize_transpose_scale.x, conf->blocksize_transpose_scale.y, conf->blocksize_transpose_scale.z);
    
  /* attach to input ring buffer */
  conf->hdu_in = dada_hdu_create(NULL);
  dada_hdu_set_key(conf->hdu_in, conf->key_in);
  if(dada_hdu_connect(conf->hdu_in) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      destroy_baseband2baseband(*conf);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);    
    }  
  conf->db_in = (ipcbuf_t *) conf->hdu_in->data_block;
  conf->rbufin_size = ipcbuf_get_bufsz(conf->db_in);
  
  if(conf->rbufin_size % conf->bufin_size != 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      destroy_baseband2baseband(*conf);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);    
    }
  struct timespec start, stop;
  double elapsed_time;
  clock_gettime(CLOCK_REALTIME, &start);
  dada_cuda_dbregister(conf->hdu_in);  // registers the existing host memory range for use by CUDA
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "elapsed_time for dbregister of input ring buffer is %f\n", elapsed_time);
  fflush(stdout);

  hdrsz = ipcbuf_get_bufsz(conf->hdu_in->header_block);  
  if(hdrsz != DADA_HDRSZ)    // This number should match
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      destroy_baseband2baseband(*conf);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);    
    }
  if(dada_hdu_lock_read(conf->hdu_in) < 0) // make ourselves the read client 
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error locking HDU, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      destroy_baseband2baseband(*conf);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }

  /* Prepare output ring buffer */
  conf->hdu_out = dada_hdu_create(NULL);
  dada_hdu_set_key(conf->hdu_out, conf->key_out);
  if(dada_hdu_connect(conf->hdu_out) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      destroy_baseband2baseband(*conf);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);    
    }
  conf->db_out = (ipcbuf_t *) conf->hdu_out->data_block;
  conf->rbufout_size = ipcbuf_get_bufsz(conf->db_out);
  //fprintf(stdout, "%"PRIu64"\t%"PRIu64"\n", conf->rbufout_size, conf->bufout_size);
    
  if(conf->rbufout_size % conf->bufout_size != 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      destroy_baseband2baseband(*conf);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);    
    }  
  hdrsz = ipcbuf_get_bufsz(conf->hdu_out->header_block);  
  if(hdrsz != DADA_HDRSZ)    // This number should match
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      destroy_baseband2baseband(*conf);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);    
    }  
  if(dada_hdu_lock_write(conf->hdu_out) < 0)   // make ourselves the write client 
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error locking HDU, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      destroy_baseband2baseband(*conf);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  clock_gettime(CLOCK_REALTIME, &start);
  dada_cuda_dbregister(conf->hdu_out);  // registers the existing host memory range for use by CUDA
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "elapsed_time for dbregister of output ring buffer is %f\n", elapsed_time);
  fflush(stdout);
  
  if(conf->sod == 0)
    {
      if(ipcbuf_disable_sod(conf->db_out) < 0)
	{
	  log_add(conf->log_file, "ERR", 1, log_mutex, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.", __FILE__, __LINE__);
	  fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  
	  destroy_baseband2baseband(*conf);
	  fclose(conf->log_file);
	  CudaSafeCall(cudaProfilerStop());
	  exit(EXIT_FAILURE);
	}
    }
  
  return EXIT_SUCCESS;
}

int baseband2baseband(conf_t conf)
{
  /*
    The whole procedure for fold mode is :
    1. Unpack the data and reorder it from TFTFP to PFT order, prepare for the forward FFT;
    2. Forward FFT the PFT data to get finer channelzation and the data is in PFTF order after FFT;
    3. Swap the FFT output to put the frequency centre on the right place, drop frequency channel edge and band edge and put the data into PTF order, swap the data and put the centre frequency at bin 0 for each FFT block, prepare for inverse FFT;
    4. Inverse FFT the data to get PTFT order data;
    5. Transpose the data to get TFP data and scale it;    
  */
  uint64_t i, j;
  uint64_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset, hbufout_offset, dbufout_offset;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap;
  dim3 gridsize_transpose_scale, blocksize_transpose_scale;
  uint64_t curbufsz;
  int first = 1;
  double time_res_blk, time_offset = 0;
  
  gridsize_unpack                      = conf.gridsize_unpack;
  blocksize_unpack                     = conf.blocksize_unpack;
  gridsize_swap_select_transpose_swap  = conf.gridsize_swap_select_transpose_swap;   
  blocksize_swap_select_transpose_swap = conf.blocksize_swap_select_transpose_swap;  
  gridsize_transpose_scale             = conf.gridsize_transpose_scale;
  blocksize_transpose_scale            = conf.blocksize_transpose_scale;

  fprintf(stdout, "BASEBAND2BASEBAND_READY\n");  // Ready to take data from ring buffer, just before the header thing
  fflush(stdout);
  log_add(conf.log_file, "INFO", 1, log_mutex, "BASEBAND2BASEBAND_READY");

  read_dada_header(&conf); 
  time_res_blk = conf.tsamp * conf.ndf_per_chunk_rbufin * NSAMP_DF / 1.0E6; // This has to be after read_register_header, in seconds
  if(conf.sod == 1)
    register_dada_header(&conf); 
  
  /* Do the real job */  
  while(!ipcbuf_eod(conf.db_in))
    {
      conf.curbuf_in  = ipcbuf_get_next_read(conf.db_in, &curbufsz);
      conf.curbuf_out = ipcbuf_get_next_write(conf.db_out);
      
      /* Get scale of data */
      if(first)
	{
	  first = 0;
	  offset_scale(conf);
	}
      for(i = 0; i < conf.nrepeat_per_blk; i ++)
	{
	  for(j = 0; j < conf.nstream; j++)
	    {
	      hbufin_offset = j * conf.hbufin_offset + i * conf.bufin_size;
	      dbufin_offset = j * conf.dbufin_offset; 
	      bufrt1_offset = j * conf.bufrt1_offset;
	      bufrt2_offset = j * conf.bufrt2_offset;

	      dbufout_offset = j * conf.dbufout_offset;
	      hbufout_offset = j * conf.hbufout_offset + i * conf.bufout_size;
	      
	      CudaSafeCall(cudaMemcpyAsync(&conf.dbuf_in[dbufin_offset], &conf.curbuf_in[hbufin_offset], conf.sbufin_size, cudaMemcpyHostToDevice, conf.streams[j]));
	      
	      /* Unpack raw data into cufftComplex array */
	      unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, conf.streams[j]>>>(&conf.dbuf_in[dbufin_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp1);
	      CudaSafeKernelLaunch();
	      
	      /* Do forward FFT */
	      CufftSafeCall(cufftExecC2C(conf.fft_plans1[j], &conf.buf_rt1[bufrt1_offset], &conf.buf_rt1[bufrt1_offset], CUFFT_FORWARD));

	      /* Prepare for inverse FFT */
	      swap_select_transpose_swap_kernel<<<gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2, conf.cufft_nx, conf.cufft_mod, conf.nchan_keep_chan);
	      CudaSafeKernelLaunch();
	      
	      /* Do inverse FFT */
	      CufftSafeCall(cufftExecC2C(conf.fft_plans2[j], &conf.buf_rt2[bufrt2_offset], &conf.buf_rt2[bufrt2_offset], CUFFT_INVERSE));
	      
	      /* Get final output */
	      transpose_scale_kernel<<<gridsize_transpose_scale, blocksize_transpose_scale, 0, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nchan_keep_chan, conf.nchan, conf.nsamp2, conf.offset_scale_d);
	      CudaSafeKernelLaunch();
	      
	      /* Copy the final output to host */
	      CudaSafeCall(cudaMemcpyAsync(&conf.curbuf_out[hbufout_offset], &conf.dbuf_out[dbufout_offset], conf.sbufout_size, cudaMemcpyDeviceToHost, conf.streams[j]));
	    }
	}
      CudaSynchronizeCall(); // Sync here is for multiple streams

      /* Close current buffer */
      ipcbuf_mark_filled(conf.db_out, (uint64_t)(curbufsz * conf.scale_dtsz));
      ipcbuf_mark_cleared(conf.db_in);

      time_offset += time_res_blk;
      fprintf(stdout, "BASEBAND2BASEBAND, finished %f seconds data\n", time_offset);
      fflush(stdout);
    }
  return EXIT_SUCCESS;
}

int destroy_baseband2baseband(conf_t conf)
{
  int i;
  
  for (i = 0; i < conf.nstream; i++)
    {
      if(conf.streams[i])
	CudaSafeCall(cudaStreamDestroy(conf.streams[i]));
      if(conf.fft_plans1[i])
	CufftSafeCall(cufftDestroy(conf.fft_plans1[i]));
      if(conf.fft_plans2[i])
	CufftSafeCall(cufftDestroy(conf.fft_plans2[i]));
    }

  if(conf.dbuf_in)
    cudaFree(conf.dbuf_in);
  if(conf.buf_rt1)
    cudaFree(conf.buf_rt1);
  if(conf.buf_rt2)
    cudaFree(conf.buf_rt2);

  if(conf.dbuf_out)
    cudaFree(conf.dbuf_out);
  if(conf.offset_scale_h)
    cudaFreeHost(conf.offset_scale_h);
  if(conf.offset_scale_d)
    cudaFree(conf.offset_scale_d);

  if(conf.db_out)
    {
      dada_cuda_dbunregister(conf.hdu_out);  
      dada_hdu_unlock_write(conf.hdu_out);
      dada_hdu_destroy(conf.hdu_out);
    }

  if(conf.db_in)
    {
      dada_cuda_dbunregister(conf.hdu_in);  
      dada_hdu_unlock_read(conf.hdu_in);
      dada_hdu_destroy(conf.hdu_in);
    }
  
  if(conf.streams)
    free(conf.streams);
  if(conf.fft_plans1)
    free(conf.fft_plans1);
  if(conf.fft_plans2)
    free(conf.fft_plans2);
  
  return EXIT_SUCCESS;
}

int offset_scale(conf_t conf)
{
  /*
    The procedure for fold mode is:
    1. Get PTFT data as we did at process;
    2. Pad the data;
    3. Add the padded data in time;
    4. Get the mean of the added data;
    5. Get the scale with the mean;

    The procedure for search mode is:
    1. Get PTF data as we did at process;
    2. Add the data in frequency to get NCHAN_SEARCH channels, detect the added data and pad it;
    3. Add the padded data in time;    
    4. Get the mean of the added data;
    5. Get the scale with the mean;
  */
  size_t i, j;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap;
  dim3 gridsize_scale, blocksize_scale; 
  dim3 gridsize_transpose_pad, blocksize_transpose_pad;
  dim3 gridsize_taccumulate, blocksize_taccumulate;
  
  size_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset;
    
  gridsize_unpack                      = conf.gridsize_unpack;
  blocksize_unpack                     = conf.blocksize_unpack;
  gridsize_swap_select_transpose_swap  = conf.gridsize_swap_select_transpose_swap;   
  blocksize_swap_select_transpose_swap = conf.blocksize_swap_select_transpose_swap; 
  gridsize_transpose_pad               = conf.gridsize_transpose_pad;
  blocksize_transpose_pad              = conf.blocksize_transpose_pad;
  	         	               	
  gridsize_scale             = conf.gridsize_scale;	       
  blocksize_scale            = conf.blocksize_scale;
  gridsize_taccumulate  = conf.gridsize_taccumulate;
  blocksize_taccumulate = conf.blocksize_taccumulate;
  
  for(i = 0; i < conf.rbufin_size; i += conf.bufin_size)
    {
      for (j = 0; j < conf.nstream; j++)
	{
	  hbufin_offset = j * conf.hbufin_offset + i;
	  dbufin_offset = j * conf.dbufin_offset; 
	  bufrt1_offset = j * conf.bufrt1_offset;
	  bufrt2_offset = j * conf.bufrt2_offset;
	  
	  /* Copy data into device */
	  CudaSafeCall(cudaMemcpyAsync(&conf.dbuf_in[dbufin_offset], &conf.curbuf_in[hbufin_offset], conf.sbufin_size, cudaMemcpyHostToDevice, conf.streams[j]));

	  /* Unpack raw data into cufftComplex array */
	  unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, conf.streams[j]>>>(&conf.dbuf_in[dbufin_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp1);
	  CudaSafeKernelLaunch();
	  
	  /* Do forward FFT */
	  CufftSafeCall(cufftExecC2C(conf.fft_plans1[j], &conf.buf_rt1[bufrt1_offset], &conf.buf_rt1[bufrt1_offset], CUFFT_FORWARD));

	  /* Prepare for inverse FFT */
	  swap_select_transpose_swap_kernel<<<gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2, conf.cufft_nx, conf.cufft_mod, conf.nchan_keep_chan);
	  CudaSafeKernelLaunch();
	  
	  /* Do inverse FFT */
	  CufftSafeCall(cufftExecC2C(conf.fft_plans2[j], &conf.buf_rt2[bufrt2_offset], &conf.buf_rt2[bufrt2_offset], CUFFT_INVERSE));
	  
	  /* Transpose the data from PTFT to FTP for later calculation */
	  transpose_pad_kernel<<<gridsize_transpose_pad, blocksize_transpose_pad, 0, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], conf.nsamp2, &conf.buf_rt1[bufrt1_offset]);
	  CudaSafeKernelLaunch();
	  
	  switch (blocksize_taccumulate.x)
	    {
	    case 1024:
	      reduce10_kernel<1024><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 512:
	      reduce10_kernel< 512><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 256:
	      reduce10_kernel< 256><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 128:
	      reduce10_kernel< 128><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 64:
	      reduce10_kernel<  64><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 32:
	      reduce10_kernel<  32><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 16:
	      reduce10_kernel<  16><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 8:
	      reduce10_kernel<   8><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 4:
	      reduce10_kernel<   4><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 2:
	      reduce10_kernel<   2><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 1:
	      reduce10_kernel<   1><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan], conf.naccumulate, conf.ndim_scale);
	      break;
	    }
	  CudaSafeKernelLaunch();
	}
    }
  CudaSynchronizeCall();
  
  /* Get the scale of each chanel */
  scale3_kernel<<<gridsize_scale, blocksize_scale>>>(conf.offset_scale_d, conf.nchan, conf.nstream, SCL_NSIG, SCL_INT8);
  CudaSafeKernelLaunch();
  CudaSynchronizeCall();
  
  CudaSafeCall(cudaMemcpy(conf.offset_scale_h, conf.offset_scale_d, NBYTE_CUFFT_COMPLEX * conf.nchan, cudaMemcpyDeviceToHost));
  CudaSynchronizeCall();
  
  /* Record scale into file */
  for (i = 0; i< conf.nchan; i++)
    {
      fprintf(stdout, "%E\t%E\n", conf.offset_scale_h[i].x, conf.offset_scale_h[i].y);
      fflush(stdout);
    }
  char fname[MSTR_LEN];
  FILE *fp=NULL;
  sprintf(fname, "%s/%s_scale.txt", conf.dir, conf.utc_start);
  fp = fopen(fname, "w");
  if(fp == NULL)
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Can not open scale file, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Can not open scale file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      destroy_baseband2baseband(conf);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  for (i = 0; i< conf.nchan; i++)
    fprintf(fp, "%E\t%E\n", conf.offset_scale_h[i].x, conf.offset_scale_h[i].y);

  fclose(fp);
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
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: ndf_per_chunk_rbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_rbufin, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "ndf_per_chunk_rbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort", conf.ndf_per_chunk_rbufin, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "Each input ring buffer block has %"PRIu64" packets per frequency chunk", conf.ndf_per_chunk_rbufin); 

  if(conf.nstream <= 0)
    {
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: nstream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.nstream, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "nstream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.nstream, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "%d streams run on GPU", conf.nstream);
  
  if(conf.ndf_per_chunk_stream == 0)
    {
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: ndf_per_chunk_stream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_stream, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "ndf_per_chunk_stream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.ndf_per_chunk_stream, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "Each stream process %d packets per frequency chunk", conf.ndf_per_chunk_stream);
  log_add(conf.log_file, "INFO", 1, log_mutex, "The runtime information is %s", conf.dir);  // Checked already
  
  if(conf.nchunk<=0 || conf.nchunk>NCHUNK_FULL_BEAM)    
    {
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: nchunk shoule be in (0 %d], but it is %d, which happens at \"%s\", line [%d], has to abort\n", NCHUNK_FULL_BEAM, conf.nchunk, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "nchunk shoule be in (0 %d], but it is %d, which happens at \"%s\", line [%d], has to abort", NCHUNK_FULL_BEAM, conf.nchunk, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }  
  log_add(conf.log_file, "INFO", 1, log_mutex, "%d chunks of input data", conf.nchunk);

  if(conf.cufft_nx<=0)    
    {
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: cufft_nx shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.cufft_nx, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "cufft_nx shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.cufft_nx, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "We use %d points FFT", conf.cufft_nx);
  
  if(conf.sod == 1)
    log_add(conf.log_file, "INFO", 1, log_mutex, "The filterbank data is enabled at the beginning");
  else if(conf.sod == 0)
    log_add(conf.log_file, "INFO", 1, log_mutex, "The filterbank data is NOT enabled at the beginning");
  else
    {      
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: The SOD is not set, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "The SOD is not set, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      
      log_close(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  
  return EXIT_SUCCESS;
}


int read_dada_header(conf_t *conf)
{  
  uint64_t hdrsz;
  
  conf->hdrbuf_in  = ipcbuf_get_next_read(conf->hdu_in->header_block, &hdrsz);  
  if (!conf->hdrbuf_in)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting header_buf, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2baseband(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  if(hdrsz != DADA_HDRSZ)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Header size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Header size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2baseband(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  
  if (ascii_header_get(conf->hdrbuf_in, "FILE_SIZE", "%"SCNu64"", &conf->file_size_in) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting FILE_SIZE, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Error getting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2baseband(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }   
  log_add(conf->log_file, "INFO", 1, log_mutex, "FILE_SIZE from DADA header is %"PRIu64"", conf->file_size_in);
  
  if (ascii_header_get(conf->hdrbuf_in, "BYTES_PER_SECOND", "%"SCNu64"", &conf->bytes_per_second_in) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2baseband(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "BYTES_PER_SECOND from DADA header is %"PRIu64"", conf->bytes_per_second_in);
  
  if (ascii_header_get(conf->hdrbuf_in, "TSAMP", "%lf", &conf->tsamp) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting TSAMP, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Error getting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2baseband(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "TSAMP from DADA header is %f", conf->tsamp);
  
  /* Get utc_start from hdrin */
  if (ascii_header_get(conf->hdrbuf_in, "UTC_START", "%s", conf->utc_start) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting UTC_START, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Error getting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      destroy_baseband2baseband(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "UTC_START from DADA header is %s", conf->utc_start);
    
  if(ipcbuf_mark_cleared (conf->hdu_in->header_block))  // We are the only one reader, so that we can clear it after read;
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error header_clear, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Error header_clear, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2baseband(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  return EXIT_SUCCESS;
}

int register_dada_header(conf_t *conf)
{
  char *hdrbuf_out = NULL;
  uint64_t file_size, bytes_per_second;
  
  hdrbuf_out = ipcbuf_get_next_write(conf->hdu_out->header_block);
  if (!hdrbuf_out)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting header_buf, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2baseband(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }  
  memcpy(hdrbuf_out, conf->hdrbuf_in, DADA_HDRSZ); // Pass the header
  
  file_size = (uint64_t)(conf->file_size_in * conf->scale_dtsz);
  bytes_per_second = (uint64_t)(conf->bytes_per_second_in * conf->scale_dtsz);
  
  if (ascii_header_set(hdrbuf_out, "NBIT", "%d", NBIT_BASEBAND) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Error setting NBIT, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2baseband(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "NBIT to DADA header is %d", NBIT_BASEBAND);
  
  if (ascii_header_set(hdrbuf_out, "NDIM", "%d", NDIM_BASEBAND) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting NDIM, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Error setting NDIM, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2baseband(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "NDIM to DADA header is %d", NDIM_BASEBAND);
  
  if (ascii_header_set(hdrbuf_out, "NPOL", "%d", NPOL_BASEBAND) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting NPOL, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Error setting NPOL, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2baseband(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "NPOL to DADA header is %d", NPOL_BASEBAND);
  
  if (ascii_header_set(hdrbuf_out, "FILE_SIZE", "%"PRIu64"", file_size) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: BASEBAND2BASEBAND_ERROR:\tError setting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2baseband(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "FILE_SIZE to DADA header is %"PRIu64"", file_size);
  
  if (ascii_header_set(hdrbuf_out, "BYTES_PER_SECOND", "%"PRIu64"", bytes_per_second) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2baseband(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "BYTES_PER_SECOND to DADA header is %"PRIu64"", bytes_per_second);
  
  /* donot set header parameters anymore */
  if (ipcbuf_mark_filled (conf->hdu_out->header_block, DADA_HDRSZ) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error header_fill, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2baseband(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }

  return EXIT_SUCCESS;
}
