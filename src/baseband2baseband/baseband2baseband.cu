#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>

#include "multilog.h"
#include "baseband2baseband.cuh"
#include "cudautil.cuh"
#include "kernel.cuh"

extern multilog_t *runtime_log;
int quit;
pthread_mutex_t quit_mutex = PTHREAD_MUTEX_INITIALIZER;

int init_baseband2baseband(conf_t *conf)
{
  int i;
  int iembed1, istride1, idist1, oembed1, ostride1, odist1, batch1, nx1;
  int iembed2, istride2, idist2, oembed2, ostride2, odist2, batch2, nx2;
  ipcbuf_t *db_in = NULL, *db_out = NULL;
  uint64_t hdrsz;
  
  /* Prepare buffer, stream and fft plan for process */
  conf->sclndim = conf->rbufin_ndf_chk * NSAMP_DF * NPOL_SAMP * NDIM_POL; // Only works when two polarisations has similar power level
  conf->nsamp1  = conf->stream_ndf_chk * NCHK_CAPTURE * NCHAN_CHK * NSAMP_DF;
  conf->npol1   = conf->nsamp1 * NPOL_SAMP;
  conf->ndata1  = conf->npol1  * NDIM_POL;
		
  conf->nsamp2  = conf->nsamp1 * OSAMP_RATEI / NCHAN_RATEI;
  conf->npol2   = conf->nsamp2 * NPOL_SAMP;
  conf->ndata2  = conf->npol2  * NDIM_POL;

  nx1        = CUFFT_NX1;
  batch1     = conf->npol1 / CUFFT_NX1;
  
  iembed1    = nx1;
  istride1   = 1;
  idist1     = nx1;
  
  oembed1    = nx1;
  ostride1   = 1;
  odist1     = nx1;
  
  nx2        = CUFFT_NX2;
  batch2     = conf->npol2 / CUFFT_NX2;
  
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
      CufftSafeCall(cufftPlanMany(&conf->fft_plans1[i], CUFFT_RANK1, &nx1, &iembed1, istride1, idist1, &oembed1, ostride1, odist1, CUFFT_C2C, batch1));
      CufftSafeCall(cufftPlanMany(&conf->fft_plans2[i], CUFFT_RANK2, &nx2, &iembed2, istride2, idist2, &oembed2, ostride2, odist2, CUFFT_C2C, batch2));
      
      CufftSafeCall(cufftSetStream(conf->fft_plans1[i], conf->streams[i]));
      CufftSafeCall(cufftSetStream(conf->fft_plans2[i], conf->streams[i]));
    }
  
  conf->sbufin_size    = conf->ndata1 * NBYTE_IN;
  conf->sbufout_size   = conf->ndata2 * NBYTE_OUT;
  
  conf->bufin_size     = conf->nstream * conf->sbufin_size;
  conf->bufout_size    = conf->nstream * conf->sbufout_size;
  
  conf->sbufrt1_size = conf->npol1 * sizeof(cufftComplex);
  conf->sbufrt2_size = conf->npol2 * sizeof(cufftComplex);
  conf->bufrt1_size  = conf->nstream * conf->sbufrt1_size;
  conf->bufrt2_size  = conf->nstream * conf->sbufrt2_size;
    
  //conf->hbufin_offset = conf->sbufin_size / sizeof(char);
  conf->hbufin_offset = conf->sbufin_size;
  conf->dbufin_offset = conf->sbufin_size / sizeof(int64_t);
  conf->bufrt1_offset = conf->sbufrt1_size / sizeof(cufftComplex);
  conf->bufrt2_offset = conf->sbufrt2_size / sizeof(cufftComplex);
  
  conf->dbufout_offset   = conf->sbufout_size / NBYTE_OUT;
  //conf->hbufout_offset   = conf->sbufout_size / sizeof(char);
  conf->hbufout_offset   = conf->sbufout_size;

  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_in, conf->bufin_size));
  
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out, conf->bufout_size));       
  CudaSafeCall(cudaMalloc((void **)&conf->ddat_offs, NCHAN * sizeof(float)));
  CudaSafeCall(cudaMalloc((void **)&conf->dsquare_mean, NCHAN * sizeof(float)));
  CudaSafeCall(cudaMalloc((void **)&conf->ddat_scl, NCHAN * sizeof(float)));
      
  CudaSafeCall(cudaMemset((void *)conf->ddat_offs, 0, NCHAN * sizeof(float)));   // We have to clear the memory for this parameter
  CudaSafeCall(cudaMemset((void *)conf->dsquare_mean, 0, NCHAN * sizeof(float)));// We have to clear the memory for this parameter
  
  CudaSafeCall(cudaMallocHost((void **)&conf->hdat_scl, NCHAN * sizeof(float)));   // Malloc host memory to receive data from device
  CudaSafeCall(cudaMallocHost((void **)&conf->hdat_offs, NCHAN * sizeof(float)));   // Malloc host memory to receive data from device
  CudaSafeCall(cudaMallocHost((void **)&conf->hsquare_mean, NCHAN * sizeof(float)));   // Malloc host memory to receive data from device
  
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt1, conf->bufrt1_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt2, conf->bufrt2_size)); 

  /* Prepare the setup of kernels */
  conf->gridsize_unpack.x = conf->stream_ndf_chk;
  conf->gridsize_unpack.y = NCHK_CAPTURE;
  conf->gridsize_unpack.z = 1;
  conf->blocksize_unpack.x = NSAMP_DF; 
  conf->blocksize_unpack.y = NCHAN_CHK;
  conf->blocksize_unpack.z = 1;
  
  conf->gridsize_swap_select_transpose_swap.x = NCHK_CAPTURE * NCHAN_CHK;
  conf->gridsize_swap_select_transpose_swap.y = conf->stream_ndf_chk * NSAMP_DF / CUFFT_NX1;
  conf->gridsize_swap_select_transpose_swap.z = 1;  
  conf->blocksize_swap_select_transpose_swap.x = CUFFT_NX1;
  conf->blocksize_swap_select_transpose_swap.y = 1;
  conf->blocksize_swap_select_transpose_swap.z = 1;
  
  conf->gridsize_mean.x = 1; 
  conf->gridsize_mean.y = 1; 
  conf->gridsize_mean.z = 1;
  conf->blocksize_mean.x = NCHAN; 
  conf->blocksize_mean.y = 1;
  conf->blocksize_mean.z = 1;
  
  conf->gridsize_scale.x = 1;
  conf->gridsize_scale.y = 1;
  conf->gridsize_scale.z = 1;
  conf->blocksize_scale.x = NCHAN;
  conf->blocksize_scale.y = 1;
  conf->blocksize_scale.z = 1;
  
  conf->gridsize_transpose_pad.x = conf->stream_ndf_chk * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_pad.y = NCHAN;
  conf->gridsize_transpose_pad.z = 1;
  conf->blocksize_transpose_pad.x = CUFFT_NX2;
  conf->blocksize_transpose_pad.y = 1;
  conf->blocksize_transpose_pad.z = 1;

  conf->gridsize_sum1.x = NCHAN;
  conf->gridsize_sum1.y = conf->stream_ndf_chk * NPOL_SAMP;
  conf->gridsize_sum1.z = 1;
  conf->blocksize_sum1.x = NSAMP_DF * CUFFT_NX2 / (2 * CUFFT_NX1);  // This is the right setup if CUFFT_NX2 is not equal to CUFFT_NX1
  conf->blocksize_sum1.y = 1;
  conf->blocksize_sum1.z = 1;
  
  conf->gridsize_sum2.x = NCHAN;
  conf->gridsize_sum2.y = 1;
  conf->gridsize_sum2.z = 1;
  conf->blocksize_sum2.x = conf->stream_ndf_chk * NPOL_SAMP / 2;
  conf->blocksize_sum2.y = 1;
  conf->blocksize_sum2.z = 1;
  
  conf->gridsize_transpose_scale.x = conf->stream_ndf_chk * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_scale.y = NCHAN / TILE_DIM;
  conf->gridsize_transpose_scale.z = 1;
  conf->blocksize_transpose_scale.x = TILE_DIM;
  conf->blocksize_transpose_scale.y = NROWBLOCK_TRANS;
  conf->blocksize_transpose_scale.z = 1;
  
  conf->gridsize_transpose_float.x = conf->stream_ndf_chk * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_float.y = NCHAN / TILE_DIM;
  conf->gridsize_transpose_float.z = 1;
  conf->blocksize_transpose_float.x = TILE_DIM;
  conf->blocksize_transpose_float.y = NROWBLOCK_TRANS;
  conf->blocksize_transpose_float.z = 1;
  
  /* attach to input ring buffer */
  conf->hdu_in = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu_in, conf->key_in);
  if(dada_hdu_connect(conf->hdu_in) < 0)
    {
      multilog(runtime_log, LOG_ERR, "could not connect to hdu\n");
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }  
  db_in = (ipcbuf_t *) conf->hdu_in->data_block;
  conf->rbufin_size = ipcbuf_get_bufsz(db_in);  
  if(conf->rbufin_size % conf->bufin_size != 0)  
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  dada_cuda_dbregister(conf->hdu_in);  // registers the existing host memory range for use by CUDA   
  hdrsz = ipcbuf_get_bufsz(conf->hdu_in->header_block);  
  if(hdrsz != DADA_HDRSZ)    // This number should match
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  if(dada_hdu_lock_read(conf->hdu_in) < 0) // make ourselves the read client 
    {
      multilog(runtime_log, LOG_ERR, "open_hdu: could not lock write\n");
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  /* Prepare output ring buffer */
  conf->hdu_out = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu_out, conf->key_out);
  if(dada_hdu_connect(conf->hdu_out) < 0)
    {
      multilog(runtime_log, LOG_ERR, "could not connect to hdu\n");
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  db_out = (ipcbuf_t *) conf->hdu_out->data_block;
  conf->rbufout_size = ipcbuf_get_bufsz(db_out);
  if(conf->rbufout_size % conf->bufout_size != 0)  
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }  
  hdrsz = ipcbuf_get_bufsz(conf->hdu_out->header_block);  
  if(hdrsz != DADA_HDRSZ)    // This number should match
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }  
  if(dada_hdu_lock_write(conf->hdu_out) < 0)   // make ourselves the write client 
    {
      multilog(runtime_log, LOG_ERR, "open_hdu: could not lock write\n");
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if(ipcbuf_disable_sod(db_out) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  quit = 0;
  
  return EXIT_SUCCESS;
}

void *baseband2baseband(void *conf)
{
  /*
    The whole procedure for fold mode is :
    1. Unpack the data and reorder it from TFTFP to PFT order, prepare for the forward FFT;
    2. Forward FFT the PFT data to get finer channelzation and the data is in PFTF order after FFT;
    3. Swap the FFT output to put the frequency centre on the right place, drop frequency channel edge and band edge and put the data into PTF order, swap the data and put the centre frequency at bin 0 for each FFT block, prepare for inverse FFT;
    4. Inverse FFT the data to get PTFT order data;
    5. Transpose the data to get TFP data and scale it;    
  */
  conf_t *b2bconf = (conf_t *)conf;
  uint64_t i, j;
  uint64_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset, hbufout_offset, dbufout_offset;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap;
  dim3 gridsize_transpose_scale, blocksize_transpose_scale;
  dim3 gridsize_transpose_float, blocksize_transpose_float;
  uint64_t curbufsz;
  int quit_status;
  ipcbuf_t *db_in = NULL, *db_out = NULL;
  
  db_in  = (ipcbuf_t *)b2bconf->hdu_in->data_block;
  db_out = (ipcbuf_t *)b2bconf->hdu_out->data_block;
  
  gridsize_unpack                      = b2bconf->gridsize_unpack;
  blocksize_unpack                     = b2bconf->blocksize_unpack;
  gridsize_swap_select_transpose_swap  = b2bconf->gridsize_swap_select_transpose_swap;   
  blocksize_swap_select_transpose_swap = b2bconf->blocksize_swap_select_transpose_swap;  
  gridsize_transpose_scale             = b2bconf->gridsize_transpose_scale;
  blocksize_transpose_scale            = b2bconf->blocksize_transpose_scale;
  gridsize_transpose_float             = b2bconf->gridsize_transpose_float;
  blocksize_transpose_float            = b2bconf->blocksize_transpose_float;

  
  pthread_mutex_lock(&quit_mutex);
  quit_status = quit;
  pthread_mutex_unlock(&quit_mutex);
  
  /* Do the real job */  
  while(!ipcbuf_eod(db_in) && (quit_status == 0))
    // The first time we open a block at the scale calculation, we need to make sure that the input ring buffer block is bigger than the block needed for scale calculation
    // Otherwise we have to open couple of blocks to calculate scales and these blocks will dropped after that
    {      
      b2bconf->curbuf_in  = ipcbuf_get_next_read(db_in, &curbufsz);
      b2bconf->curbuf_out = ipcbuf_get_next_write(db_out);
      
      for(i = 0; i < b2bconf->nrun_blk; i ++)
	{
	  for(j = 0; j < b2bconf->nstream; j++)
	    {
	      hbufin_offset = j * b2bconf->hbufin_offset + i * b2bconf->bufin_size;
	      dbufin_offset = j * b2bconf->dbufin_offset; 
	      bufrt1_offset = j * b2bconf->bufrt1_offset;
	      bufrt2_offset = j * b2bconf->bufrt2_offset;

	      dbufout_offset = j * b2bconf->dbufout_offset;
	      hbufout_offset = j * b2bconf->hbufout_offset + i * b2bconf->bufout_size;
	      
	      CudaSafeCall(cudaMemcpyAsync(&b2bconf->dbuf_in[dbufin_offset], &b2bconf->curbuf_in[hbufin_offset], b2bconf->sbufin_size, cudaMemcpyHostToDevice, b2bconf->streams[j]));
	      
	      /* Unpack raw data into cufftComplex array */
	      unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, b2bconf->streams[j]>>>(&b2bconf->dbuf_in[dbufin_offset], &b2bconf->buf_rt1[bufrt1_offset], b2bconf->nsamp1);
	      
	      /* Do forward FFT */
	      CufftSafeCall(cufftExecC2C(b2bconf->fft_plans1[j], &b2bconf->buf_rt1[bufrt1_offset], &b2bconf->buf_rt1[bufrt1_offset], CUFFT_FORWARD));

	      /* Prepare for inverse FFT */
	      swap_select_transpose_swap_kernel<<<gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap, 0, b2bconf->streams[j]>>>(&b2bconf->buf_rt1[bufrt1_offset], &b2bconf->buf_rt2[bufrt2_offset], b2bconf->nsamp1, b2bconf->nsamp2); 
	      /* Do inverse FFT */
	      CufftSafeCall(cufftExecC2C(b2bconf->fft_plans2[j], &b2bconf->buf_rt2[bufrt2_offset], &b2bconf->buf_rt2[bufrt2_offset], CUFFT_INVERSE));
	      /* Get final output */
	      transpose_scale_kernel<<<gridsize_transpose_scale, blocksize_transpose_scale, 0, b2bconf->streams[j]>>>(&b2bconf->buf_rt2[bufrt2_offset], &b2bconf->dbuf_out[dbufout_offset], b2bconf->nsamp2, b2bconf->ddat_offs, b2bconf->ddat_scl);   
	      /* Copy the final output to host */
	      CudaSafeCall(cudaMemcpyAsync(&b2bconf->curbuf_out[hbufout_offset], &b2bconf->dbuf_out[dbufout_offset], b2bconf->sbufout_size, cudaMemcpyDeviceToHost, b2bconf->streams[j]));
	    }
	  CudaSynchronizeCall(); // Sync here is for multiple streams
	}
      	  
      /* Close current buffer */
      ipcbuf_mark_filled(db_out, curbufsz);
      ipcbuf_mark_cleared(db_in);
      
      pthread_mutex_lock(&quit_mutex);
      quit_status = quit;
      pthread_mutex_unlock(&quit_mutex);
    }
  
  pthread_exit(NULL);
  ipcbuf_enable_eod(db_out);
  
  return NULL;
}

int destroy_baseband2baseband(conf_t conf)
{
  int i;
  
  for (i = 0; i < conf.nstream; i++)
    {
      CudaSafeCall(cudaStreamDestroy(conf.streams[i]));
      CufftSafeCall(cufftDestroy(conf.fft_plans1[i]));
      CufftSafeCall(cufftDestroy(conf.fft_plans2[i]));
    }
  
  cudaFree(conf.dbuf_in);

  cudaFree(conf.dbuf_out);
  cudaFreeHost(conf.hdat_offs);
  cudaFreeHost(conf.hsquare_mean);
  cudaFreeHost(conf.hdat_scl);
  cudaFree(conf.ddat_offs);
  cudaFree(conf.dsquare_mean);
  cudaFree(conf.ddat_scl);
  
  dada_hdu_unlock_write(conf.hdu_out);
  dada_hdu_disconnect(conf.hdu_out);
  dada_hdu_destroy(conf.hdu_out);

  cudaFree(conf.buf_rt1);
  cudaFree(conf.buf_rt2);

  dada_cuda_dbunregister(conf.hdu_in);
  
  dada_hdu_unlock_read(conf.hdu_in);
  dada_hdu_disconnect(conf.hdu_in);
  dada_hdu_destroy(conf.hdu_in);

  free(conf.streams);
  free(conf.fft_plans1);
  free(conf.fft_plans2);
  
  return EXIT_SUCCESS;
}

int register_header(conf_t *conf)
{
  uint64_t hdrsz;
  char *hdrbuf_in, *hdrbuf_out;
  
  hdrbuf_in  = ipcbuf_get_next_read(conf->hdu_in->header_block, &hdrsz);  
  if (!hdrbuf_in)
    {
      multilog(runtime_log, LOG_ERR, "get next header block error.\n");
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if(hdrsz != DADA_HDRSZ)
    {
      multilog(runtime_log, LOG_ERR, "get next header block error.\n");
      fprintf(stderr, "Header size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  hdrbuf_out = ipcbuf_get_next_write(conf->hdu_out->header_block);
  if (!hdrbuf_out)
    {
      multilog(runtime_log, LOG_ERR, "get next header block error.\n");
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header 
  
  if (ascii_header_set(hdrbuf_out, "NCHAN", "%d", NCHAN) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set NCHAN\n");
      fprintf(stderr, "Error setting NCHAN, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_get(hdrbuf_out, "UTC_START", "%s", conf->utc_start) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set NCHAN\n");
      fprintf(stderr, "Error setting NCHAN, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "BW", "%d", NCHAN) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set BW\n");
      fprintf(stderr, "Error setting BW, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "TSAMP", "1.0") < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set TSAMP\n");
      fprintf(stderr, "Error setting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ipcbuf_mark_filled(conf->hdu_in->header_block, DADA_HDRSZ) < 0)      
    {
      multilog(runtime_log, LOG_ERR, "Could not close header block\n");
      fprintf(stderr, "Error mark_filled, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if(ipcbuf_mark_cleared(conf->hdu_in->header_block))  
    {
      multilog(runtime_log, LOG_ERR, "Could not clear header block\n");
      fprintf(stderr, "Error header_clear, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }    

  return EXIT_SUCCESS;
}