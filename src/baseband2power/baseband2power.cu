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
#include "baseband2power.cuh"
#include "cudautil.cuh"
#include "kernel.cuh"

extern multilog_t *runtime_log;

int init_baseband2power(conf_t *conf)
{
  int i;
  conf->nsamp_in     = conf->stream_ndf_chk * NCHAN_CHK * NCHK_BEAM * NSAMP_DF; // For each stream, here one stream will produce one final output
  conf->nsamp_rt     = conf->nsamp_in;
  conf->nsamp_out    = NCHAN_CHK * NCHK_BEAM;

  conf->ndata_in     = conf->nsamp_in * NPOL_SAMP * NDIM_POL;
  conf->ndata_rt     = conf->nsamp_rt;
  conf->ndata_out    = conf->nsamp_out;

  conf->sbufin_size  = conf->ndata_in * NBYTE_IN;
  conf->sbufrt_size  = conf->ndata_rt * NBYTE_RT; 
  conf->sbufout_size = conf->ndata_out * NBYTE_OUT; 
  
  conf->bufin_size  = conf->nstream * conf->sbufin_size;
  conf->bufrt_size  = conf->nstream * conf->sbufrt_size;
  conf->bufout_size = conf->nstream * conf->sbufout_size;
  
  conf->hbufin_offset = conf->sbufin_size;
  conf->dbufin_offset = conf->sbufin_size / (NBYTE_IN * NPOL_SAMP * NDIM_POL);
  conf->bufrt1_offset = conf->sbufrt_size / NBYTE_RT;
  conf->bufrt2_offset = conf->sbufrt_size / NBYTE_RT;
  
  conf->dbufout_offset   = conf->sbufout_size / NBYTE_OUT;
  conf->hbufout_offset   = conf->sbufout_size;
  
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_in, conf->bufin_size));
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out, conf->bufout_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt1, conf->bufrt_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt2, conf->bufrt_size));
  
  conf->streams = (cudaStream_t *)malloc(conf->nstream * sizeof(cudaStream_t));  
  for(i = 0; i < conf->nstream; i ++)
    CudaSafeCall(cudaStreamCreate(&conf->streams[i]));
  
  /* Prepare the setup of kernels */
  conf->gridsize_unpack_detect.x = conf->stream_ndf_chk;
  conf->gridsize_unpack_detect.y = NCHK_BEAM;
  conf->gridsize_unpack_detect.z = 1;
  conf->blocksize_unpack_detect.x = NSAMP_DF; 
  conf->blocksize_unpack_detect.y = NCHAN_CHK;
  conf->blocksize_unpack_detect.z = 1;

  conf->gridsize_sum1.x = NCHK_BEAM * NCHAN_CHK;
  conf->gridsize_sum1.y = conf->stream_ndf_chk * NSAMP_DF / (2 * SUM1_BLKSZ);
  conf->gridsize_sum1.z = 1;
  conf->blocksize_sum1.x = SUM1_BLKSZ;
  conf->blocksize_sum1.y = 1;
  conf->blocksize_sum1.z = 1;

  conf->gridsize_sum2.x = NCHK_BEAM * NCHAN_CHK;
  conf->gridsize_sum2.y = 1;
  conf->gridsize_sum2.z = 1;
  conf->blocksize_sum2.x = conf->stream_ndf_chk * NSAMP_DF / (4 * SUM1_BLKSZ);
  conf->blocksize_sum2.y = 1;
  conf->blocksize_sum2.z = 1;

  /* Attach to input ring buffer */
  conf->hdu_in = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu_in, conf->key_in);
  if(dada_hdu_connect(conf->hdu_in) < 0)
    {
      multilog(runtime_log, LOG_ERR, "could not connect to hdu\n");
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }  
  conf->db_in = (ipcbuf_t *) conf->hdu_in->data_block;
  if(ipcbuf_get_bufsz(conf->db_in) != conf->bufin_size)
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }

  /* registers the existing host memory range for use by CUDA */
  dada_cuda_dbregister(conf->hdu_in);
  
  if(ipcbuf_get_bufsz(conf->hdu_in->header_block) != DADA_HDRSZ)    // This number should match
    {
      multilog(runtime_log, LOG_ERR, "Header buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  
  /* make ourselves the read client */
  if(dada_hdu_lock_read(conf->hdu_in) < 0)
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
  conf->db_out = (ipcbuf_t *) conf->hdu_out->data_block;
  if(ipcbuf_get_bufsz(conf->db_out) != conf->bufout_size)  
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  
  if(ipcbuf_get_bufsz(conf->hdu_out->header_block) != DADA_HDRSZ)    // This number should match
    {
      multilog(runtime_log, LOG_ERR, "Header buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }  
  /* make ourselves the write client */
  if(dada_hdu_lock_write(conf->hdu_out) < 0)
    {
      multilog(runtime_log, LOG_ERR, "open_hdu: could not lock write\n");
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  /* Register header */
  if(register_header(conf))
    {
      multilog(runtime_log, LOG_ERR, "header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
    
  return EXIT_SUCCESS;
}

int destroy_baseband2power(conf_t conf)
{
  int i;
  for (i = 0; i < conf.nstream; i++)
    CudaSafeCall(cudaStreamDestroy(conf.streams[i]));
  
  cudaFree(conf.dbuf_in);
  cudaFree(conf.dbuf_out);
  cudaFree(conf.buf_rt1);
  cudaFree(conf.buf_rt2);

  dada_cuda_dbunregister(conf.hdu_in);

  dada_hdu_unlock_read(conf.hdu_in);
  dada_hdu_disconnect(conf.hdu_in);
  dada_hdu_destroy(conf.hdu_in);

  dada_hdu_unlock_write(conf.hdu_out);
  dada_hdu_disconnect(conf.hdu_out);
  dada_hdu_destroy(conf.hdu_out);
  
  return EXIT_SUCCESS;
}

int baseband2power(conf_t conf)
{
  int i, j;
  dim3 gridsize_unpack_detect, blocksize_unpack_detect;
  dim3 gridsize_sum1, blocksize_sum1;
  dim3 gridsize_sum2, blocksize_sum2;
  size_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset, hbufout_offset, dbufout_offset;
  
  gridsize_unpack_detect = conf.gridsize_unpack_detect;
  blocksize_unpack_detect = conf.blocksize_unpack_detect;
  gridsize_sum1 = conf.gridsize_sum1;
  blocksize_sum1 = conf.blocksize_sum1;
  gridsize_sum2 = conf.gridsize_sum2;
  blocksize_sum2 = conf.blocksize_sum2;
  
  while(!ipcbuf_eod(conf.db_in))
    {      
      for(i = 0; i < conf.nrun_blk; i ++)
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
	      
	      unpack_detect_kernel<<<gridsize_unpack_detect, blocksize_unpack_detect, 0, conf.streams[j]>>>(&conf.dbuf_in[dbufin_offset], &conf.buf_rt1[bufrt1_offset]);
	      sum_kernel<<<gridsize_sum1, blocksize_sum1, blocksize_sum1.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset]);
	      sum_kernel<<<gridsize_sum2, blocksize_sum2, blocksize_sum2.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset]);
	      
	      /* Copy the final output to host */
	      CudaSafeCall(cudaMemcpyAsync(&conf.curbuf_out[hbufout_offset], &conf.dbuf_out[dbufout_offset], conf.sbufout_size, cudaMemcpyDeviceToHost, conf.streams[j]));
	    }
	  CudaSynchronizeCall(); // Sync here is for multiple streams
	}
      
      /* Close current buffer */
      ipcbuf_mark_filled(conf.db_out, conf.bufout_size);
      ipcbuf_mark_cleared(conf.db_in);      
    }
  
  return EXIT_SUCCESS;
}

int register_header(conf_t *conf)
{
  uint64_t hdrsz, file_size, bytes_per_seconds;
  char *hdrbuf_in, *hdrbuf_out;
  double scale;
  
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
  if (ascii_header_get(hdrbuf_in, "FILE_SIZE", "%"PRIu64"", &file_size) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_get FILE_SIZE\n");
      fprintf(stderr, "Error getting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }   
  if (ascii_header_get(hdrbuf_in, "BYTES_PER_SECOND", "%"PRIu64"", &bytes_per_seconds) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_get BYTES_PER_SECOND\n");
      fprintf(stderr, "Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header
  scale = (double)NBYTE_OUT/(conf->stream_ndf_chk * NSAMP_DF * NPOL_SAMP * NDIM_POL * NBYTE_IN);
  file_size = (uint64_t)(file_size * scale);
  bytes_per_seconds = (uint64_t)(bytes_per_seconds * scale);
  
  if (ascii_header_set(hdrbuf_out, "TSAMP", "%lf", TSAMP * NSAMP_DF * conf->stream_ndf_chk) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set TSAMP\n");
      fprintf(stderr, "Error setting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "NBIT", "%d", NBYTE_OUT * 8) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set NBIT\n");
      fprintf(stderr, "Error setting NBIT, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "FILE_SIZE", "%"PRIu64"", file_size) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set NBIT\n");
      fprintf(stderr, "Error setting NBIT, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "BYTES_PER_SECOND", "%"PRIu64"", bytes_per_seconds) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set BYTES_PER_SECOND\n");
      fprintf(stderr, "Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ipcbuf_mark_filled(conf->hdu_out->header_block, DADA_HDRSZ) < 0)      
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
