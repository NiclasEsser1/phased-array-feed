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
  conf->nsamp_in     = conf->stream_ndf_chk * NCHAN_IN * NSAMP_DF; // For each stream, here one stream will produce one final output
  conf->ndata_in     = conf->nsamp_in * NPOL_IN * NDIM_IN;
  
  conf->nsamp_rt1     = conf->nsamp_in / (2 * SUM1_BLKSZ);
  conf->ndata_rt1     = conf->nsamp_rt1;

  conf->nsamp_rt2     = NCHAN_OUT;
  conf->ndata_rt2     = conf->nsamp_rt2 * NPOL_OUT * NDIM_OUT;
  
  conf->nsamp_out    = NCHAN_OUT;
  conf->ndata_out    = conf->nsamp_out * NPOL_OUT * NDIM_OUT;

  conf->sbufin_size  = conf->ndata_in * NBYTE_IN;
  conf->sbufrt1_size  = conf->ndata_rt1 * NBYTE_RT;
  conf->sbufrt2_size  = conf->ndata_rt2 * NBYTE_RT; 
  conf->sbufout_size = conf->ndata_out * NBYTE_OUT; 
  
  conf->bufin_size  = conf->nstream * conf->sbufin_size;
  conf->bufrt1_size  = conf->nstream * conf->sbufrt1_size;
  conf->bufrt2_size  = conf->nstream * conf->sbufrt2_size;
  conf->bufout_size = conf->nstream * conf->sbufout_size;
  
  conf->hbufin_offset = conf->sbufin_size;
  conf->dbufin_offset = conf->sbufin_size / (NBYTE_IN * NPOL_IN * NDIM_IN);
  conf->bufrt1_offset = conf->sbufrt1_size / NBYTE_RT;
  conf->bufrt2_offset = conf->sbufrt2_size / NBYTE_RT;
  
  conf->dbufout_offset   = conf->sbufout_size / NBYTE_OUT;
  conf->hbufout_offset   = conf->sbufout_size;
  
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_in, conf->bufin_size));
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out, conf->bufout_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt1, conf->bufrt1_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt2, conf->bufrt2_size));
  
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

  conf->gridsize_sum1.x = NCHAN_OUT;
  conf->gridsize_sum1.y = conf->stream_ndf_chk * NSAMP_DF / (2 * SUM1_BLKSZ);
  conf->gridsize_sum1.z = 1;
  conf->blocksize_sum1.x = SUM1_BLKSZ;
  conf->blocksize_sum1.y = 1;
  conf->blocksize_sum1.z = 1;

  conf->gridsize_sum2.x = NCHAN_OUT;
  conf->gridsize_sum2.y = 1;
  conf->gridsize_sum2.z = 1;
  conf->blocksize_sum2.x = conf->stream_ndf_chk * NSAMP_DF / (4 * SUM1_BLKSZ);
  conf->blocksize_sum2.y = 1;
  conf->blocksize_sum2.z = 1;

  conf->scale = (double)NBYTE_OUT * NPOL_OUT * NDIM_OUT / (conf->stream_ndf_chk * NSAMP_DF * NPOL_IN * NDIM_IN * NBYTE_IN);
  
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
  fprintf(stdout, "%"PRIu64"\t%"PRIu64"\n", conf->bufin_size, ipcbuf_get_bufsz(conf->db_in));
  if(ipcbuf_get_bufsz(conf->db_in) % conf->bufin_size)
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
  fprintf(stdout, "%"PRIu64"\t%"PRIu64"\n", conf->bufout_size, ipcbuf_get_bufsz(conf->db_out));
  if(ipcbuf_get_bufsz(conf->db_out) % conf->bufout_size)  
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
  size_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset, hbufout_offset, dbufout_offset, curbufsz;
  
  gridsize_unpack_detect = conf.gridsize_unpack_detect;
  blocksize_unpack_detect = conf.blocksize_unpack_detect;
  gridsize_sum1 = conf.gridsize_sum1;
  blocksize_sum1 = conf.blocksize_sum1;
  gridsize_sum2 = conf.gridsize_sum2;
  blocksize_sum2 = conf.blocksize_sum2;
  
  while(!ipcbuf_eod(conf.db_in))
    {      
      conf.curbuf_in  = ipcbuf_get_next_read(conf.db_in, &curbufsz);
      conf.curbuf_out = ipcbuf_get_next_write(conf.db_out);
      
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
      ipcbuf_mark_filled(conf.db_out, (uint64_t)(curbufsz * conf.scale));
      ipcbuf_mark_cleared(conf.db_in);      
    }
  
  return EXIT_SUCCESS;
}

int register_header(conf_t *conf)
{
  uint64_t hdrsz, file_size, bytes_per_seconds;
  char *hdrbuf_in, *hdrbuf_out;
  double tsamp;
  
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
  if (ascii_header_get(hdrbuf_in, "TSAMP", "%lf", &tsamp) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_get TSAMP\n");
      fprintf(stderr, "Error getting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }   
  if (ascii_header_get(hdrbuf_in, "BYTES_PER_SECOND", "%"PRIu64"", &bytes_per_seconds) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_get BYTES_PER_SECOND\n");
      fprintf(stderr, "Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header
  file_size = (uint64_t)(file_size * conf->scale);
  bytes_per_seconds = (uint64_t)(bytes_per_seconds * conf->scale);
  
  if (ascii_header_set(hdrbuf_out, "TSAMP", "%lf", tsamp * NSAMP_DF * conf->stream_ndf_chk) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set TSAMP\n");
      fprintf(stderr, "Error setting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "NBIT", "%d", NBIT_OUT) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set NBIT\n");
      fprintf(stderr, "Error setting NBIT, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "NDIM", "%d", NDIM_OUT) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set NDIM\n");
      fprintf(stderr, "Error setting NDIM, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "NPOL", "%d", NPOL_OUT) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set NPOL\n");
      fprintf(stderr, "Error setting NPOL, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
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
