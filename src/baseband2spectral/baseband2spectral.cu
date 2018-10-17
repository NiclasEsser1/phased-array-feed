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
#include "baseband2spectral.cuh"
#include "cudautil.cuh"
#include "kernel.cuh"

extern multilog_t *runtime_log;

int init_baseband2spectral(conf_t *conf)
{
  int i;
  int iembed, istride, idist, oembed, ostride, odist, batch, nx;
  
  /* Prepare buffer, stream and fft plan for process */
  conf->nsamp_in       = conf->stream_ndf_chk * NCHAN_IN * NSAMP_DF;
  conf->npol_in        = conf->nsamp_in * NPOL_IN;
  conf->ndata_in       = conf->npol_in  * NDIM_IN;
  
  conf->nsamp_out      = NCHAN_KEEP_BAND;
  conf->npol_out       = conf->nsamp_out * NPOL_OUT;
  conf->ndata_out      = conf->npol_out  * NDIM_OUT;

  conf->nsamp_rtc      = conf->nsamp_in;
  conf->npol_rtc       = conf->nsamp_rtc * NPOL_IN;
  conf->ndata_rtc      = conf->npol_rtc  * NDIM_IN;

  conf->nsamp_rtf1     = conf->nsamp_in * OSAMP_RATEI;
  conf->npol_rtf1      = conf->nsamp_rtf1 * NPOL_OUT;
  conf->ndata_rtf1     = conf->npol_rtf1   * NDIM_OUT;
  
  conf->nsamp_rtf2     = conf->nsamp_rtf1 / (2 * SUM1_BLKSZ);
  conf->npol_rtf2      = conf->nsamp_rtf2 * NPOL_OUT;
  conf->ndata_rtf2     = conf->npol_rtf2   * NDIM_OUT;
  
  nx        = CUFFT_NX;
  batch     = conf->npol_rtc / CUFFT_NX;
  
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
  
  conf->sbufin_size  = conf->ndata_in * NBYTE_IN;
  conf->sbufout_size = conf->ndata_out * NBYTE_OUT;  
  conf->sbufrtc_size  = conf->npol_rtc * NBYTE_RT_C;
  conf->sbufrtf1_size = conf->npol_rtf1 * NBYTE_RT_F;
  conf->sbufrtf2_size = conf->npol_rtf2 * NBYTE_RT_F;

  conf->bufin_size   = conf->nstream * conf->sbufin_size;
  conf->bufout_size  = conf->nstream * conf->sbufout_size;
  conf->bufrtc_size  = conf->nstream * conf->sbufrtc_size;
  conf->bufrtf1_size = conf->nstream * conf->sbufrtf1_size;
  conf->bufrtf2_size = conf->nstream * conf->sbufrtf2_size;
    
  conf->hbufin_offset = conf->sbufin_size;
  conf->dbufin_offset = conf->sbufin_size / (NBYTE_IN * NPOL_IN * NDIM_IN);
  conf->bufrtc_offset  = conf->sbufrtc_size / NBYTE_RT_C;
  conf->bufrtf1_offset = conf->sbufrtf1_size / NBYTE_RT_F;
  conf->bufrtf2_offset = conf->sbufrtf2_size / NBYTE_RT_F;  
  conf->dbufout_offset = conf->sbufout_size / NBYTE_OUT;
  conf->hbufout_offset = conf->sbufout_size;

  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_in,  conf->bufin_size));  
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out, conf->bufout_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rtc,  conf->bufrtc_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rtf1, conf->bufrtf1_size)); 
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rtf2, conf->bufrtf2_size)); 

  /* Prepare the setup of kernels */
  conf->gridsize_unpack.x = conf->stream_ndf_chk;
  conf->gridsize_unpack.y = NCHK_BEAM;
  conf->gridsize_unpack.z = 1;
  conf->blocksize_unpack.x = NSAMP_DF; 
  conf->blocksize_unpack.y = NCHAN_CHK;
  conf->blocksize_unpack.z = 1;
  
  conf->gridsize_swap_select_transpose_detect.x = NCHAN_IN;
  conf->gridsize_swap_select_transpose_detect.y = conf->stream_ndf_chk * NSAMP_DF / CUFFT_NX;
  conf->gridsize_swap_select_transpose_detect.z = 1;  
  conf->blocksize_swap_select_transpose_detect.x = CUFFT_NX;
  conf->blocksize_swap_select_transpose_detect.y = 1;
  conf->blocksize_swap_select_transpose_detect.z = 1;          

  conf->gridsize_sum1.x = NCHAN_KEEP_BAND;
  conf->gridsize_sum1.y = 1;
  conf->gridsize_sum1.z = conf->ndata_rtf1 / (2 * NCHAN_KEEP_BAND * SUM1_BLKSZ);
  conf->blocksize_sum1.x = SUM1_BLKSZ;
  conf->blocksize_sum1.y = 1;
  conf->blocksize_sum1.z = 1;
  
  conf->gridsize_sum2.x = NCHAN_KEEP_BAND;
  conf->gridsize_sum2.y = 1;
  conf->gridsize_sum2.z = 1;
  conf->blocksize_sum2.x = conf->ndata_rtf1 / (4 * NCHAN_KEEP_BAND * SUM1_BLKSZ);
  conf->blocksize_sum2.y = 1;
  conf->blocksize_sum2.z = 1;
  
  /* attach to input ring buffer */
  conf->hdu_in = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu_in, conf->key_in);
  if(dada_hdu_connect(conf->hdu_in) < 0)
    {
      multilog(runtime_log, LOG_ERR, "could not connect to hdu\n");
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }  
  conf->db_in = (ipcbuf_t *) conf->hdu_in->data_block;
  conf->rbufin_size = ipcbuf_get_bufsz(conf->db_in);

  if(conf->rbufin_size % conf->bufin_size != 0)  
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  
  /* registers the existing host memory range for use by CUDA */
  dada_cuda_dbregister(conf->hdu_in);
        
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_in->header_block);  
  if(conf->hdrsz != DADA_HDRSZ)    // This number should match
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
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
  conf->rbufout_size = ipcbuf_get_bufsz(conf->db_out);
    
  if(conf->rbufout_size % conf->bufout_size != 0)  
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_out->header_block);  
  if(conf->hdrsz != DADA_HDRSZ)    // This number should match
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
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
  
  return EXIT_SUCCESS;
}

int baseband2spectral(conf_t conf)
{
  /*
    The whole procedure for fold mode is :
    1. Unpack the data and reorder it from TFTFP to PFT order, prepare for the forward FFT;
    2. Forward FFT the PFT data to get finer channelzation and the data is in PFTF order after FFT;
    3. Swap the FFT output to put the frequency centre on the right place, drop frequency channel edge and band edge and put the data into PTF order, swap the data and put the centre frequency at bin 0 for each FFT block, prepare for inverse FFT;
    4. Inverse FFT the data to get PTFT order data;
    5. Transpose the data to get TFP data and scale it;    

    The whole procedure for search mode is :
    1. Unpack the data and reorder it from TFTFP to PFT order, prepare for the forward FFT;
    2. Forward FFT the PFT data to get finer channelzation and the data is in PFTF order after FFT;
    3. Swap the FFT output to put the frequency centre on the right place, drop frequency channel edge and band edge and put the data into PTF order;
    4. Add the data in frequency to get NCHAN channels, detect the added data and scale it;
  */
  uint64_t i, j;
  uint64_t hbufin_offset, dbufin_offset, bufrtc_offset, bufrtf1_offset, bufrtf2_offset, hbufout_offset, dbufout_offset;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_detect, blocksize_swap_select_transpose_detect;
  dim3 gridsize_sum1, blocksize_sum1;
  dim3 gridsize_sum2, blocksize_sum2;
  uint64_t curbufsz;
  
  gridsize_unpack                        = conf.gridsize_unpack;
  blocksize_unpack                       = conf.blocksize_unpack;
  gridsize_swap_select_transpose_detect  = conf.gridsize_swap_select_transpose_detect;   
  blocksize_swap_select_transpose_detect = conf.blocksize_swap_select_transpose_detect;
  gridsize_sum1                          = conf.gridsize_sum1;   
  blocksize_sum1                         = conf.blocksize_sum1;
  gridsize_sum2                          = conf.gridsize_sum2;   
  blocksize_sum2                         = conf.blocksize_sum2; 
       
  /* Register header */
  if(register_header(&conf))
    {
      multilog(runtime_log, LOG_ERR, "header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
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

	      bufrtc_offset = j * conf.bufrtc_offset;
	      bufrtf1_offset = j * conf.bufrtf1_offset;
	      bufrtf2_offset = j * conf.bufrtf2_offset;
	      
	      dbufout_offset = j * conf.dbufout_offset;
	      hbufout_offset = j * conf.hbufout_offset + i * conf.bufout_size;
		      
	      /* Copy data into device */
	      CudaSafeCall(cudaMemcpyAsync(&conf.dbuf_in[dbufin_offset], &conf.curbuf_in[hbufin_offset], conf.sbufin_size, cudaMemcpyHostToDevice, conf.streams[j]));
	      
	      /* Unpack raw data into cufftComplex array */
	      unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, conf.streams[j]>>>(&conf.dbuf_in[dbufin_offset], &conf.buf_rtc[bufrtc_offset], conf.nsamp_in);
	      
	      /* Do forward FFT */
	      CufftSafeCall(cufftExecC2C(conf.fft_plans[j], &conf.buf_rtc[bufrtc_offset], &conf.buf_rtc[bufrtc_offset], CUFFT_FORWARD));

	      swap_select_transpose_detect_kernel<<<gridsize_swap_select_transpose_detect, blocksize_swap_select_transpose_detect, 0, conf.streams[j]>>>(&conf.buf_rtc[bufrtc_offset], &conf.buf_rtf1[bufrtf1_offset], conf.nsamp_rtc);
	      sum_kernel<<<gridsize_sum1, blocksize_sum1, blocksize_sum1.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rtf1[bufrtf1_offset], &conf.buf_rtf2[bufrtf2_offset]);
	      sum_kernel<<<gridsize_sum2, blocksize_sum2, blocksize_sum2.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rtf2[bufrtf2_offset], &conf.dbuf_out[dbufout_offset]);
	      
	      CudaSafeCall(cudaMemcpyAsync(&conf.curbuf_out[hbufout_offset], &conf.dbuf_out[dbufout_offset], conf.sbufout_size, cudaMemcpyDeviceToHost, conf.streams[j]));
	    }
	  CudaSynchronizeCall(); // Sync here is for multiple streams
	}
      
      ipcbuf_mark_filled(conf.db_out, conf.bufout_size);
      ipcbuf_mark_cleared(conf.db_in);       
    }
  
  return EXIT_SUCCESS;
}

int destroy_baseband2spectral(conf_t conf)
{
  int i;
  for (i = 0; i < conf.nstream; i++)
    {
      CudaSafeCall(cudaStreamDestroy(conf.streams[i]));
      CufftSafeCall(cufftDestroy(conf.fft_plans[i]));
    }
  
  cudaFree(conf.dbuf_in);
  cudaFree(conf.dbuf_out);
  dada_hdu_unlock_write(conf.hdu_out);
  dada_hdu_destroy(conf.hdu_out);
  
  cudaFree(conf.buf_rtc);
  cudaFree(conf.buf_rtf1);
  cudaFree(conf.buf_rtf2);

  dada_cuda_dbunregister(conf.hdu_in);
  
  dada_hdu_unlock_read(conf.hdu_in);
  dada_hdu_destroy(conf.hdu_in);

  free(conf.streams);
  free(conf.fft_plans);
  
  return EXIT_SUCCESS;
}

int register_header(conf_t *conf)
{
  double tsamp, scale;
  char *hdrbuf_in = NULL, *hdrbuf_out = NULL;
  uint64_t file_size, bytes_per_seconds, hdrsz;
  
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
  if (ascii_header_get(hdrbuf_in, "TSAMP", "%lf", &tsamp) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_get TSAMP\n");
      fprintf(stderr, "Error getting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header
  
  scale = (double)NBYTE_OUT/(conf->stream_ndf_chk * NSAMP_DF * NPOL_IN * NDIM_IN * NBYTE_IN) * OSAMP_RATEI;
  file_size = (uint64_t)(file_size * scale);
  bytes_per_seconds = (uint64_t)(bytes_per_seconds * scale);
  
  if (ascii_header_set(hdrbuf_out, "NCHAN", "%d", NCHAN_KEEP_BAND) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set NCHAN\n");
      fprintf(stderr, "Error setting NCHAN, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "BW", "%lf", BW_OUT) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set BW\n");
      fprintf(stderr, "Error setting BW, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "TSAMP", "%lf", tsamp) < 0)  
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
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set FILE_SIZE\n");
      fprintf(stderr, "Error setting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "BYTES_PER_SECOND", "%"PRIu64"", bytes_per_seconds) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set BYTES_PER_SECOND\n");
      fprintf(stderr, "Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
    
  if(ipcbuf_mark_cleared (conf->hdu_in->header_block))  // We are the only one reader, so that we can clear it after read;
    {
      multilog(runtime_log, LOG_ERR, "Could not clear header block\n");
      fprintf(stderr, "Error header_clear, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  /* donot set header parameters anymore - acqn. doesn't start */
  if (ipcbuf_mark_filled (conf->hdu_out->header_block, conf->hdrsz) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Could not mark filled header block\n");
      fprintf(stderr, "Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}