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
#include "baseband2filterbank.cuh"
#include "cudautil.cuh"
#include "kernel.cuh"

extern multilog_t *runtime_log;

int init_baseband2filterbank(conf_t *conf)
{
  int i;
  int iembed, istride, idist, oembed, ostride, odist, batch, nx;
  
  /* Prepare buffer, stream and fft plan for process */
  conf->nsamp1       = conf->stream_ndf_chk * NCHAN_IN * NSAMP_DF;
  conf->npol1        = conf->nsamp1 * NPOL_SAMP;
  conf->ndata1       = conf->npol1  * NDIM_POL;
  
  conf->nsamp2       = conf->nsamp1 * OSAMP_RATEI / NCHAN_RATEI;
  conf->npol2        = conf->nsamp2 * NPOL_SAMP;
  conf->ndata2       = conf->npol2  * NDIM_POL;

  conf->nsamp3         = conf->nsamp2 * NCHAN_OUT / NCHAN_KEEP_BAND;
  conf->npol3          = conf->nsamp3;
  conf->ndata3         = conf->nsamp3;  

  conf->sclndim = conf->rbufin_ndf_chk * NSAMP_DF / CUFFT_NX;   // We do not average in time and here we work on detected data;
  
  nx        = CUFFT_NX;
  batch     = conf->npol1 / CUFFT_NX;
  
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
      CufftSafeCall(cufftPlanMany(&conf->fft_plans[i], CUFFT_RANK1, &nx, &iembed, istride, idist, &oembed, ostride, odist, CUFFT_C2C, batch));
      CufftSafeCall(cufftSetStream(conf->fft_plans[i], conf->streams[i]));
    }
  
  conf->sbufin_size  = conf->ndata1 * NBYTE_IN;
  conf->sbufout_size = conf->ndata3 * NBYTE_OUT;
  
  conf->bufin_size   = conf->nstream * conf->sbufin_size;
  conf->bufout_size  = conf->nstream * conf->sbufout_size;
  
  conf->sbufrt1_size = conf->npol1 * NBYTE_RT;
  conf->sbufrt2_size = conf->npol2 * NBYTE_RT;
  conf->bufrt1_size  = conf->nstream * conf->sbufrt1_size;
  conf->bufrt2_size  = conf->nstream * conf->sbufrt2_size;
    
  conf->hbufin_offset = conf->sbufin_size;
  conf->dbufin_offset = conf->sbufin_size / sizeof(int64_t);
  conf->bufrt1_offset = conf->sbufrt1_size / NBYTE_RT;
  conf->bufrt2_offset = conf->sbufrt2_size / NBYTE_RT;
  
  conf->dbufout_offset = conf->sbufout_size / NBYTE_OUT;
  conf->hbufout_offset = conf->sbufout_size;

  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_in, conf->bufin_size));  
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out, conf->bufout_size));
  CudaSafeCall(cudaMalloc((void **)&conf->ddat_offs, NCHAN_OUT * sizeof(float)));
  CudaSafeCall(cudaMalloc((void **)&conf->dsquare_mean, NCHAN_OUT * sizeof(float)));
  CudaSafeCall(cudaMalloc((void **)&conf->ddat_scl, NCHAN_OUT * sizeof(float)));
  
  CudaSafeCall(cudaMemset((void *)conf->ddat_offs, 0, NCHAN_OUT * sizeof(float)));   // We have to clear the memory for this parameter
  CudaSafeCall(cudaMemset((void *)conf->dsquare_mean, 0, NCHAN_OUT * sizeof(float)));// We have to clear the memory for this parameter
  
  CudaSafeCall(cudaMallocHost((void **)&conf->hdat_scl, NCHAN_OUT * sizeof(float)));   // Malloc host memory to receive data from device
  CudaSafeCall(cudaMallocHost((void **)&conf->hdat_offs, NCHAN_OUT * sizeof(float)));   // Malloc host memory to receive data from device
  CudaSafeCall(cudaMallocHost((void **)&conf->hsquare_mean, NCHAN_OUT * sizeof(float)));   // Malloc host memory to receive data from device
  
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt1, conf->bufrt1_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt2, conf->bufrt2_size)); 

  /* Prepare the setup of kernels */
  conf->gridsize_unpack.x = conf->stream_ndf_chk;
  conf->gridsize_unpack.y = NCHK_BEAM;
  conf->gridsize_unpack.z = 1;
  conf->blocksize_unpack.x = NSAMP_DF; 
  conf->blocksize_unpack.y = NCHAN_CHK;
  conf->blocksize_unpack.z = 1;
  
  conf->gridsize_swap_select_transpose.x = NCHAN_IN;
  conf->gridsize_swap_select_transpose.y = conf->stream_ndf_chk * NSAMP_DF / CUFFT_NX;
  conf->gridsize_swap_select_transpose.z = 1;  
  conf->blocksize_swap_select_transpose.x = CUFFT_NX;
  conf->blocksize_swap_select_transpose.y = 1;
  conf->blocksize_swap_select_transpose.z = 1;      
    

  /* Only for search mode */
  conf->gridsize_detect_add_pad_transpose.x = conf->stream_ndf_chk * NSAMP_DF / CUFFT_NX;
  conf->gridsize_detect_add_pad_transpose.y = NCHAN_OUT;
  conf->gridsize_detect_add_pad_transpose.z = 1;
  conf->blocksize_detect_add_pad_transpose.x = NCHAN_KEEP_BAND/(2 * NCHAN_OUT);
  conf->blocksize_detect_add_pad_transpose.y = 1;
  conf->blocksize_detect_add_pad_transpose.z = 1;
  
  conf->gridsize_detect_add_scale.x = conf->stream_ndf_chk * NSAMP_DF / CUFFT_NX;
  conf->gridsize_detect_add_scale.y = NCHAN_OUT;
  conf->gridsize_detect_add_scale.z = 1;
  conf->blocksize_detect_add_scale.x = NCHAN_KEEP_BAND/(2 * NCHAN_OUT);
  conf->blocksize_detect_add_scale.y = 1;
  conf->blocksize_detect_add_scale.z = 1;
  
  conf->gridsize_sum.x = NCHAN_OUT;
  conf->gridsize_sum.y = 1;
  conf->gridsize_sum.z = 1;
  conf->blocksize_sum.x = conf->stream_ndf_chk * NSAMP_DF / (CUFFT_NX * 2); 
  conf->blocksize_sum.y = 1;
  conf->blocksize_sum.z = 1;
  
  conf->gridsize_mean.x = 1; 
  conf->gridsize_mean.y = 1; 
  conf->gridsize_mean.z = 1;
  conf->blocksize_mean.x = NCHAN_OUT;
  conf->blocksize_mean.y = 1;
  conf->blocksize_mean.z = 1;
  
  conf->gridsize_scale.x = 1;
  conf->gridsize_scale.y = 1;
  conf->gridsize_scale.z = 1;
  conf->blocksize_scale.x = NCHAN_OUT;
  conf->blocksize_scale.y = 1;
  conf->blocksize_scale.z = 1;
  
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
  //fprintf(stdout, "%"PRIu64"\t%"PRIu64"\n", conf->rbufin_size, conf->bufin_size);
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
  
  //fprintf(stdout, "%"PRIu64"\t%"PRIu64"\n", conf->rbufout_size, conf->bufout_size);
  
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

int baseband2filterbank(conf_t conf)
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
  uint64_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset, hbufout_offset, dbufout_offset;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose, blocksize_swap_select_transpose;
  dim3 gridsize_detect_add_scale, blocksize_detect_add_scale;
  uint64_t curbufsz;
  int first = 1;
  
  gridsize_unpack                      = conf.gridsize_unpack;
  blocksize_unpack                     = conf.blocksize_unpack;
  gridsize_detect_add_scale            = conf.gridsize_detect_add_scale ;
  blocksize_detect_add_scale           = conf.blocksize_detect_add_scale ;
  gridsize_swap_select_transpose       = conf.gridsize_swap_select_transpose;   
  blocksize_swap_select_transpose      = conf.blocksize_swap_select_transpose;  
       
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
      
      /* Get scale of data */
      if(first)
	{
	  first = 0;
	  dat_offs_scl(conf);
	  for(i = 0; i < NCHAN_OUT; i++)
	    fprintf(stdout, "DAT_OFFS:\t%E\tDAT_SCL:\t%E\n", conf.hdat_offs[i], conf.hdat_scl[i]);
	}
      
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
		      
	      /* Copy data into device */
	      CudaSafeCall(cudaMemcpyAsync(&conf.dbuf_in[dbufin_offset], &conf.curbuf_in[hbufin_offset], conf.sbufin_size, cudaMemcpyHostToDevice, conf.streams[j]));
	      
	      /* Unpack raw data into cufftComplex array */
	      unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, conf.streams[j]>>>(&conf.dbuf_in[dbufin_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp1);
	      
	      /* Do forward FFT */
	      CufftSafeCall(cufftExecC2C(conf.fft_plans[j], &conf.buf_rt1[bufrt1_offset], &conf.buf_rt1[bufrt1_offset], CUFFT_FORWARD));

	      swap_select_transpose_kernel<<<gridsize_swap_select_transpose, blocksize_swap_select_transpose, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2); 		  
	      detect_add_scale_kernel<<<gridsize_detect_add_scale, blocksize_detect_add_scale, blocksize_detect_add_scale.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.ddat_offs, conf.ddat_scl);
	      CudaSafeCall(cudaMemcpyAsync(&conf.curbuf_out[hbufout_offset], &conf.dbuf_out[dbufout_offset], conf.sbufout_size, cudaMemcpyDeviceToHost, conf.streams[j]));
	    }
	  CudaSynchronizeCall(); // Sync here is for multiple streams
	}
      
      ipcbuf_mark_filled(conf.db_out, (uint64_t)(curbufsz * SCL_DTSZ));
      ipcbuf_mark_cleared(conf.db_in);       
    }
  
  return EXIT_SUCCESS;
}

int dat_offs_scl(conf_t conf)
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
    2. Add the data in frequency to get NCHAN channels, detect the added data and pad it;
    3. Add the padded data in time;    
    4. Get the mean of the added data;
    5. Get the scale with the mean;
  */
  uint64_t i, j;
  dim3 gridsize_unpack, blocksize_unpack;
  uint64_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset;

  dim3 gridsize_swap_select_transpose, blocksize_swap_select_transpose;
  dim3 gridsize_scale, blocksize_scale;  
  dim3 gridsize_mean, blocksize_mean;
  dim3 gridsize_sum, blocksize_sum;
  dim3 gridsize_detect_add_pad_transpose, blocksize_detect_add_pad_transpose;
  
  char fname[MSTR_LEN];
  FILE *fp=NULL;
    
  gridsize_unpack                      = conf.gridsize_unpack;
  blocksize_unpack                     = conf.blocksize_unpack;
  	         	
  gridsize_sum             = conf.gridsize_sum;	       
  blocksize_sum            = conf.blocksize_sum;
  gridsize_scale           = conf.gridsize_scale;	       
  blocksize_scale          = conf.blocksize_scale;	         		           
  gridsize_mean            = conf.gridsize_mean;	       
  blocksize_mean           = conf.blocksize_mean;
  gridsize_swap_select_transpose  = conf.gridsize_swap_select_transpose;   
  blocksize_swap_select_transpose = conf.blocksize_swap_select_transpose;
  gridsize_detect_add_pad_transpose         = conf.gridsize_detect_add_pad_transpose ;
  blocksize_detect_add_pad_transpose        = conf.blocksize_detect_add_pad_transpose ;
  
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

	  /* Do forward FFT */
	  CufftSafeCall(cufftExecC2C(conf.fft_plans[j], &conf.buf_rt1[bufrt1_offset], &conf.buf_rt1[bufrt1_offset], CUFFT_FORWARD));
	  swap_select_transpose_kernel<<<gridsize_swap_select_transpose, blocksize_swap_select_transpose, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2); 		  
	  detect_add_pad_transpose_kernel<<<gridsize_detect_add_pad_transpose, blocksize_detect_add_pad_transpose, blocksize_detect_add_pad_transpose.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2);
	  sum_kernel<<<gridsize_sum, blocksize_sum, blocksize_sum.x * NBYTE_RT, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset]);
	}
      CudaSynchronizeCall(); // Sync here is for multiple streams

      mean_kernel<<<gridsize_mean, blocksize_mean>>>(conf.buf_rt2, conf.bufrt2_offset, conf.ddat_offs, conf.dsquare_mean, conf.nstream, conf.sclndim);
    }
  
  /* Get the scale of each chanel */
  scale_kernel<<<gridsize_scale, blocksize_scale>>>(conf.ddat_offs, conf.dsquare_mean, conf.ddat_scl);
  CudaSynchronizeCall();
  
  CudaSafeCall(cudaMemcpy(conf.hdat_offs, conf.ddat_offs, sizeof(float) * NCHAN_OUT, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(conf.hdat_scl, conf.ddat_scl, sizeof(float) * NCHAN_OUT, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(conf.hsquare_mean, conf.dsquare_mean, sizeof(float) * NCHAN_OUT, cudaMemcpyDeviceToHost));
  
  /* Record scale into file */
  sprintf(fname, "%s/%s_scale.txt", conf.dir, conf.utc_start);
  fp = fopen(fname, "w");
  if(fp == NULL)
    {
      multilog (runtime_log, LOG_ERR, "Can not open scale file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not open scale file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  for (i = 0; i< NCHAN_OUT; i++)
    fprintf(fp, "%E\t%E\n", conf.hdat_offs[i], conf.hdat_scl[i]);
      
  fclose(fp);
  return EXIT_SUCCESS;
}

int destroy_baseband2filterbank(conf_t conf)
{
  int i;
  for (i = 0; i < conf.nstream; i++)
    {
      CudaSafeCall(cudaStreamDestroy(conf.streams[i]));
      CufftSafeCall(cufftDestroy(conf.fft_plans[i]));
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
  dada_hdu_destroy(conf.hdu_out);
  
  cudaFree(conf.buf_rt1);
  cudaFree(conf.buf_rt2);

  dada_cuda_dbunregister(conf.hdu_in);
  
  dada_hdu_unlock_read(conf.hdu_in);
  dada_hdu_destroy(conf.hdu_in);

  free(conf.streams);
  free(conf.fft_plans);
  
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
  
  /* Get utc_start from hdrin */
  if (ascii_header_get(hdrbuf_in, "UTC_START", "%s", conf->utc_start) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_get UTC_START\n");
      fprintf(stderr, "Error getting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  //fprintf(stdout, "%s\n", conf->utc_start);
  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header
  //scale =  OSAMP_RATEI * (double)NBYTE_OUT/ (NCHAN_RATEI * NSAMP_AVE * NPOL_SAMP * NDIM_POL * (double)NBYTE_IN); 
  //fprintf(stdout, "%.10f\n", scale);
  
  file_size = (uint64_t)(file_size * SCL_DTSZ);
  bytes_per_seconds = (uint64_t)(bytes_per_seconds * SCL_DTSZ);
  
  if (ascii_header_set(hdrbuf_out, "NCHAN", "%d", NCHAN_OUT) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set NCHAN\n");
      fprintf(stderr, "Error setting NCHAN, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "BW", "%lf", BW) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set BW\n");
      fprintf(stderr, "Error setting BW, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "TSAMP", "%lf", TSAMP) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set TSAMP\n");
      fprintf(stderr, "Error setting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "NBIT", "%d", NBIT) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set NBIT\n");
      fprintf(stderr, "Error setting NBIT, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "NDIM", "%d", NDIM) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "failed ascii_header_set NDIM\n");
      fprintf(stderr, "Error setting NDIM, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if (ascii_header_set(hdrbuf_out, "NPOL", "%d", NPOL) < 0)  
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