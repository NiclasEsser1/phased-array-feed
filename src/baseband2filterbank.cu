#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>

#include "baseband2filterbank.cuh"
#include "cudautil.cuh"
#include "kernel.cuh"
#include "log.h"

extern pthread_mutex_t log_mutex;

int init_baseband2filterbank(conf_t *conf)
{
  int i;
  int iembed, istride, idist, oembed, ostride, odist, batch, nx;

  /* Prepare parameters */
  conf->nchan_in        = conf->nchk_in * NCHAN_CHK;
  conf->cufft_mod       = (int)(0.5 * conf->cufft_nx * OSAMP_RATEI);
  conf->nchan_keep_chan = (int)(conf->cufft_nx * OSAMP_RATEI);
  conf->nchan_edge      = (int)(0.5 * conf->nchan_in * conf->nchan_keep_chan - 0.5 * conf->nchan_keep_band);
  conf->nchan_ratei     = conf->nchan_in * conf->nchan_keep_chan/(double)conf->nchan_keep_band;
  conf->scl_dtsz        = OSAMP_RATEI * NBYTE_OUT * conf->nchan_out/ (double)(conf->nchan_ratei * conf->nchan_keep_band * NPOL_IN * NDIM_IN * NBYTE_IN);
  conf->bw              = conf->nchan_keep_band/(double)conf->nchan_keep_chan;

  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "We keep %d channels for input", conf->nchan_in);
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "The mod to reduce oversampling is %d", conf->cufft_mod);
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "We will keep %d fine channels for each input channel after FFT", conf->nchan_keep_chan);
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "We will drop %d fine channels at the band edge for frequency accumulation", conf->nchan_edge);
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "%f percent fine channels (after down sampling) are kept for frequency accumulation", conf->nchan_ratei * 100.);
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "The data size rate between filterbank and baseband data is %f", conf->scl_dtsz);
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "The bandwidth for the final output is %f MHz", conf->bw);
  
  /* Prepare buffer, stream and fft plan for process */
  conf->nsamp1       = conf->stream_ndf_chk * conf->nchan_in * NSAMP_DF;
  conf->npol1        = conf->nsamp1 * NPOL_IN;
  conf->ndata1       = conf->npol1  * NDIM_IN;
  
  conf->nsamp2       = conf->nsamp1 * OSAMP_RATEI / conf->nchan_ratei;
  conf->npol2        = conf->nsamp2 * NPOL_IN;
  conf->ndata2       = conf->npol2  * NDIM_IN;

  conf->nsamp3       = conf->nsamp2 * conf->nchan_out / conf->nchan_keep_band;
  conf->npol3        = conf->nsamp3 * NPOL_OUT;
  conf->ndata3       = conf->npol3  * NDIM_OUT;  

  conf->sclndim = conf->rbufin_ndf_chk * NSAMP_DF / conf->cufft_nx;   // We do not average in time and here we work on detected data;
  
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
  
  conf->sbufin_size  = conf->ndata1 * NBYTE_IN;
  conf->sbufout_size = conf->ndata3 * NBYTE_OUT;
  
  conf->bufin_size   = conf->nstream * conf->sbufin_size;
  conf->bufout_size  = conf->nstream * conf->sbufout_size;
  
  conf->sbufrt1_size = conf->npol1 * NBYTE_RT;
  conf->sbufrt2_size = conf->npol2 * NBYTE_RT;
  conf->bufrt1_size  = conf->nstream * conf->sbufrt1_size;
  conf->bufrt2_size  = conf->nstream * conf->sbufrt2_size;
    
  conf->hbufin_offset = conf->sbufin_size;
  conf->dbufin_offset = conf->sbufin_size / (NBYTE_IN * NPOL_IN * NDIM_IN);
  conf->bufrt1_offset = conf->sbufrt1_size / NBYTE_RT;
  conf->bufrt2_offset = conf->sbufrt2_size / NBYTE_RT;
  
  conf->dbufout_offset = conf->sbufout_size / NBYTE_OUT;
  conf->hbufout_offset = conf->sbufout_size;

  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_in, conf->bufin_size));  
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out, conf->bufout_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt1, conf->bufrt1_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt2, conf->bufrt2_size));
  
  CudaSafeCall(cudaMalloc((void **)&conf->ddat_offs, conf->nchan_out * sizeof(float)));
  CudaSafeCall(cudaMalloc((void **)&conf->dsquare_mean, conf->nchan_out * sizeof(float)));
  CudaSafeCall(cudaMalloc((void **)&conf->ddat_scl, conf->nchan_out * sizeof(float)));
  CudaSafeCall(cudaMemset((void *)conf->ddat_offs, 0, conf->nchan_out * sizeof(float)));   // We have to clear the memory for this parameter
  CudaSafeCall(cudaMemset((void *)conf->dsquare_mean, 0, conf->nchan_out * sizeof(float)));// We have to clear the memory for this parameter
  
  CudaSafeCall(cudaMallocHost((void **)&conf->hdat_scl, conf->nchan_out * sizeof(float)));   // Malloc host memory to receive data from device
  CudaSafeCall(cudaMallocHost((void **)&conf->hdat_offs, conf->nchan_out * sizeof(float)));   // Malloc host memory to receive data from device
  CudaSafeCall(cudaMallocHost((void **)&conf->hsquare_mean, conf->nchan_out * sizeof(float)));   // Malloc host memory to receive data from device

  /* Prepare the setup of kernels */
  conf->gridsize_unpack.x = conf->stream_ndf_chk;
  conf->gridsize_unpack.y = conf->nchk_in;
  conf->gridsize_unpack.z = 1;
  conf->blocksize_unpack.x = NSAMP_DF; 
  conf->blocksize_unpack.y = NCHAN_CHK;
  conf->blocksize_unpack.z = 1;
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "The configuration of unpack kernel is (%d, %d, %d) and (%d, %d, %d)",
	      conf->gridsize_unpack.x, conf->gridsize_unpack.y, conf->gridsize_unpack.z,
	      conf->blocksize_unpack.x, conf->blocksize_unpack.y, conf->blocksize_unpack.z);
  
  conf->gridsize_swap_select_transpose.x = conf->nchan_in;
  conf->gridsize_swap_select_transpose.y = conf->stream_ndf_chk * NSAMP_DF / conf->cufft_nx;
  conf->gridsize_swap_select_transpose.z = 1;  
  conf->blocksize_swap_select_transpose.x = conf->cufft_nx;
  conf->blocksize_swap_select_transpose.y = 1;
  conf->blocksize_swap_select_transpose.z = 1;  
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "The configuration of swap_select_transpose kernel is (%d, %d, %d) and (%d, %d, %d)",
	      conf->gridsize_swap_select_transpose.x, conf->gridsize_swap_select_transpose.y, conf->gridsize_swap_select_transpose.z,
	      conf->blocksize_swap_select_transpose.x, conf->blocksize_swap_select_transpose.y, conf->blocksize_swap_select_transpose.z);        

  conf->gridsize_detect_faccumulate_pad_transpose.x = conf->stream_ndf_chk * NSAMP_DF / conf->cufft_nx;
  conf->gridsize_detect_faccumulate_pad_transpose.y = conf->nchan_out;
  conf->gridsize_detect_faccumulate_pad_transpose.z = 1;
  conf->blocksize_detect_faccumulate_pad_transpose.x = conf->nchan_keep_band/(2 * conf->nchan_out);
  conf->blocksize_detect_faccumulate_pad_transpose.y = 1;
  conf->blocksize_detect_faccumulate_pad_transpose.z = 1;
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "The configuration of detect_faccumulate_pad_transpose kernel is (%d, %d, %d) and (%d, %d, %d)",
	      conf->gridsize_detect_faccumulate_pad_transpose.x, conf->gridsize_detect_faccumulate_pad_transpose.y, conf->gridsize_detect_faccumulate_pad_transpose.z,
	      conf->blocksize_detect_faccumulate_pad_transpose.x, conf->blocksize_detect_faccumulate_pad_transpose.y, conf->blocksize_detect_faccumulate_pad_transpose.z);
  
  conf->gridsize_detect_faccumulate_scale.x = conf->stream_ndf_chk * NSAMP_DF / conf->cufft_nx;
  conf->gridsize_detect_faccumulate_scale.y = conf->nchan_out;
  conf->gridsize_detect_faccumulate_scale.z = 1;
  conf->blocksize_detect_faccumulate_scale.x = conf->nchan_keep_band/(2 * conf->nchan_out);
  conf->blocksize_detect_faccumulate_scale.y = 1;
  conf->blocksize_detect_faccumulate_scale.z = 1;
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "The configuration of detect_faccumulate_scale kernel is (%d, %d, %d) and (%d, %d, %d)",
	      conf->gridsize_detect_faccumulate_scale.x, conf->gridsize_detect_faccumulate_scale.y, conf->gridsize_detect_faccumulate_scale.z,
	      conf->blocksize_detect_faccumulate_scale.x, conf->blocksize_detect_faccumulate_scale.y, conf->blocksize_detect_faccumulate_scale.z);
  
  conf->gridsize_accumulate.x = conf->nchan_out;
  conf->gridsize_accumulate.y = 1;
  conf->gridsize_accumulate.z = 1;
  conf->blocksize_accumulate.x = conf->stream_ndf_chk * NSAMP_DF / (conf->cufft_nx * 2); 
  conf->blocksize_accumulate.y = 1;
  conf->blocksize_accumulate.z = 1;
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "The configuration of accumulate kernel is (%d, %d, %d) and (%d, %d, %d)",
	      conf->gridsize_accumulate.x, conf->gridsize_accumulate.y, conf->gridsize_accumulate.z,
	      conf->blocksize_accumulate.x, conf->blocksize_accumulate.y, conf->blocksize_accumulate.z);
  
  conf->gridsize_mean.x = 1; 
  conf->gridsize_mean.y = 1; 
  conf->gridsize_mean.z = 1;
  conf->blocksize_mean.x = conf->nchan_out;
  conf->blocksize_mean.y = 1;
  conf->blocksize_mean.z = 1;
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "The configuration of mean kernel is (%d, %d, %d) and (%d, %d, %d)",
	      conf->gridsize_mean.x, conf->gridsize_mean.y, conf->gridsize_mean.z,
	      conf->blocksize_mean.x, conf->blocksize_mean.y, conf->blocksize_mean.z);
  
  conf->gridsize_scale.x = 1;
  conf->gridsize_scale.y = 1;
  conf->gridsize_scale.z = 1;
  conf->blocksize_scale.x = conf->nchan_out;
  conf->blocksize_scale.y = 1;
  conf->blocksize_scale.z = 1;
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "The configuration of scale kernel is (%d, %d, %d) and (%d, %d, %d)",
	      conf->gridsize_scale.x, conf->gridsize_scale.y, conf->gridsize_scale.z,
	      conf->blocksize_scale.x, conf->blocksize_scale.y, conf->blocksize_scale.z);
  
  /* attach to input ring buffer */
  conf->hdu_in = dada_hdu_create(NULL);
  dada_hdu_set_key(conf->hdu_in, conf->key_in);
  if(dada_hdu_connect(conf->hdu_in) < 0)
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      exit(EXIT_FAILURE);    
    }  
  conf->db_in = (ipcbuf_t *) conf->hdu_in->data_block;
  conf->rbufin_size = ipcbuf_get_bufsz(conf->db_in);
  if(conf->rbufin_size % conf->bufin_size != 0)  
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);    
    }
  
  /* registers the existing host memory range for use by CUDA */
  dada_cuda_dbregister(conf->hdu_in);
  
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_in->header_block);  
  if(conf->hdrsz != DADA_HDRSZ)    // This number should match
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);    
    }
  
  /* make ourselves the read client */
  if(dada_hdu_lock_read(conf->hdu_in) < 0)
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }

  /* Prepare output ring buffer */
  conf->hdu_out = dada_hdu_create(NULL);
  dada_hdu_set_key(conf->hdu_out, conf->key_out);
  if(dada_hdu_connect(conf->hdu_out) < 0)
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);    
    }
  conf->db_out = (ipcbuf_t *) conf->hdu_out->data_block;
  conf->rbufout_size = ipcbuf_get_bufsz(conf->db_out);
   
  if(conf->rbufout_size % conf->bufout_size != 0)  
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);    
    }
  
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_out->header_block);  
  if(conf->hdrsz != DADA_HDRSZ)    // This number should match
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);    
    }
  dada_cuda_dbregister(conf->hdu_out);
  
  /* make ourselves the write client */
  if(dada_hdu_lock_write(conf->hdu_out) < 0)
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }

  if(!conf->sod)
    {
      if(ipcbuf_disable_sod(conf->db_out) < 0)
	{
	  paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);	  
	  dada_hdu_unlock_write(conf->hdu_out);
	  exit(EXIT_FAILURE);
	}
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
  dim3 gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale;
  uint64_t curbufsz;
  int first = 1;
  int nrun_blk = conf.rbufin_size / conf.bufin_size;
  
  gridsize_unpack                      = conf.gridsize_unpack;
  blocksize_unpack                     = conf.blocksize_unpack;
  gridsize_detect_faccumulate_scale            = conf.gridsize_detect_faccumulate_scale ;
  blocksize_detect_faccumulate_scale           = conf.blocksize_detect_faccumulate_scale ;
  gridsize_swap_select_transpose       = conf.gridsize_swap_select_transpose;   
  blocksize_swap_select_transpose      = conf.blocksize_swap_select_transpose;

  fprintf(stdout, "BASEBAND2FILTERBANK_READY\n");  // Ready to take data from ring buffer, just before the header thing
  fflush(stdout);
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "BASEBAND2FILTERBANK_READY");
  
  /* Register header */
  if(register_header(&conf))
    {
      paf_log_add(conf.logfile, "ERR", 1, log_mutex, "header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "register_header done");
  
  while(!ipcbuf_eod(conf.db_in))
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "before getting new buffer block");
      conf.curbuf_in  = ipcbuf_get_next_read(conf.db_in, &curbufsz);
      conf.curbuf_out = ipcbuf_get_next_write(conf.db_out);
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "after getting new buffer block");
      
      /* Get scale of data */
      if(first)
      	{
      	  first = 0;
      	  dat_offs_scl(conf);
	  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "dat_offs_scl done");
	}

      for(i = 0; i < nrun_blk; i ++)
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

	      swap_select_transpose_kernel<<<gridsize_swap_select_transpose, blocksize_swap_select_transpose, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2, conf.cufft_nx, conf.cufft_mod, conf.nchan_keep_chan, conf.nchan_keep_band, conf.nchan_edge);
	      
	      detect_faccumulate_scale_kernel<<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.ddat_offs, conf.ddat_scl);
	      
	      CudaSafeCall(cudaMemcpyAsync(&conf.curbuf_out[hbufout_offset], &conf.dbuf_out[dbufout_offset], conf.sbufout_size, cudaMemcpyDeviceToHost, conf.streams[j]));
	    }
	}
      CudaSynchronizeCall(); // Sync here is for multiple streams

      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "before closing old buffer block");
      ipcbuf_mark_filled(conf.db_out, (uint64_t)(curbufsz * conf.scl_dtsz));
      ipcbuf_mark_cleared(conf.db_in);
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "after closing old buffer block");
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
  dim3 gridsize_accumulate, blocksize_accumulate;
  dim3 gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose;
  
  char fname[MSTR_LEN];
  FILE *fp=NULL;
    
  gridsize_unpack                 = conf.gridsize_unpack;
  blocksize_unpack                = conf.blocksize_unpack;
  	         	
  gridsize_accumulate             = conf.gridsize_accumulate;	       
  blocksize_accumulate            = conf.blocksize_accumulate;
  gridsize_scale                  = conf.gridsize_scale;	       
  blocksize_scale                 = conf.blocksize_scale;	         		           
  gridsize_mean                   = conf.gridsize_mean;	       
  blocksize_mean                  = conf.blocksize_mean;
  gridsize_swap_select_transpose  = conf.gridsize_swap_select_transpose;   
  blocksize_swap_select_transpose = conf.blocksize_swap_select_transpose;
  gridsize_detect_faccumulate_pad_transpose         = conf.gridsize_detect_faccumulate_pad_transpose ;
  blocksize_detect_faccumulate_pad_transpose        = conf.blocksize_detect_faccumulate_pad_transpose ;

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
	  //CudaSafeCall(cudaMemcpyAsync(temp_out, &conf.buf_rt1[bufrt1_offset + (NCHAN_CHK - 1) * NSAMP_DF * conf.stream_ndf_chk], temp_out_len * sizeof(cufftComplex), cudaMemcpyDeviceToHost, conf.streams[j]));
	  
	  /* Do forward FFT */
	  CufftSafeCall(cufftExecC2C(conf.fft_plans[j], &conf.buf_rt1[bufrt1_offset], &conf.buf_rt1[bufrt1_offset], CUFFT_FORWARD));
	  swap_select_transpose_kernel<<<gridsize_swap_select_transpose, blocksize_swap_select_transpose, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2, conf.cufft_nx, conf.cufft_mod, conf.nchan_keep_chan, conf.nchan_keep_band, conf.nchan_edge);
	  
	  detect_faccumulate_pad_transpose_kernel<<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2);
	  
	  accumulate_kernel<<<gridsize_accumulate, blocksize_accumulate, blocksize_accumulate.x * NBYTE_RT, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset]);
	}
      CudaSynchronizeCall(); // Sync here is for multiple streams

      mean_kernel<<<gridsize_mean, blocksize_mean>>>(conf.buf_rt2, conf.bufrt2_offset, conf.ddat_offs, conf.dsquare_mean, conf.nstream, conf.sclndim);
    }
  
  /* Get the scale of each chanel */
  scale_kernel<<<gridsize_scale, blocksize_scale>>>(conf.ddat_offs, conf.dsquare_mean, conf.ddat_scl);
  CudaSynchronizeCall();
  
  CudaSafeCall(cudaMemcpy(conf.hdat_offs, conf.ddat_offs, sizeof(float) * conf.nchan_out, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(conf.hdat_scl, conf.ddat_scl, sizeof(float) * conf.nchan_out, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(conf.hsquare_mean, conf.dsquare_mean, sizeof(float) * conf.nchan_out, cudaMemcpyDeviceToHost));
  
  /* Record scale into file */
  sprintf(fname, "%s/%s_scale.txt", conf.dir, conf.utc_start);
  fp = fopen(fname, "w");
  if(fp == NULL)
    {
      paf_log_add(conf.logfile, "ERR", 1, log_mutex, "Can not open scale file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }
  for (i = 0; i< conf.nchan_out; i++)
    fprintf(fp, "%E\t%E\t%E\n", conf.hdat_offs[i], conf.hdat_scl[i], conf.hsquare_mean[i]);
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
  dada_hdu_disconnect(conf.hdu_out);
  dada_hdu_destroy(conf.hdu_out);
  
  cudaFree(conf.buf_rt1);
  cudaFree(conf.buf_rt2);

  dada_cuda_dbunregister(conf.hdu_in);
  dada_cuda_dbunregister(conf.hdu_out);
  
  dada_hdu_unlock_read(conf.hdu_in);
  dada_hdu_disconnect(conf.hdu_in);
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
  double tsamp;
  
  hdrbuf_in  = ipcbuf_get_next_read(conf->hdu_in->header_block, &hdrsz);  
  if (!hdrbuf_in)
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
  if(hdrsz != DADA_HDRSZ)
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Header size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }

  hdrbuf_out = ipcbuf_get_next_write(conf->hdu_out->header_block);
  if (!hdrbuf_out)
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }  
  if (ascii_header_get(hdrbuf_in, "FILE_SIZE", "%"PRIu64"", &file_size) < 0)  
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error getting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }   
  if (ascii_header_get(hdrbuf_in, "BYTES_PER_SECOND", "%"PRIu64"", &bytes_per_seconds) < 0)  
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }  
  if (ascii_header_get(hdrbuf_in, "TSAMP", "%lf", &tsamp) < 0)  
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error getting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }   
  /* Get utc_start from hdrin */
  if (ascii_header_get(hdrbuf_in, "UTC_START", "%s", conf->utc_start) < 0)  
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error getting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }
  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header
  
  file_size = (uint64_t)(file_size * conf->scl_dtsz);
  bytes_per_seconds = (uint64_t)(bytes_per_seconds * conf->scl_dtsz);
  
  if (ascii_header_set(hdrbuf_out, "NCHAN", "%d", conf->nchan_out) < 0)  
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error setting NCHAN, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "BW", "%lf", -conf->bw) < 0)  // Reverse frequency order
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error setting BW, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "TSAMP", "%lf", tsamp * conf->cufft_nx) < 0)  
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error setting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "NBIT", "%d", NBIT_OUT) < 0)  
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR:\tError setting NBIT, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "NDIM", "%d", NDIM_OUT) < 0)  
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error setting NDIM, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "NPOL", "%d", NPOL_OUT) < 0)  
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error setting NPOL, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "FILE_SIZE", "%"PRIu64"", file_size) < 0)  
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR:\tError setting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "BYTES_PER_SECOND", "%"PRIu64"", bytes_per_seconds) < 0)  
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }
    
  if(ipcbuf_mark_cleared (conf->hdu_in->header_block))  // We are the only one reader, so that we can clear it after read;
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error header_clear, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }
  /* donot set header parameters anymore - acqn. doesn't start */
  if (ipcbuf_mark_filled (conf->hdu_out->header_block, conf->hdrsz) < 0)
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      exit(EXIT_FAILURE);
    }

  return EXIT_SUCCESS;
}