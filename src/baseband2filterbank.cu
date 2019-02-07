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
#include "constants.h"

extern pthread_mutex_t log_mutex;

int initialize_baseband2filterbank(conf_t *conf)
{
  int i;
  int iembed, istride, idist, oembed, ostride, odist, batch, nx;
  int naccumulate_pow2;
  
  /* Prepare parameters */
  conf->nrepeat_per_blk        = conf->ndf_chunk_rbufin / (conf->ndf_per_chunk_stream * conf->nstream);
  conf->nchan_in        = conf->nchunk_in * NCHAN_PER_CHUNK;
  conf->nchan_keep_chan = (int)(conf->cufft_nx / OVER_SAMP_RATE);
  conf->cufft_mod       = (int)(0.5 * conf->cufft_nx / OVER_SAMP_RATE);
  conf->nchan_edge      = (int)(0.5 * conf->nchan_in * conf->nchan_keep_chan - 0.5 * conf->nchan_keep_band);
  conf->inverse_nchan_rate     = conf->nchan_in * conf->nchan_keep_chan/(double)conf->nchan_keep_band;
  conf->scale_dtsz        = NBYTE_OUT / OVER_SAMP_RATE * conf->nchan_out/ (double)(conf->inverse_nchan_rate * conf->nchan_keep_band * NPOL_IN * NDIM_IN * NBYTE_IN);
  conf->bandwidth       = conf->nchan_keep_band/(double)conf->nchan_keep_chan;

  log_add(conf->log_file, "INFO", 1, log_mutex, "We have %d channels input", conf->nchan_in);
  log_add(conf->log_file, "INFO", 1, log_mutex, "The mod to reduce oversampling is %d", conf->cufft_mod);
  log_add(conf->log_file, "INFO", 1, log_mutex, "We will keep %d fine channels for each input channel after FFT", conf->nchan_keep_chan);
  log_add(conf->log_file, "INFO", 1, log_mutex, "We will drop %d fine channels at the band edge for frequency accumulation", conf->nchan_edge);
  log_add(conf->log_file, "INFO", 1, log_mutex, "%f percent fine channels (after down sampling) are kept for frequency accumulation", 1.0/conf->inverse_nchan_rate * 100.);
  log_add(conf->log_file, "INFO", 1, log_mutex, "The data size rate between filterbank and baseband data is %f", conf->scale_dtsz);
  log_add(conf->log_file, "INFO", 1, log_mutex, "The bandwidth for the final output is %f MHz", conf->bandwidth);
  log_add(conf->log_file, "INFO", 1, log_mutex, "%d run to finish one ring buffer block", conf->nrepeat_per_blk);
  
  /* Prepare buffer, stream and fft plan for process */
  conf->nsamp1       = conf->ndf_per_chunk_stream * conf->nchan_in * NSAMP_DF;
  conf->npol1        = conf->nsamp1 * NPOL_IN;
  conf->ndata1       = conf->npol1  * NDIM_IN;
  
  conf->nsamp2       = conf->nsamp1 / OVER_SAMP_RATE / conf->inverse_nchan_rate;
  conf->npol2        = conf->nsamp2 * NPOL_IN;
  conf->ndata2       = conf->npol2  * NDIM_IN;

  conf->nsamp3       = conf->nsamp2 * conf->nchan_out / conf->nchan_keep_band;
  conf->npol3        = conf->nsamp3 * NPOL_OUT;
  conf->ndata3       = conf->npol3  * NDIM_OUT;  

  conf->ndim_scale      = conf->ndf_chunk_rbufin * NSAMP_DF / conf->cufft_nx;   // We do not average in time and here we work on detected data;
  
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

  CudaSafeCall(cudaMalloc((void **)&conf->offset_scale_d, conf->nchan_out * sizeof(cufftComplex)));
  CudaSafeCall(cudaMallocHost((void **)&conf->offset_scale_h, conf->nchan_out * sizeof(cufftComplex)));
  CudaSafeCall(cudaMemset((void *)conf->offset_scale_d, 0, conf->nchan_out * sizeof(cufftComplex)));// We have to clear the memory for this parameter
  
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
  
  conf->gridsize_swap_select_transpose.x = conf->nchan_in;
  conf->gridsize_swap_select_transpose.y = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx;
  conf->gridsize_swap_select_transpose.z = 1;  
  conf->blocksize_swap_select_transpose.x = conf->cufft_nx;
  conf->blocksize_swap_select_transpose.y = 1;
  conf->blocksize_swap_select_transpose.z = 1;  
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of swap_select_transpose kernel is (%d, %d, %d) and (%d, %d, %d)",
	      conf->gridsize_swap_select_transpose.x, conf->gridsize_swap_select_transpose.y, conf->gridsize_swap_select_transpose.z,
	      conf->blocksize_swap_select_transpose.x, conf->blocksize_swap_select_transpose.y, conf->blocksize_swap_select_transpose.z);        

  conf->naccumulate_pad = conf->nchan_keep_band/conf->nchan_out;
  naccumulate_pow2      = (int)pow(2.0, floor(log2((double)conf->naccumulate_pad)));
  conf->gridsize_detect_faccumulate_pad_transpose.x = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx;
  conf->gridsize_detect_faccumulate_pad_transpose.y = conf->nchan_out;
  conf->gridsize_detect_faccumulate_pad_transpose.z = 1;
  conf->blocksize_detect_faccumulate_pad_transpose.x = (naccumulate_pow2<1024)?naccumulate_pow2:1024;
  conf->blocksize_detect_faccumulate_pad_transpose.y = 1;
  conf->blocksize_detect_faccumulate_pad_transpose.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of detect_faccumulate_pad_transpose kernel is (%d, %d, %d) and (%d, %d, %d), naccumulate is %d",
	      conf->gridsize_detect_faccumulate_pad_transpose.x, conf->gridsize_detect_faccumulate_pad_transpose.y, conf->gridsize_detect_faccumulate_pad_transpose.z,
	      conf->blocksize_detect_faccumulate_pad_transpose.x, conf->blocksize_detect_faccumulate_pad_transpose.y, conf->blocksize_detect_faccumulate_pad_transpose.z, conf->naccumulate_pad);

  conf->naccumulate_scale = conf->nchan_keep_band/conf->nchan_out;
  naccumulate_pow2        = (int)pow(2.0, floor(log2((double)conf->naccumulate_scale)));
  conf->gridsize_detect_faccumulate_scale.x = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx;
  conf->gridsize_detect_faccumulate_scale.y = conf->nchan_out;
  conf->gridsize_detect_faccumulate_scale.z = 1;
  conf->blocksize_detect_faccumulate_scale.x = (naccumulate_pow2<1024)?naccumulate_pow2:1024;
  conf->blocksize_detect_faccumulate_scale.y = 1;
  conf->blocksize_detect_faccumulate_scale.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of detect_faccumulate_scale kernel is (%d, %d, %d) and (%d, %d, %d), naccumulate is %d",
	      conf->gridsize_detect_faccumulate_scale.x, conf->gridsize_detect_faccumulate_scale.y, conf->gridsize_detect_faccumulate_scale.z,
	      conf->blocksize_detect_faccumulate_scale.x, conf->blocksize_detect_faccumulate_scale.y, conf->blocksize_detect_faccumulate_scale.z, conf->naccumulate_scale);

  conf->naccumulate = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx; 
  naccumulate_pow2  = (int)pow(2.0, floor(log2((double)conf->naccumulate)));
  conf->gridsize_taccumulate.x = conf->nchan_out;
  conf->gridsize_taccumulate.y = 1;
  conf->gridsize_taccumulate.z = 1;
  conf->blocksize_taccumulate.x = (naccumulate_pow2<1024)?naccumulate_pow2:1024;
  conf->blocksize_taccumulate.y = 1;
  conf->blocksize_taccumulate.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of accumulate kernel is (%d, %d, %d) and (%d, %d, %d), naccumulate is %d",
	      conf->gridsize_taccumulate.x, conf->gridsize_taccumulate.y, conf->gridsize_taccumulate.z,
	      conf->blocksize_taccumulate.x, conf->blocksize_taccumulate.y, conf->blocksize_taccumulate.z, conf->naccumulate);
  
  conf->gridsize_scale.x = 1;
  conf->gridsize_scale.y = 1;
  conf->gridsize_scale.z = 1;
  conf->blocksize_scale.x = conf->nchan_out;
  conf->blocksize_scale.y = 1;
  conf->blocksize_scale.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of scale kernel is (%d, %d, %d) and (%d, %d, %d)",
	      conf->gridsize_scale.x, conf->gridsize_scale.y, conf->gridsize_scale.z,
	      conf->blocksize_scale.x, conf->blocksize_scale.y, conf->blocksize_scale.z);
  
  /* attach to input ring buffer */
  conf->hdu_in = dada_hdu_create(NULL);
  dada_hdu_set_key(conf->hdu_in, conf->key_in);
  if(dada_hdu_connect(conf->hdu_in) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }  
  conf->db_in = (ipcbuf_t *) conf->hdu_in->data_block;
  conf->rbufin_size = ipcbuf_get_bufsz(conf->db_in);
  if((conf->rbufin_size % conf->bufin_size != 0) || (conf->rbufin_size/conf->bufin_size)!= conf->nrepeat_per_blk)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }
  
  /* registers the existing host memory range for use by CUDA */
  dada_cuda_dbregister(conf->hdu_in);
  
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_in->header_block);  
  if(conf->hdrsz != DADA_HDRSZ)    // This number should match
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }
  
  /* make ourselves the read client */
  if(dada_hdu_lock_read(conf->hdu_in) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error locking HDU, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }

  /* Prepare output ring buffer */
  conf->hdu_out = dada_hdu_create(NULL);
  dada_hdu_set_key(conf->hdu_out, conf->key_out);
  if(dada_hdu_connect(conf->hdu_out) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }
  conf->db_out = (ipcbuf_t *) conf->hdu_out->data_block;
  conf->rbufout_size = ipcbuf_get_bufsz(conf->db_out);
   
  if(conf->rbufout_size % conf->bufout_size != 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }
  
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_out->header_block);  
  if(conf->hdrsz != DADA_HDRSZ)    // This number should match
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }
  dada_cuda_dbregister(conf->hdu_out);
  
  /* make ourselves the write client */
  if(dada_hdu_lock_write(conf->hdu_out) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error locking HDU, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }

  if(!conf->sod)
    {
      if(ipcbuf_disable_sod(conf->db_out) < 0)
	{
	  log_add(conf->log_file, "ERR", 1, log_mutex, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.", __FILE__, __LINE__);
	  fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  
	  destroy_baseband2filterbank(*conf);
	  fclose(conf->log_file);
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
  uint64_t cbufsz;
  int first = 1;
  
  gridsize_unpack                      = conf.gridsize_unpack;
  blocksize_unpack                     = conf.blocksize_unpack;
  gridsize_detect_faccumulate_scale    = conf.gridsize_detect_faccumulate_scale ;
  blocksize_detect_faccumulate_scale   = conf.blocksize_detect_faccumulate_scale ;
  gridsize_swap_select_transpose       = conf.gridsize_swap_select_transpose;   
  blocksize_swap_select_transpose      = conf.blocksize_swap_select_transpose;

  fprintf(stdout, "BASEBAND2FILTERBANK_READY\n");  // Ready to take data from ring buffer, just before the header thing
  fflush(stdout);
  log_add(conf.log_file, "INFO", 1, log_mutex, "BASEBAND2FILTERBANK_READY");
  
  /* Register header only with sod */
  if(conf.sod)
    {
      if(register_header(&conf))
	{
	  log_add(conf.log_file, "ERR", 1, log_mutex, "header register failed, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	  fprintf(stderr, "header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

	  destroy_baseband2filterbank(conf);
	  fclose(conf.log_file);
	  exit(EXIT_FAILURE);
	}
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "register_header done");
  
  while(!ipcbuf_eod(conf.db_in))
    {
      log_add(conf.log_file, "INFO", 1, log_mutex, "before getting new buffer block");
      conf.cbuf_in  = ipcbuf_get_next_read(conf.db_in, &cbufsz);
      conf.cbuf_out = ipcbuf_get_next_write(conf.db_out);
      log_add(conf.log_file, "INFO", 1, log_mutex, "after getting new buffer block");
      
      /* Get scale of data */
      if(first)
      	{
      	  first = 0;
      	  offset_scale(conf);
	  log_add(conf.log_file, "INFO", 1, log_mutex, "offset_scale done");  // I may need to put this part before while and make the first output buffer block empty
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
	      
	      /* Copy data into device */
	      CudaSafeCall(cudaMemcpyAsync(&conf.dbuf_in[dbufin_offset], &conf.cbuf_in[hbufin_offset], conf.sbufin_size, cudaMemcpyHostToDevice, conf.streams[j]));
	      
	      /* Unpack raw data into cufftComplex array */
	      unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, conf.streams[j]>>>(&conf.dbuf_in[dbufin_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp1);
	      CudaSafeKernelLaunch();
	      
	      /* Do forward FFT */
	      CufftSafeCall(cufftExecC2C(conf.fft_plans[j], &conf.buf_rt1[bufrt1_offset], &conf.buf_rt1[bufrt1_offset], CUFFT_FORWARD));

	      swap_select_transpose_kernel<<<gridsize_swap_select_transpose, blocksize_swap_select_transpose, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2, conf.cufft_nx, conf.cufft_mod, conf.nchan_keep_chan, conf.nchan_keep_band, conf.nchan_edge);
	      CudaSafeKernelLaunch();
	      	      	  
	      switch (blocksize_detect_faccumulate_scale.x)
		{
		case 1024:
		  detect_faccumulate_scale_kernel2<1024><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.naccumulate_scale, conf.offset_scale_d);
		  break;
		case 512:
		  detect_faccumulate_scale_kernel2< 512><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.naccumulate_scale, conf.offset_scale_d);
		  break;
		case 256:
		  detect_faccumulate_scale_kernel2< 256><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.naccumulate_scale, conf.offset_scale_d);
		  break;
		case 128:
		  detect_faccumulate_scale_kernel2< 128><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.naccumulate_scale, conf.offset_scale_d);
		  break;
		case 64:
		  detect_faccumulate_scale_kernel2<  64><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.naccumulate_scale, conf.offset_scale_d);
		  break;
		case 32:
		  detect_faccumulate_scale_kernel2<  32><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.naccumulate_scale, conf.offset_scale_d);
		  break;
		case 16:
		  detect_faccumulate_scale_kernel2<  16><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.naccumulate_scale, conf.offset_scale_d);
		  break;
		case 8:
		  detect_faccumulate_scale_kernel2<   8><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.naccumulate_scale, conf.offset_scale_d);
		  break;
		case 4:
		  detect_faccumulate_scale_kernel2<   4><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.naccumulate_scale, conf.offset_scale_d);
		  break;
		case 2:
		  detect_faccumulate_scale_kernel2<   2><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.naccumulate_scale, conf.offset_scale_d);
		  break;
		case 1:
		  detect_faccumulate_scale_kernel2<   1><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out[dbufout_offset], conf.nsamp2, conf.naccumulate_scale, conf.offset_scale_d);
		  break;
		}
	      CudaSafeKernelLaunch();	      
	      CudaSafeCall(cudaMemcpyAsync(&conf.cbuf_out[hbufout_offset], &conf.dbuf_out[dbufout_offset], conf.sbufout_size, cudaMemcpyDeviceToHost, conf.streams[j]));
	    }
	}
      CudaSynchronizeCall(); // Sync here is for multiple streams

      log_add(conf.log_file, "INFO", 1, log_mutex, "before closing old buffer block");
      ipcbuf_mark_filled(conf.db_out, (uint64_t)(cbufsz * conf.scale_dtsz));
      ipcbuf_mark_cleared(conf.db_in);
      log_add(conf.log_file, "INFO", 1, log_mutex, "after closing old buffer block");
    }

  log_add(conf.log_file, "INFO", 1, log_mutex, "FINISH the process");

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
  dim3 gridsize_taccumulate, blocksize_taccumulate;
  dim3 gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose;
  
  char fname[MSTR_LEN];
  FILE *fp=NULL;
    
  gridsize_unpack                 = conf.gridsize_unpack;
  blocksize_unpack                = conf.blocksize_unpack;
  	         	
  gridsize_taccumulate             = conf.gridsize_taccumulate;	       
  blocksize_taccumulate            = conf.blocksize_taccumulate;
  gridsize_scale                  = conf.gridsize_scale;	       
  blocksize_scale                 = conf.blocksize_scale;	         		           
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
	  CudaSafeCall(cudaMemcpyAsync(&conf.dbuf_in[dbufin_offset], &conf.cbuf_in[hbufin_offset], conf.sbufin_size, cudaMemcpyHostToDevice, conf.streams[j]));

	  /* Unpack raw data into cufftComplex array */
	  unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, conf.streams[j]>>>(&conf.dbuf_in[dbufin_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp1);
	  CudaSafeKernelLaunch();
	  
	  /* Do forward FFT */
	  CufftSafeCall(cufftExecC2C(conf.fft_plans[j], &conf.buf_rt1[bufrt1_offset], &conf.buf_rt1[bufrt1_offset], CUFFT_FORWARD));
	  swap_select_transpose_kernel<<<gridsize_swap_select_transpose, blocksize_swap_select_transpose, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2, conf.cufft_nx, conf.cufft_mod, conf.nchan_keep_chan, conf.nchan_keep_band, conf.nchan_edge);
	  CudaSafeKernelLaunch();
	  
	  switch (blocksize_detect_faccumulate_pad_transpose.x )
	    {
	    case 1024:
	      detect_faccumulate_pad_transpose_kernel1<1024><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;
	    case 512:
	      detect_faccumulate_pad_transpose_kernel1< 512><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 256:
	      detect_faccumulate_pad_transpose_kernel1< 256><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 128:
	      detect_faccumulate_pad_transpose_kernel1< 128><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 64:
	      detect_faccumulate_pad_transpose_kernel1<  64><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 32:
	      detect_faccumulate_pad_transpose_kernel1<  32><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 16:
	      detect_faccumulate_pad_transpose_kernel1<  16><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 8:
	      detect_faccumulate_pad_transpose_kernel1<   8><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 4:
	      detect_faccumulate_pad_transpose_kernel1<   4><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 2:
	      detect_faccumulate_pad_transpose_kernel1<   2><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 1:
	      detect_faccumulate_pad_transpose_kernel1<   1><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;
	    }
	  CudaSafeKernelLaunch();
	}
      CudaSynchronizeCall(); // Sync here is for multiple streams
      
      switch (blocksize_taccumulate.x)
        {
        case 1024:
          reduce9_kernel<1024><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_RT>>>(conf.buf_rt1, conf.offset_scale_d, conf.bufrt1_offset, conf.naccumulate, conf.nstream, conf.ndim_scale);
          break;
        case 512:
          reduce9_kernel< 512><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_RT>>>(conf.buf_rt1, conf.offset_scale_d, conf.bufrt1_offset, conf.naccumulate, conf.nstream, conf.ndim_scale);
          break;
        case 256:
          reduce9_kernel< 256><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_RT>>>(conf.buf_rt1, conf.offset_scale_d, conf.bufrt1_offset, conf.naccumulate, conf.nstream, conf.ndim_scale);
          break;
        case 128:
          reduce9_kernel< 128><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_RT>>>(conf.buf_rt1, conf.offset_scale_d, conf.bufrt1_offset, conf.naccumulate, conf.nstream, conf.ndim_scale);
          break;
        case 64:
          reduce9_kernel<  64><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_RT>>>(conf.buf_rt1, conf.offset_scale_d, conf.bufrt1_offset, conf.naccumulate, conf.nstream, conf.ndim_scale);
          break;
        case 32:
          reduce9_kernel<  32><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_RT>>>(conf.buf_rt1, conf.offset_scale_d, conf.bufrt1_offset, conf.naccumulate, conf.nstream, conf.ndim_scale);
          break;
        case 16:
          reduce9_kernel<  16><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_RT>>>(conf.buf_rt1, conf.offset_scale_d, conf.bufrt1_offset, conf.naccumulate, conf.nstream, conf.ndim_scale);
          break;
        case 8:
          reduce9_kernel<   8><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_RT>>>(conf.buf_rt1, conf.offset_scale_d, conf.bufrt1_offset, conf.naccumulate, conf.nstream, conf.ndim_scale);
          break;
        case 4:
          reduce9_kernel<   4><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_RT>>>(conf.buf_rt1, conf.offset_scale_d, conf.bufrt1_offset, conf.naccumulate, conf.nstream, conf.ndim_scale);
          break;
        case 2:
          reduce9_kernel<   2><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_RT>>>(conf.buf_rt1, conf.offset_scale_d, conf.bufrt1_offset, conf.naccumulate, conf.nstream, conf.ndim_scale);
          break;
        case 1:
          reduce9_kernel<   1><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_RT>>>(conf.buf_rt1, conf.offset_scale_d, conf.bufrt1_offset, conf.naccumulate, conf.nstream, conf.ndim_scale);
          break;
        }
      CudaSafeKernelLaunch();
    }
  
  /* Get the scale of each chanel */
  scale2_kernel<<<gridsize_scale, blocksize_scale>>>(conf.offset_scale_d, SCL_NSIG, SCL_UINT8);
  CudaSafeKernelLaunch();
  CudaSynchronizeCall();
  
  CudaSafeCall(cudaMemcpy(conf.offset_scale_h, conf.offset_scale_d, sizeof(cufftComplex) * conf.nchan_out, cudaMemcpyDeviceToHost));
  
  /* Record scale into file */
  sprintf(fname, "%s/%s_scale.txt", conf.dir, conf.utc_start);
  fp = fopen(fname, "w");
  if(fp == NULL)
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Can not open scale file, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Can not open scale file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(conf);
      fclose(conf.log_file);
      exit(EXIT_FAILURE);
    }
  for (i = 0; i < conf.nchan_out; i++)
    fprintf(fp, "%E\t%E\n", conf.offset_scale_h[i].x, conf.offset_scale_h[i].y);
  fclose(fp);

  return EXIT_SUCCESS;
}

int destroy_baseband2filterbank(conf_t conf)
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
  if(conf.offset_scale_h)
    cudaFreeHost(conf.offset_scale_h);
  if(conf.offset_scale_d)
    cudaFree(conf.offset_scale_d);
  log_add(conf.log_file, "INFO", 1, log_mutex, "Free cuda memory done");

  if(conf.db_in)
    {
      dada_cuda_dbunregister(conf.hdu_in);
      dada_hdu_unlock_read(conf.hdu_in);
      dada_hdu_destroy(conf.hdu_in);
    }
  if(conf.db_out)
    {
      dada_cuda_dbunregister(conf.hdu_out);
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
  double tsamp;
  
  hdrbuf_in  = ipcbuf_get_next_read(conf->hdu_in->header_block, &hdrsz);  
  if (!hdrbuf_in)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting header_buf, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if(hdrsz != DADA_HDRSZ)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Header size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Header size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }

  hdrbuf_out = ipcbuf_get_next_write(conf->hdu_out->header_block);
  if (!hdrbuf_out)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting header_buf, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }  
  if (ascii_header_get(hdrbuf_in, "FILE_SIZE", "%"PRIu64"", &file_size) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting FILE_SIZE, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error getting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }   
  if (ascii_header_get(hdrbuf_in, "BYTES_PER_SECOND", "%"PRIu64"", &bytes_per_seconds) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }  
  if (ascii_header_get(hdrbuf_in, "TSAMP", "%lf", &tsamp) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting TSAMP, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error getting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }   
  /* Get utc_start from hdrin */
  if (ascii_header_get(hdrbuf_in, "UTC_START", "%s", conf->utc_start) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting UTC_START, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error getting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header
  
  file_size = (uint64_t)(file_size * conf->scale_dtsz);
  bytes_per_seconds = (uint64_t)(bytes_per_seconds * conf->scale_dtsz);
  
  if (ascii_header_set(hdrbuf_out, "NCHAN", "%d", conf->nchan_out) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting NCHAN, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error setting NCHAN, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "BW", "%lf", -conf->bandwidth) < 0)  // Reverse frequency order
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting BW, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error setting BW, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "TSAMP", "%lf", tsamp * conf->cufft_nx) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting TSAMP, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error setting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "NBIT", "%d", NBIT_OUT) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error setting NBIT, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "NDIM", "%d", NDIM_OUT) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting NDIM, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error setting NDIM, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "NPOL", "%d", NPOL_OUT) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting NPOL, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error setting NPOL, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "FILE_SIZE", "%"PRIu64"", file_size) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR:\tError setting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if (ascii_header_set(hdrbuf_out, "BYTES_PER_SECOND", "%"PRIu64"", bytes_per_seconds) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
    
  if(ipcbuf_mark_cleared (conf->hdu_in->header_block))  // We are the only one reader, so that we can clear it after read;
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error header_clear, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error header_clear, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  /* donot set header parameters anymore - acqn. doesn't start */
  if (ipcbuf_mark_filled (conf->hdu_out->header_block, conf->hdrsz) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error header_fill, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }

  return EXIT_SUCCESS;
}