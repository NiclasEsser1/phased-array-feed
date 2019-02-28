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

#include "baseband2filterbank.cuh"
#include "cudautil.cuh"
#include "kernel.cuh"
#include "log.h"
#include "constants.h"

// Have to discuss the sensor thing!!!
// Tsamp of the monitor for search
// Type of pol;

// We can time accumulate in each stream to get monitor data for search mode
// Tsamp for this case is 0.131072 seconds for 1024 DF per stream configuration
// If we only want to monitor Stokes I, we can tranpose the TF data to FT and time accumulate it
// If we want to monitor Stokes I, AABB or IQUV, we need to calculate the spectrum after the PTF
// Frequency accumulate it, tranpose from PTF to PFT and accumulate in time
// The spectrum calculation and frequency accumulation can happen wiht the detect_faccumulate_scale
// Filterbank data and spectrum data will out with separate buffers
// Send the data once the buffer block is done

// Make the scale calculation unblock, done;
extern pthread_mutex_t log_mutex;

int default_arguments(conf_t *conf)
{
  memset(conf->dir, 0x00, sizeof(conf->dir));
  sprintf(conf->dir, "unset"); // Default with "unset"
  
  conf->ndf_per_chunk_rbufin = 0; // Default with an impossible value
  conf->nstream          = -1; // Default with an impossible value
  conf->ndf_per_chunk_stream = 0; // Default with an impossible value
  conf->sod = 0;                   // Default no SOD at the beginning
  conf->nchunk_in = -1;
  conf->cufft_nx = -1;
  conf->nchan_out = -1;
  conf->nchan_keep_band = -1;
  
  return EXIT_SUCCESS;
}

int initialize_baseband2filterbank(conf_t *conf)
{
  int i;
  int iembed, istride, idist, oembed, ostride, odist, batch, nx;
  int naccumulate_pow2;
  
  /* Prepare parameters */
  conf->nrepeat_per_blk = conf->ndf_per_chunk_rbufin / (conf->ndf_per_chunk_stream * conf->nstream);
  conf->nchan_in        = conf->nchunk_in * NCHAN_PER_CHUNK;
  conf->nchan_keep_chan = (int)(conf->cufft_nx / OVER_SAMP_RATE);
  conf->cufft_mod       = (int)(0.5 * conf->cufft_nx / OVER_SAMP_RATE);
  conf->nchan_edge      = (int)(0.5 * conf->nchan_in * conf->nchan_keep_chan - 0.5 * conf->nchan_keep_band);
  conf->inverse_nchan_rate = conf->nchan_in * conf->nchan_keep_chan/(double)conf->nchan_keep_band;
  conf->scale_dtsz         = NBYTE_FILTERBANK / OVER_SAMP_RATE * conf->nchan_out/ (double)(conf->inverse_nchan_rate * conf->nchan_keep_band * NPOL_BASEBAND * NDIM_BASEBAND * NBYTE_BASEBAND);
  conf->bandwidth       = conf->nchan_keep_band/(double)conf->nchan_keep_chan;

  log_add(conf->log_file, "INFO", 1, log_mutex, "We have %d channels input", conf->nchan_in);
  log_add(conf->log_file, "INFO", 1, log_mutex, "The mod to reduce oversampling is %d", conf->cufft_mod);
  log_add(conf->log_file, "INFO", 1, log_mutex, "We will keep %d fine channels for each input channel after FFT", conf->nchan_keep_chan);
  if(conf->nchan_edge<0) // Check the nchan_keep_band further
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: nchan_edge can not be a negative number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf->nchan_edge, __FILE__, __LINE__);
      log_add(conf->log_file, "ERR", 1, log_mutex, "nchan_edge can not be a negative number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf->nchan_edge, __FILE__, __LINE__);
      
      log_close(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "We keep %d fine channels for the whole band after FFT", conf->nchan_keep_band); 
  
  log_add(conf->log_file, "INFO", 1, log_mutex, "We will drop %d fine channels at the band edge for frequency accumulation", conf->nchan_edge);
  log_add(conf->log_file, "INFO", 1, log_mutex, "%f percent fine channels (after down sampling) are kept for frequency accumulation", 1.0/conf->inverse_nchan_rate * 100.);
  log_add(conf->log_file, "INFO", 1, log_mutex, "The data size rate between filterbank and baseband data is %f", conf->scale_dtsz);
  log_add(conf->log_file, "INFO", 1, log_mutex, "The bandwidth for the final output is %f MHz", conf->bandwidth);
  log_add(conf->log_file, "INFO", 1, log_mutex, "%d run to finish one ring buffer block", conf->nrepeat_per_blk);
  
  /* Prepare buffer, stream and fft plan for process */
  conf->nsamp1       = conf->ndf_per_chunk_stream * conf->nchan_in * NSAMP_DF;
  conf->npol1        = conf->nsamp1 * NPOL_BASEBAND;
  conf->ndata1       = conf->npol1  * NDIM_BASEBAND;
  
  conf->nsamp2       = conf->nsamp1 / OVER_SAMP_RATE / conf->inverse_nchan_rate;
  conf->npol2        = conf->nsamp2 * NPOL_BASEBAND;
  conf->ndata2       = conf->npol2  * NDIM_BASEBAND;

  conf->nsamp3       = conf->nsamp2 * conf->nchan_out / conf->nchan_keep_band;
  conf->npol3        = conf->nsamp3 * NPOL_FILTERBANK;
  conf->ndata3       = conf->npol3  * NDIM_FILTERBANK;
  conf->nseg_per_blk = conf->nstream * conf->nrepeat_per_blk;
  conf->neth_per_blk = conf->nseg_per_blk * NDATA_PER_SAMP_FULL;
  conf->ndim_scale   = conf->ndf_per_chunk_rbufin * NSAMP_DF / conf->cufft_nx;   // We do not average in time and here we work on detected data;
  
  conf->fits         = (fits_t *)malloc(conf->neth_per_blk * sizeof(fits_t));
  for(i = 0; i < conf->neth_per_blk; i++)
    cudaHostRegister ((void *) conf->fits[i].data, UDP_PAYLOAD_SIZE_MAX, 0);
  
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
  
  conf->sbufin_size   = conf->ndata1 * NBYTE_BASEBAND;
  conf->sbufout1_size = conf->ndata3 * NBYTE_FILTERBANK;
  conf->sbufout2_size = conf->nsamp3 * NBYTE_FLOAT * NDATA_PER_SAMP_RT;
  
  conf->bufin_size    = conf->nstream * conf->sbufin_size;
  conf->bufout1_size  = conf->nstream * conf->sbufout1_size;
  conf->bufout2_size  = conf->nstream * conf->sbufout2_size;
  
  conf->sbufrt1_size = conf->npol1 * NBYTE_CUFFT_COMPLEX;
  conf->sbufrt2_size = conf->npol2 * NBYTE_CUFFT_COMPLEX;
  conf->bufrt1_size  = conf->nstream * conf->sbufrt1_size;
  conf->bufrt2_size  = conf->nstream * conf->sbufrt2_size;
    
  conf->hbufin_offset = conf->sbufin_size;
  conf->dbufin_offset = conf->sbufin_size / (NBYTE_BASEBAND * NPOL_BASEBAND * NDIM_BASEBAND);
  conf->bufrt1_offset = conf->sbufrt1_size / NBYTE_CUFFT_COMPLEX;
  conf->bufrt2_offset = conf->sbufrt2_size / NBYTE_CUFFT_COMPLEX;
  
  conf->dbufout1_offset = conf->sbufout1_size / NBYTE_FILTERBANK;
  conf->hbufout1_offset = conf->sbufout1_size;
  conf->dbufout2_offset = conf->sbufout2_size / NBYTE_FLOAT;
  conf->hbufout2_offset = conf->sbufout2_size;

  conf->dbufout4_offset = conf->nchan_out * NDATA_PER_SAMP_RT;
  
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_in, conf->bufin_size));  
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out1, conf->bufout1_size));
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out2, conf->bufout2_size));
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out3, conf->bufout2_size));
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out4, conf->nchan_out * NDATA_PER_SAMP_RT * conf->nstream * NBYTE_FLOAT));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt1, conf->bufrt1_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt2, conf->bufrt2_size));

  CudaSafeCall(cudaMalloc((void **)&conf->offset_scale_d, conf->nstream * conf->nchan_out * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&conf->offset_scale_h, conf->nchan_out * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMemset((void *)conf->offset_scale_d, 0, conf->nstream * conf->nchan_out * NBYTE_CUFFT_COMPLEX));// We have to clear the memory for this parameter
  
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

  conf->n_transpose = conf->nchan_out;
  conf->m_transpose = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx;
  
  conf->gridsize_transpose.x = ceil(conf->n_transpose / (double)TILE_DIM);
  conf->gridsize_transpose.y = ceil(conf->m_transpose / (double)TILE_DIM);
  conf->gridsize_transpose.z = 1;
  conf->blocksize_transpose.x = TILE_DIM;
  conf->blocksize_transpose.y = NROWBLOCK_TRANS;
  conf->blocksize_transpose.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of transpose kernel is (%d, %d, %d) and (%d, %d, %d)",
	      conf->gridsize_transpose.x, conf->gridsize_transpose.y, conf->gridsize_transpose.z,
	      conf->blocksize_transpose.x, conf->blocksize_transpose.y, conf->blocksize_transpose.z);
  
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
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);    
    }  
  conf->db_in = (ipcbuf_t *) conf->hdu_in->data_block;
  conf->rbufin_size = ipcbuf_get_bufsz(conf->db_in);
  if((conf->rbufin_size % conf->bufin_size != 0) || (conf->rbufin_size/conf->bufin_size)!= conf->nrepeat_per_blk)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);    
    }

  struct timespec start, stop;
  double elapsed_time;
  clock_gettime(CLOCK_REALTIME, &start);
  /* registers the existing host memory range for use by CUDA */
  dada_cuda_dbregister(conf->hdu_in);  // To put this into capture does not improve the memcpy!!!
  
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "elapsed_time for filterbank for dbregister is %f\n", elapsed_time);
  fflush(stdout);
  
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_in->header_block);  
  if(conf->hdrsz != DADA_HDRSZ)    // This number should match
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);    
    }
  
  /* make ourselves the read client */
  if(dada_hdu_lock_read(conf->hdu_in) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error locking HDU, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }

  /* Prepare output ring buffer */
  conf->hdu_out = dada_hdu_create(NULL);
  dada_hdu_set_key(conf->hdu_out, conf->key_out);
  if(dada_hdu_connect(conf->hdu_out) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);    
    }
  conf->db_out = (ipcbuf_t *) conf->hdu_out->data_block;
  conf->rbufout1_size = ipcbuf_get_bufsz(conf->db_out);
  dada_cuda_dbregister(conf->hdu_out);  // To put this into capture does not improve the memcpy!!!
  
  if(conf->rbufout1_size % conf->bufout1_size != 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);    
    }
  
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_out->header_block);  
  if(conf->hdrsz != DADA_HDRSZ)    // This number should match
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);    
    }
  
  /* make ourselves the write client */
  if(dada_hdu_lock_write(conf->hdu_out) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error locking HDU, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }

  if(!(conf->sod == 1))
    {
      if(ipcbuf_disable_sod(conf->db_out) < 0)
	{
	  log_add(conf->log_file, "ERR", 1, log_mutex, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.", __FILE__, __LINE__);
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  
	  destroy_baseband2filterbank(*conf);
	  fclose(conf->log_file);
	  CudaSafeCall(cudaProfilerStop());
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
  uint64_t i, j, k;
  uint64_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset, hbufout1_offset, dbufout1_offset, dbufout2_offset, dbufout4_offset;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_transpose, blocksize_transpose;
  dim3 gridsize_swap_select_transpose, blocksize_swap_select_transpose;
  dim3 gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale;
  dim3 gridsize_taccumulate, blocksize_taccumulate;
  uint64_t cbufsz;
  int first = 1;
  double chan_width; 
  double time_res_blk, time_offset = 0;
  double time_res_stream;
  int eth_index;
  struct tm tm_stamp;
  char time_stamp[MSTR_LEN];
  double time_stamp_f;
  time_t time_stamp_i;
  
  gridsize_unpack                      = conf.gridsize_unpack;
  blocksize_unpack                     = conf.blocksize_unpack;
  gridsize_taccumulate                 = conf.gridsize_taccumulate;
  blocksize_taccumulate                = conf.blocksize_taccumulate;
  gridsize_transpose                   = conf.gridsize_transpose;
  blocksize_transpose                  = conf.blocksize_transpose;
  gridsize_detect_faccumulate_scale    = conf.gridsize_detect_faccumulate_scale ;
  blocksize_detect_faccumulate_scale   = conf.blocksize_detect_faccumulate_scale ;
  gridsize_swap_select_transpose       = conf.gridsize_swap_select_transpose;   
  blocksize_swap_select_transpose      = conf.blocksize_swap_select_transpose;

  fprintf(stdout, "BASEBAND2FILTERBANK_READY\n");  // Ready to take data from ring buffer, just before the header thing
  fflush(stdout);
  log_add(conf.log_file, "INFO", 1, log_mutex, "BASEBAND2FILTERBANK_READY");
  
  if(read_register_header(&conf))
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "header register failed, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(conf);
      fclose(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }

  time_res_blk = conf.tsamp_in * conf.ndf_per_chunk_rbufin * NSAMP_DF / 1.0E6; // This has to be after read_register_header, in seconds
  time_res_stream = conf.tsamp_in * conf.ndf_per_chunk_stream * NSAMP_DF / 1.0E6; // This has to be after read_register_header, in seconds
  strptime(conf.utc_start, DADA_TIMESTR, &tm_stamp);
  time_stamp_f = mktime(&tm_stamp) + conf.picoseconds / 1.0E12 + 0.5 * time_res_stream;
  chan_width = conf.bandwidth/conf.nchan_out;
  log_add(conf.log_file, "INFO", 1, log_mutex, "read_register_header done");
  
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
	      dbufout1_offset = j * conf.dbufout1_offset;
	      dbufout2_offset = j * conf.dbufout2_offset;
	      dbufout4_offset = j * conf.dbufout4_offset;
	      hbufout1_offset = j * conf.hbufout1_offset + i * conf.bufout1_size;
	      
	      /* Copy data into device */
	      CudaSafeCall(cudaMemcpyAsync(&conf.dbuf_in[dbufin_offset], &conf.cbuf_in[hbufin_offset], conf.sbufin_size, cudaMemcpyHostToDevice, conf.streams[j]));
	      
	      /* Unpack raw data into cufftComplex array */
	      unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, conf.streams[j]>>>(&conf.dbuf_in[dbufin_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp1);
	      CudaSafeKernelLaunch();
	      
	      /* Do forward FFT */
	      CufftSafeCall(cufftExecC2C(conf.fft_plans[j], &conf.buf_rt1[bufrt1_offset], &conf.buf_rt1[bufrt1_offset], CUFFT_FORWARD));

	      swap_select_transpose_ptf_kernel<<<gridsize_swap_select_transpose, blocksize_swap_select_transpose, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2, conf.cufft_nx, conf.cufft_mod, conf.nchan_keep_chan, conf.nchan_keep_band, conf.nchan_edge);
	      CudaSafeKernelLaunch();
	      
	      switch (blocksize_detect_faccumulate_scale.x)
		{
		case 1024:
		  detect_faccumulate_scale2_spectral_faccumulate_kernel
		    <1024>
		    <<<gridsize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.buf_rt2[bufrt2_offset],
		     &conf.dbuf_out1[dbufout1_offset],
		     &conf.dbuf_out2[dbufout2_offset],
		     conf.nsamp2,
		     conf.nsamp3,
		     conf.naccumulate_scale,
		     conf.offset_scale_d);
		  break;
		  
		case 512:
		  detect_faccumulate_scale2_spectral_faccumulate_kernel
		    < 512>
		    <<<gridsize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.buf_rt2[bufrt2_offset],
		     &conf.dbuf_out1[dbufout1_offset],
		     &conf.dbuf_out2[dbufout2_offset],
		     conf.nsamp2,
		     conf.nsamp3,
		     conf.naccumulate_scale,
		     conf.offset_scale_d);
		  break;
		  
		case 256:
		  detect_faccumulate_scale2_spectral_faccumulate_kernel
		    < 256>
		    <<<gridsize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.buf_rt2[bufrt2_offset],
		     &conf.dbuf_out1[dbufout1_offset],
		     &conf.dbuf_out2[dbufout2_offset],
		     conf.nsamp2,
		     conf.nsamp3,
		     conf.naccumulate_scale,
		     conf.offset_scale_d);
		  break;
		  
		case 128:
		  detect_faccumulate_scale2_spectral_faccumulate_kernel
		    < 128>
		    <<<gridsize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.buf_rt2[bufrt2_offset],
		     &conf.dbuf_out1[dbufout1_offset],
		     &conf.dbuf_out2[dbufout2_offset],
		     conf.nsamp2,
		     conf.nsamp3,
		     conf.naccumulate_scale,
		     conf.offset_scale_d);
		  break;																															                                  
		case 64:
		  detect_faccumulate_scale2_spectral_faccumulate_kernel
		    <  64>
		    <<<gridsize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.buf_rt2[bufrt2_offset],
		     &conf.dbuf_out1[dbufout1_offset],
		     &conf.dbuf_out2[dbufout2_offset],
		     conf.nsamp2,
		     conf.nsamp3,
		     conf.naccumulate_scale,
		     conf.offset_scale_d);
		  break;																															                                  
		case 32:
		  detect_faccumulate_scale2_spectral_faccumulate_kernel
		    <  32>
		    <<<gridsize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.buf_rt2[bufrt2_offset],
		     &conf.dbuf_out1[dbufout1_offset],
		     &conf.dbuf_out2[dbufout2_offset],
		     conf.nsamp2,
		     conf.nsamp3,
		     conf.naccumulate_scale,
		     conf.offset_scale_d);
		  break;
		  
		case 16:
		  detect_faccumulate_scale2_spectral_faccumulate_kernel
		    <  16>
		    <<<gridsize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.buf_rt2[bufrt2_offset],
		     &conf.dbuf_out1[dbufout1_offset],
		     &conf.dbuf_out2[dbufout2_offset],
		     conf.nsamp2,
		     conf.nsamp3,
		     conf.naccumulate_scale,
		     conf.offset_scale_d);
		  break;
		  
		case 8:
		  detect_faccumulate_scale2_spectral_faccumulate_kernel
		    <   8>
		    <<<gridsize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.buf_rt2[bufrt2_offset],
		     &conf.dbuf_out1[dbufout1_offset],
		     &conf.dbuf_out2[dbufout2_offset],
		     conf.nsamp2,
		     conf.nsamp3,
		     conf.naccumulate_scale,
		     conf.offset_scale_d);
		  break;

		case 4:
		  detect_faccumulate_scale2_spectral_faccumulate_kernel
		    <   4>
		    <<<gridsize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.buf_rt2[bufrt2_offset],
		     &conf.dbuf_out1[dbufout1_offset],
		     &conf.dbuf_out2[dbufout2_offset],
		     conf.nsamp2,
		     conf.nsamp3,
		     conf.naccumulate_scale,
		     conf.offset_scale_d);
		  break;

		case 2:
		  detect_faccumulate_scale2_spectral_faccumulate_kernel
		    <   2>
		    <<<gridsize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.buf_rt2[bufrt2_offset],
		     &conf.dbuf_out1[dbufout1_offset],
		     &conf.dbuf_out2[dbufout2_offset],
		     conf.nsamp2,
		     conf.nsamp3,
		     conf.naccumulate_scale,
		     conf.offset_scale_d);
		  break;
		  
		case 1:
		  detect_faccumulate_scale2_spectral_faccumulate_kernel
		    <   1>
		    <<<gridsize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale,
		    blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.buf_rt2[bufrt2_offset],
		     &conf.dbuf_out1[dbufout1_offset],
		     &conf.dbuf_out2[dbufout2_offset],
		     conf.nsamp2,
		     conf.nsamp3,
		     conf.naccumulate_scale,
		     conf.offset_scale_d);
		  break;
		}
	      CudaSafeKernelLaunch();	      
	      CudaSafeCall(cudaMemcpyAsync(&conf.cbuf_out[hbufout1_offset], &conf.dbuf_out1[dbufout1_offset], conf.sbufout1_size, cudaMemcpyDeviceToHost, conf.streams[j]));

	      /* Further process for another mode */
	      transpose_kernel<<<gridsize_transpose, blocksize_transpose, 0, conf.streams[j]>>>(&conf.dbuf_out2[dbufout2_offset], &conf.dbuf_out3[dbufout2_offset], conf.nsamp3, conf.n_transpose, conf.m_transpose);
	      CudaSafeKernelLaunch();
	      
	      switch (blocksize_taccumulate.x)
		{
		case 1024:
		  taccumulate_float_kernel
		    <1024>
		    <<<gridsize_taccumulate,
		    blocksize_taccumulate,
		    blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.dbuf_out3[dbufout2_offset],
		     &conf.dbuf_out4[dbufout4_offset],
		     conf.nsamp3,
		     conf.naccumulate);
		  break;
      
		case 512:
		  taccumulate_float_kernel
		    < 512>
		    <<<gridsize_taccumulate,
		    blocksize_taccumulate,
		    blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.dbuf_out3[dbufout2_offset],
		     &conf.dbuf_out4[dbufout4_offset],
		     conf.nsamp3,
		     conf.naccumulate);
		  break;
		  
		case 256:
		  taccumulate_float_kernel
		    < 256>
		    <<<gridsize_taccumulate,
		    blocksize_taccumulate,
		    blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.dbuf_out3[dbufout2_offset],
		     &conf.dbuf_out4[dbufout4_offset],
		     conf.nsamp3,
		     conf.naccumulate);
		  break;
		  
		case 128:
		  taccumulate_float_kernel
		    < 128>
		    <<<gridsize_taccumulate,
		    blocksize_taccumulate,
		    blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.dbuf_out3[dbufout2_offset],
		     &conf.dbuf_out4[dbufout4_offset],
		     conf.nsamp3,
		     conf.naccumulate);
		  break;
		  
		case 64:
		  taccumulate_float_kernel
		    <  64>
		    <<<gridsize_taccumulate,
		    blocksize_taccumulate,
		    blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.dbuf_out3[dbufout2_offset],
		     &conf.dbuf_out4[dbufout4_offset],
		     conf.nsamp3,
		     conf.naccumulate);
		  break;
		  
		case 32:
		  taccumulate_float_kernel
		    <  32>
		    <<<gridsize_taccumulate,
		    blocksize_taccumulate,
		    blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.dbuf_out3[dbufout2_offset],
		     &conf.dbuf_out4[dbufout4_offset],
		     conf.nsamp3,
		     conf.naccumulate);
		  break;
		  
		case 16:
		  taccumulate_float_kernel
		    <  16>
		    <<<gridsize_taccumulate,
		    blocksize_taccumulate,
		    blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.dbuf_out3[dbufout2_offset],
		     &conf.dbuf_out4[dbufout4_offset],
		     conf.nsamp3,
		     conf.naccumulate);
		  break;
		  
		case 8:
		  taccumulate_float_kernel
		    <   8>
		    <<<gridsize_taccumulate,
		    blocksize_taccumulate,
		    blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.dbuf_out3[dbufout2_offset],
		     &conf.dbuf_out4[dbufout4_offset],
		     conf.nsamp3,
		     conf.naccumulate);
		  break;
		  
		case 4:
		  taccumulate_float_kernel
		    <   4>
		    <<<gridsize_taccumulate,
		    blocksize_taccumulate,
		    blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.dbuf_out3[dbufout2_offset],
		     &conf.dbuf_out4[dbufout4_offset],
		     conf.nsamp3,
		     conf.naccumulate);
		  break;
		  
		case 2:
		  taccumulate_float_kernel
		    <   2>
		    <<<gridsize_taccumulate,
		    blocksize_taccumulate,
		    blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.dbuf_out3[dbufout2_offset],
		     &conf.dbuf_out4[dbufout4_offset],
		     conf.nsamp3,
		     conf.naccumulate);
		  break;
		  
		case 1:
		  taccumulate_float_kernel
		    <   1>
		    <<<gridsize_taccumulate,
		    blocksize_taccumulate,
		    blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
		    conf.streams[j]>>>
		    (&conf.dbuf_out3[dbufout2_offset],
		     &conf.dbuf_out4[dbufout4_offset],
		     conf.nsamp3,
		     conf.naccumulate);
		  break;
		}
	      CudaSafeKernelLaunch();
	      
	      /* Setup ethernet packets */
	      time_stamp_i = (time_t)time_stamp_f;
	      strftime(time_stamp, FITS_TIME_STAMP_LEN, FITS_TIMESTR, gmtime(&time_stamp_i)); 
	      sprintf(time_stamp, "%s.%04dUTC ", time_stamp, (int)((time_stamp_f - time_stamp_i) * 1E4 + 0.5));
	      for(k = 0; k < NDATA_PER_SAMP_FULL; k++)
	      	{		  
	      	  eth_index = i * conf.nstream * NDATA_PER_SAMP_FULL + j * NDATA_PER_SAMP_FULL + k;
	      	  
	      	  strncpy(conf.fits[eth_index].time_stamp, time_stamp, FITS_TIME_STAMP_LEN);		  
	      	  conf.fits[eth_index].tsamp = time_res_stream;
	      	  conf.fits[eth_index].nchan = conf.nchan_out;
	      	  conf.fits[eth_index].chan_width = chan_width;
	      	  conf.fits[eth_index].pol_type = conf.pol_type;
	      	  conf.fits[eth_index].beam_index  = conf.beam_index;
	      	  conf.fits[eth_index].center_freq = conf.center_freq;
	      	  conf.fits[eth_index].nchunk = 1;
	      	  conf.fits[eth_index].nchunk = 0;
	      	  conf.fits[eth_index].chunk_index = k;
	      	  
	      	  if(conf.pol_type == 2)
	      	    {
	      	      if(k < conf.pol_type)
	      		CudaSafeCall(cudaMemcpyAsync(conf.fits[eth_index].data,
	      					     &conf.dbuf_out4[dbufout4_offset +
	      							     conf.nchan_out  *
	      							     (NDATA_PER_SAMP_FULL + k)],
	      					     conf.nchan_out * NBYTE_FLOAT,
	      					     cudaMemcpyDeviceToHost,
	      					     conf.streams[j]));
	      	    }
	      	  else
	      	    CudaSafeCall(cudaMemcpyAsync(conf.fits[eth_index].data,
	      					 &conf.dbuf_out4[dbufout4_offset +
	      							 k * conf.nchan_out],
	      					 conf.nchan_out * NBYTE_FLOAT,
	      					 cudaMemcpyDeviceToHost,
	      					 conf.streams[j]));
	      	}
	      time_stamp_f += time_res_stream;
	    }
	}
      CudaSynchronizeCall(); // Sync here is for multiple streams

      /* Send all packets from the previous buffer block with one go */
	      
      log_add(conf.log_file, "INFO", 1, log_mutex, "before closing old buffer block");
      ipcbuf_mark_filled(conf.db_out, (uint64_t)(cbufsz * conf.scale_dtsz));
      ipcbuf_mark_cleared(conf.db_in);
      log_add(conf.log_file, "INFO", 1, log_mutex, "after closing old buffer block");

      time_offset += time_res_blk;
      fprintf(stdout, "BASEBAND2FILTERBANK, finished %f seconds data\n", time_offset);
      fflush(stdout);
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
	  swap_select_transpose_ptf_kernel<<<gridsize_swap_select_transpose, blocksize_swap_select_transpose, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2, conf.cufft_nx, conf.cufft_mod, conf.nchan_keep_chan, conf.nchan_keep_band, conf.nchan_edge);
	  CudaSafeKernelLaunch();
	  
	  switch (blocksize_detect_faccumulate_pad_transpose.x )
	    {
	    case 1024:
	      detect_faccumulate_pad_transpose1_kernel<1024><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;
	    case 512:
	      detect_faccumulate_pad_transpose1_kernel< 512><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 256:
	      detect_faccumulate_pad_transpose1_kernel< 256><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 128:
	      detect_faccumulate_pad_transpose1_kernel< 128><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 64:
	      detect_faccumulate_pad_transpose1_kernel<  64><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 32:
	      detect_faccumulate_pad_transpose1_kernel<  32><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 16:
	      detect_faccumulate_pad_transpose1_kernel<  16><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 8:
	      detect_faccumulate_pad_transpose1_kernel<   8><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 4:
	      detect_faccumulate_pad_transpose1_kernel<   4><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 2:
	      detect_faccumulate_pad_transpose1_kernel<   2><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;			     
	    case 1:
	      detect_faccumulate_pad_transpose1_kernel<   1><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2, conf.naccumulate_pad);
	      break;
	    }
	  CudaSafeKernelLaunch();
	  
	  switch (blocksize_taccumulate.x)
	    {
	    case 1024:
	      reduce10_kernel<1024><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan_out], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 512:
	      reduce10_kernel< 512><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan_out], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 256:
	      reduce10_kernel< 256><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan_out], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 128:
	      reduce10_kernel< 128><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan_out], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 64:
	      reduce10_kernel<  64><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan_out], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 32:
	      reduce10_kernel<  32><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan_out], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 16:
	      reduce10_kernel<  16><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan_out], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 8:
	      reduce10_kernel<   8><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan_out], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 4:
	      reduce10_kernel<   4><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan_out], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 2:
	      reduce10_kernel<   2><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan_out], conf.naccumulate, conf.ndim_scale);
	      break;
	    case 1:
	      reduce10_kernel<   1><<<gridsize_taccumulate, blocksize_taccumulate, blocksize_taccumulate.x * NBYTE_CUFFT_COMPLEX, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.offset_scale_d[j*conf.nchan_out], conf.naccumulate, conf.ndim_scale);
	      break;
	    }
	  CudaSafeKernelLaunch();
	}
    }
  CudaSynchronizeCall(); // Sync here is for multiple streams
  
  /* Get the scale of each chanel */
  scale3_kernel<<<gridsize_scale, blocksize_scale>>>(conf.offset_scale_d, conf.nchan_out, conf.nstream, SCL_NSIG, SCL_UINT8);
  CudaSafeKernelLaunch();
  CudaSynchronizeCall();
  
  CudaSafeCall(cudaMemcpy(conf.offset_scale_h, conf.offset_scale_d, NBYTE_CUFFT_COMPLEX * conf.nchan_out, cudaMemcpyDeviceToHost));
  CudaSynchronizeCall();
  
  /* Record scale into file */
  sprintf(fname, "%s/%s_scale.txt", conf.dir, conf.utc_start);
  fp = fopen(fname, "w");
  if(fp == NULL)
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Can not open scale file, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Can not open scale file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(conf);
      fclose(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
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

  if(conf.fits)
    free(conf.fits);
  
  if(conf.dbuf_in)
    cudaFree(conf.dbuf_in);
  if(conf.dbuf_out1)
    cudaFree(conf.dbuf_out1);
  if(conf.dbuf_out2)
    cudaFree(conf.dbuf_out2);
  if(conf.dbuf_out3)
    cudaFree(conf.dbuf_out3);
  if(conf.dbuf_out4)
    cudaFree(conf.dbuf_out4);
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

int read_register_header(conf_t *conf)
{
  uint64_t hdrsz;
  char *hdrbuf_in = NULL, *hdrbuf_out = NULL;
  uint64_t file_size, bytes_per_second;
  
  hdrbuf_in  = ipcbuf_get_next_read(conf->hdu_in->header_block, &hdrsz);  
  if (!hdrbuf_in)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting header_buf, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  if(hdrsz != DADA_HDRSZ)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Header size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Header size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }

  hdrbuf_out = ipcbuf_get_next_write(conf->hdu_out->header_block);
  if (!hdrbuf_out)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting header_buf, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }  
  if (ascii_header_get(hdrbuf_in, "FILE_SIZE", "%"SCNu64"", &file_size) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting FILE_SIZE, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "FILE_SIZE from DADA header is %"PRIu64"", file_size);
  
  if (ascii_header_get(hdrbuf_in, "BYTES_PER_SECOND", "%"SCNu64"", &bytes_per_second) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "BYTES_PER_SECOND from DADA header is %"PRIu64"", bytes_per_second);
  
  if (ascii_header_get(hdrbuf_in, "TSAMP", "%lf", &conf->tsamp_in) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting TSAMP, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "TSAMP from DADA header is %f", conf->tsamp_in);
  
  /* Get utc_start from hdrin */
  
  if (ascii_header_get(hdrbuf_in, "BEAM_INDEX", "%d", &conf->beam_index) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting BEAM_INDEX, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting BEAM_INDEX, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "BEAM_INDEX from DADA header is %d", conf->beam_index);
  
  if(ascii_header_get(hdrbuf_in, "FREQ", "%lf", &(conf->center_freq)) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error egtting FREQ, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting FREQ, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      log_close(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  if (ascii_header_get(hdrbuf_in, "UTC_START", "%s", conf->utc_start) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting UTC_START, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "UTC_START from DADA header is %f", conf->utc_start);
  
  if (ascii_header_get(hdrbuf_in, "PICOSECONDS", "%"SCNu64"", &(conf->picoseconds)) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting PICOSECONDS, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting PICOSECONDS, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "UTC_START from DADA header is %f", conf->utc_start);
  
  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header
  
  file_size = (uint64_t)(file_size * conf->scale_dtsz);
  bytes_per_second = (uint64_t)(bytes_per_second * conf->scale_dtsz);
  
  if (ascii_header_set(hdrbuf_out, "NCHAN", "%d", conf->nchan_out) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting NCHAN, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting NCHAN, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "NCHAN to DADA header is %d", conf->nchan_out);
  
  if (ascii_header_set(hdrbuf_out, "BW", "%f", -conf->bandwidth) < 0)  // Reverse frequency order
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting BW, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting BW, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "BW to DADA header is %f", -conf->bandwidth);
  
  conf->tsamp_out1 = conf->tsamp_in * conf->cufft_nx;
  if (ascii_header_set(hdrbuf_out, "TSAMP", "%f", conf->tsamp_out1) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting TSAMP, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "TSAMP to DADA header is %f microseconds", conf->tsamp_out1);
  
  if (ascii_header_set(hdrbuf_out, "NBIT", "%d", NBIT_FILTERBANK) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting NBIT, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "NBIT to DADA header is %d", NBIT_FILTERBANK);
  
  if (ascii_header_set(hdrbuf_out, "NDIM", "%d", NDIM_FILTERBANK) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting NDIM, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting NDIM, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "NDIM to DADA header is %d", NDIM_FILTERBANK);
  
  if (ascii_header_set(hdrbuf_out, "NPOL", "%d", NPOL_FILTERBANK) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting NPOL, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting NPOL, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "NPOL to DADA header is %d", NPOL_FILTERBANK);
  
  if (ascii_header_set(hdrbuf_out, "FILE_SIZE", "%"PRIu64"", file_size) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: BASEBAND2FILTERBANK_ERROR:\tError setting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "FILE_SIZE to DADA header is %"PRIu64"", file_size);
  
  if (ascii_header_set(hdrbuf_out, "BYTES_PER_SECOND", "%"PRIu64"", bytes_per_second) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "BYTES_PER_SECOND to DADA header is %"PRIu64"", bytes_per_second);
  
  if(ipcbuf_mark_cleared (conf->hdu_in->header_block))  // We are the only one reader, so that we can clear it after read;
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error header_clear, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error header_clear, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  /* donot set header parameters anymore */
  if (ipcbuf_mark_filled (conf->hdu_out->header_block, conf->hdrsz) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error header_fill, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      CudaSafeCall(cudaProfilerStop());
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
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: ndf_per_chunk_rbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_rbufin, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "ndf_per_chunk_rbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort", conf.ndf_per_chunk_rbufin, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "Each input ring buffer block has %"PRIu64" packets per frequency chunk", conf.ndf_per_chunk_rbufin); 

  if(conf.nstream <= 0)
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: nstream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.nstream, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "nstream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.nstream, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "%d streams run on GPU", conf.nstream);
  
  if(conf.ndf_per_chunk_stream == 0)
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: ndf_per_chunk_stream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_stream, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "ndf_per_chunk_stream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.ndf_per_chunk_stream, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "Each stream process %d packets per frequency chunk", conf.ndf_per_chunk_stream);

  log_add(conf.log_file, "INFO", 1, log_mutex, "The runtime information is %s", conf.dir);  // Checked already
  
  if(conf.sod == 1)
    log_add(conf.log_file, "INFO", 1, log_mutex, "The filterbank data is enabled at the beginning");
  else
    log_add(conf.log_file, "INFO", 1, log_mutex, "The filterbank data is NOT enabled at the beginning");

  if(conf.nchunk_in<=0 || conf.nchunk_in>NCHUNK_MAX)    
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: nchunk_in shoule be in (0 %d], but it is %d, which happens at \"%s\", line [%d], has to abort\n", NCHUNK_MAX, conf.nchunk_in, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "nchunk_in shoule be in (0 %d], but it is %d, which happens at \"%s\", line [%d], has to abort", NCHUNK_MAX, conf.nchunk_in, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }  
  log_add(conf.log_file, "INFO", 1, log_mutex, "%d chunks of input data", conf.nchunk_in);

  if(conf.cufft_nx<=0)    
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: cufft_nx shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.cufft_nx, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "cufft_nx shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.cufft_nx, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "We use %d points FFT", conf.cufft_nx);
  
  if(conf.nchan_out <= 0)
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: nchan_out should be positive, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.nchan_keep_band, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "nchan_out should be positive, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.nchan_keep_band, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }

  if((log2((double)conf.nchan_out) - floor(log2((double)conf.nchan_out))) != 0)
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: nchan_out should be power of 2, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.nchan_keep_band, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "nchan_out should be power of 2, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.nchan_keep_band, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "We output %d channels", conf.nchan_out);
  
  if(conf.nchan_keep_band<=0)    
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: nchan_keep_band shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.nchan_keep_band, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "nchan_keep_band shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.nchan_keep_band, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      CudaSafeCall(cudaProfilerStop());
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "We keep %d fine channels for the whole band after FFT", conf.nchan_keep_band); 
  
  return EXIT_SUCCESS;
}