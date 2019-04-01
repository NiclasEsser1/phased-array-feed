#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <cuda_profiler_api.h>
#include <unistd.h>

#include "baseband2filterbank.cuh"
#include "cudautil.cuh"
#include "kernel.cuh"
#include "log.h"
#include "constants.h"

// Have to discuss the sensor thing!!!
// Tsamp of the monitor for search
// Type of pol;

// We can time accumulate in each stream to get monitor data for search mode
// Tsamp for this case is 0.110592 seconds for 1024 DF per stream configuration
// If we only want to monitor Stokes I, we can tranpose the TF data to FT and time accumulate it
// If we want to monitor Stokes I, AABB or IQUV, we need to calculate the spectrum after the PTF
// Frequency accumulate it, tranpose from PTF to PFT and accumulate in time
// The spectrum calculation and frequency accumulation can happen wiht the detect_faccumulate_scale
// Filterbank data and spectrum data will out with separate buffers
// Send the data once the buffer block is done

// Make the scale calculation unblock, done;

// Not because of nchan_keep_band
queue_t *queue_fits_monitor;
queue_t *queue_fits_spectral;
int quit = 0; // 0 means no quit, 1 means quit normal, 2 means quit with problem;

int default_arguments(conf_t *conf)
{
  memset(conf->dir, 0x00, sizeof(conf->dir));
  sprintf(conf->dir, "unset"); // Default with "unset"
  memset(conf->ip_monitor, 0x00, sizeof(conf->ip_monitor));
  sprintf(conf->ip_monitor, "unset"); // Default with "unset"
  memset(conf->ip_spectral, 0x00, sizeof(conf->ip_spectral));
  sprintf(conf->ip_spectral, "unset"); // Default with "unset"
  
  conf->ndf_per_chunk_rbufin = 0; // Default with an impossible value
  conf->nstream          = -1; // Default with an impossible value
  conf->ndf_per_chunk_stream = 0; // Default with an impossible value
  conf->sod = 0;                   // Default no SOD at the beginning
  conf->nchunk_in = -1;
  conf->cufft_nx = -1;
  conf->nchan_out = -1;
  conf->nchan_keep_band = -1;
  
  conf->port_monitor = -1;
  conf->monitor = 0; // default no monitor
  conf->ptype_monitor = -1;

  conf->spectral2disk = 0; // 
  conf->spectral2network = 0; //
  conf->port_spectral = -1;
  conf->nblk_accumulate = -1;
  conf->ptype_spectral = -1;
  conf->start_chunk = -1;
  conf->nchunk_in_spectral = -1;
  conf->cufft_nx_spectral = -1;
  conf->sod_spectral = 0;
  
  return EXIT_SUCCESS;
}

int initialize_baseband2filterbank(conf_t *conf)
{
  int i;
  int iembed, istride, idist, oembed, ostride, odist, batch, nx;
  uint64_t naccumulate_pow2;
  
  /* Prepare parameters */
  conf->nrepeat_per_blk = conf->ndf_per_chunk_rbufin / (conf->ndf_per_chunk_stream * conf->nstream);
  conf->nchan_in        = conf->nchunk_in * NCHAN_PER_CHUNK;
  conf->nchan_keep_chan = (int)(conf->cufft_nx / OVER_SAMP_RATE);
  conf->nchan_keep_band = conf->nchan_keep_chan * conf->nchan_in - (conf->nchan_keep_chan * conf->nchan_in) % conf->nchan_out;
  conf->cufft_mod       = (int)(0.5 * conf->nchan_keep_chan);
  conf->nchan_edge      = (int)(0.5 * conf->nchan_in * conf->nchan_keep_chan - 0.5 * conf->nchan_keep_band);
  conf->inverse_nchan_rate = conf->nchan_in * conf->nchan_keep_chan/(double)conf->nchan_keep_band;
  conf->scale_dtsz         = conf->nchan_out * NBYTE_FILTERBANK / (double)(conf->nchan_in * conf->cufft_nx * NPOL_BASEBAND * NDIM_BASEBAND * NBYTE_BASEBAND);
  conf->bandwidth       = conf->nchan_keep_band/(double)conf->nchan_keep_chan;
  log_add(conf->log_file, "INFO", 1,  "We have %d channels input", conf->nchan_in);
  log_add(conf->log_file, "INFO", 1,  "The mod to reduce oversampling is %d", conf->cufft_mod);
  log_add(conf->log_file, "INFO", 1,  "We will keep %d fine channels for each input channel after FFT", conf->nchan_keep_chan);
  log_add(conf->log_file, "INFO", 1,  "We keep %d fine channels for the whole band after FFT", conf->nchan_keep_band); 
  if(conf->nchan_edge<0) // Check the nchan_keep_band further
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: nchan_edge can not be a negative number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf->nchan_edge, __FILE__, __LINE__);
      log_add(conf->log_file, "ERR", 1,  "nchan_edge can not be a negative number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf->nchan_edge, __FILE__, __LINE__);
      
      log_close(conf->log_file);
      exit(EXIT_FAILURE);
    }  
  log_add(conf->log_file, "INFO", 1,  "We will drop %d fine channels at the band edge for frequency accumulation", conf->nchan_edge);
  log_add(conf->log_file, "INFO", 1,  "%f percent fine channels (after down sampling) are kept for frequency accumulation", 1.0/conf->inverse_nchan_rate * 100.);
  log_add(conf->log_file, "INFO", 1,  "The data size rate between filterbank and baseband data is %f", conf->scale_dtsz);
  log_add(conf->log_file, "INFO", 1,  "The bandwidth for the final output is %f MHz", conf->bandwidth);
  log_add(conf->log_file, "INFO", 1,  "%d run to finish one ring buffer block", conf->nrepeat_per_blk);
  
  if(conf->monitor == 1)
    {
      conf->nseg_per_blk = conf->nstream * conf->nrepeat_per_blk;
      conf->neth_per_blk = conf->nseg_per_blk * NDATA_PER_SAMP_FULL;
      log_add(conf->log_file, "INFO", 1,  "%d network packets are requied for each buffer block", conf->neth_per_blk);
      
      conf->dtsz_network    = NBYTE_FLOAT * conf->nchan_out;
      conf->pktsz_network   = conf->dtsz_network + 3 * NBYTE_FLOAT + 6 * NBYTE_INT + FITS_TIME_STAMP_LEN;
      log_add(conf->log_file, "INFO", 1,  "Network data size for monitor is %d", conf->dtsz_network);
      log_add(conf->log_file, "INFO", 1,  "Network packet size for monitor is %d", conf->pktsz_network);
      queue_fits_monitor = create_queue(10 * conf->neth_per_blk);
    }
  
  /* Prepare buffer, stream and fft plan for process */
  conf->nsamp_in       = conf->ndf_per_chunk_stream * conf->nchan_in * NSAMP_DF;
  conf->npol_in        = conf->nsamp_in * NPOL_BASEBAND;
  conf->ndata_in       = conf->npol_in  * NDIM_BASEBAND;
  
  conf->nsamp_keep       = conf->nsamp_in * conf->nchan_keep_band / (conf->nchan_in * conf->cufft_nx);
  conf->npol_keep        = conf->nsamp_keep * NPOL_BASEBAND;
  conf->ndata_keep       = conf->npol_keep  * NDIM_BASEBAND;

  conf->nsamp_filterbank = conf->nsamp_keep * conf->nchan_out / conf->nchan_keep_band;
  conf->npol_filterbank  = conf->nsamp_filterbank * NPOL_FILTERBANK;
  conf->ndata_filterbank = conf->npol_filterbank  * NDIM_FILTERBANK;
  conf->ndim_scale       = conf->ndf_per_chunk_rbufin * NSAMP_DF / conf->cufft_nx;   // We do not average in time and here we work on detected data;

  nx        = conf->cufft_nx;
  batch     = conf->npol_in / conf->cufft_nx;
  
  iembed    = nx;
  istride   = 1;
  idist     = nx;
  
  oembed    = nx;
  ostride   = 1;
  odist     = nx;

  conf->streams = NULL;
  conf->fft_plans = NULL;
  conf->streams = (cudaStream_t *)malloc(conf->nstream * sizeof(cudaStream_t));
  conf->fft_plans = (cufftHandle *)malloc(conf->nstream * sizeof(cufftHandle));
  for(i = 0; i < conf->nstream; i ++)
    {
      CudaSafeCall(cudaStreamCreate(&conf->streams[i]));
      CufftSafeCall(cufftPlanMany(&conf->fft_plans[i], CUFFT_RANK, &nx, &iembed, istride, idist, &oembed, ostride, odist, CUFFT_C2C, batch));
      CufftSafeCall(cufftSetStream(conf->fft_plans[i], conf->streams[i]));
    }
  
  conf->sbufin_size   = conf->ndata_in * NBYTE_BASEBAND;
  conf->sbufout_size_filterbank = conf->ndata_filterbank * NBYTE_FILTERBANK;
  conf->sbufout_size_monitor = conf->nsamp_filterbank * NBYTE_FLOAT * NDATA_PER_SAMP_RT;
  
  conf->bufin_size    = conf->nstream * conf->sbufin_size;
  conf->bufout_size_filterbank  = conf->nstream * conf->sbufout_size_filterbank;
  conf->bufout_size_monitor  = conf->nstream * conf->sbufout_size_monitor;
  
  conf->sbufrt1_size = conf->npol_in * NBYTE_CUFFT_COMPLEX;
  conf->sbufrt2_size = conf->npol_keep * NBYTE_CUFFT_COMPLEX;
  conf->bufrt1_size  = conf->nstream * conf->sbufrt1_size;
  conf->bufrt2_size  = conf->nstream * conf->sbufrt2_size;
    
  conf->hbufin_offset = conf->sbufin_size;
  conf->dbufin_offset = conf->sbufin_size / (NBYTE_BASEBAND * NPOL_BASEBAND * NDIM_BASEBAND);
  conf->bufrt1_offset = conf->sbufrt1_size / NBYTE_CUFFT_COMPLEX;
  conf->bufrt2_offset = conf->sbufrt2_size / NBYTE_CUFFT_COMPLEX;
  
  conf->dbufout_offset_filterbank = conf->sbufout_size_filterbank / NBYTE_FILTERBANK;
  conf->hbufout_offset_filterbank = conf->sbufout_size_filterbank;
  conf->dbufout_offset_monitor1 = conf->sbufout_size_monitor / NBYTE_FLOAT;
  conf->dbufout_offset_monitor2 = conf->dbufout_offset_monitor1;
  conf->dbufout_offset_monitor3 = conf->nchan_out * NDATA_PER_SAMP_RT;

  conf->dbuf_in = NULL;
  conf->dbuf_out_filterbank = NULL;
  conf->buf_rt1 = NULL;
  conf->buf_rt2 = NULL;
  conf->offset_scale_d = NULL;
  conf->offset_scale_h = NULL;
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_in, conf->bufin_size));
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out_filterbank, conf->bufout_size_filterbank));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt1, conf->bufrt1_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt2, conf->bufrt2_size));
  log_add(conf->log_file, "INFO", 1,  "bufin_size is %"PRIu64", bufout_size_filterbank is %"PRIu64", bufrt1_size is %"PRIu64" and bufrt2_size is %"PRIu64"",
	  conf->bufin_size, conf->bufrt1_size, conf->bufrt2_size, conf->bufout_size_filterbank);
  
  if(conf->monitor == 1)
    {      
      conf->dbuf_out_monitor1 = NULL;
      conf->dbuf_out_monitor2 = NULL;
      conf->dbuf_out_monitor3 = NULL;
      CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out_monitor1, conf->bufout_size_monitor));
      log_add(conf->log_file, "INFO", 1,  "bufout_size_monitor is %"PRIu64"", conf->bufout_size_monitor);      
      CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out_monitor2, conf->bufout_size_monitor));
      CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out_monitor3, conf->nchan_out * NDATA_PER_SAMP_RT * conf->nstream * NBYTE_FLOAT));
      log_add(conf->log_file, "INFO", 1,  "bufout_size_monitors is %"PRIu64"", conf->nchan_out * NDATA_PER_SAMP_RT * conf->nstream * NBYTE_FLOAT);      
    }
  CudaSafeCall(cudaMalloc((void **)&conf->offset_scale_d, conf->nstream * conf->nchan_out * NBYTE_CUFFT_COMPLEX));
  CudaSafeCall(cudaMallocHost((void **)&conf->offset_scale_h, conf->nchan_out * NBYTE_CUFFT_COMPLEX));
  //CudaSafeCall(cudaMemset((void *)conf->offset_scale_d, 0, sizeof(conf->offset_scale_d)));// We have to clear the memory for this parameter
  CudaSafeCall(cudaMemset((void *)conf->offset_scale_d, 0, conf->nstream * conf->nchan_out * NBYTE_CUFFT_COMPLEX));// We have to clear the memory for this parameter
  
  /* Prepare the setup of kernels */
  conf->gridsize_unpack.x = conf->ndf_per_chunk_stream;
  conf->gridsize_unpack.y = conf->nchunk_in;
  conf->gridsize_unpack.z = 1;
  conf->blocksize_unpack.x = NSAMP_DF; 
  conf->blocksize_unpack.y = NCHAN_PER_CHUNK;
  conf->blocksize_unpack.z = 1;
  log_add(conf->log_file, "INFO", 1,  "The configuration of unpack kernel is (%d, %d, %d) and (%d, %d, %d)",
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
  log_add(conf->log_file, "INFO", 1,  "The configuration of transpose kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_transpose.x, conf->gridsize_transpose.y, conf->gridsize_transpose.z,
	  conf->blocksize_transpose.x, conf->blocksize_transpose.y, conf->blocksize_transpose.z);
  
  conf->gridsize_swap_select_transpose.x = conf->nchan_in;
  conf->gridsize_swap_select_transpose.y = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx;
  conf->gridsize_swap_select_transpose.z = 1;  
  conf->blocksize_swap_select_transpose.x = conf->cufft_nx;
  conf->blocksize_swap_select_transpose.y = 1;
  conf->blocksize_swap_select_transpose.z = 1;  
  log_add(conf->log_file, "INFO", 1,  "The configuration of swap_select_transpose kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_swap_select_transpose.x, conf->gridsize_swap_select_transpose.y, conf->gridsize_swap_select_transpose.z,
	  conf->blocksize_swap_select_transpose.x, conf->blocksize_swap_select_transpose.y, conf->blocksize_swap_select_transpose.z);        

  conf->naccumulate_pad = conf->nchan_keep_band/conf->nchan_out;
  naccumulate_pow2      = (uint64_t)pow(2.0, floor(log2((double)conf->naccumulate_pad)));
  conf->gridsize_detect_faccumulate_pad_transpose.x = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx;
  conf->gridsize_detect_faccumulate_pad_transpose.y = conf->nchan_out;
  conf->gridsize_detect_faccumulate_pad_transpose.z = 1;
  conf->blocksize_detect_faccumulate_pad_transpose.x = (naccumulate_pow2<1024)?naccumulate_pow2:1024;
  conf->blocksize_detect_faccumulate_pad_transpose.y = 1;
  conf->blocksize_detect_faccumulate_pad_transpose.z = 1;
  log_add(conf->log_file, "INFO", 1,  "The configuration of detect_faccumulate_pad_transpose kernel is (%d, %d, %d) and (%d, %d, %d), naccumulate is %"PRIu64"",
	  conf->gridsize_detect_faccumulate_pad_transpose.x, conf->gridsize_detect_faccumulate_pad_transpose.y, conf->gridsize_detect_faccumulate_pad_transpose.z,
	  conf->blocksize_detect_faccumulate_pad_transpose.x, conf->blocksize_detect_faccumulate_pad_transpose.y, conf->blocksize_detect_faccumulate_pad_transpose.z, conf->naccumulate_pad);

  conf->naccumulate_scale = conf->nchan_keep_band/conf->nchan_out;
  naccumulate_pow2        = (uint64_t)pow(2.0, floor(log2((double)conf->naccumulate_scale)));
  conf->gridsize_detect_faccumulate_scale.x = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx;
  conf->gridsize_detect_faccumulate_scale.y = conf->nchan_out;
  conf->gridsize_detect_faccumulate_scale.z = 1;
  conf->blocksize_detect_faccumulate_scale.x = (naccumulate_pow2<1024)?naccumulate_pow2:1024;
  conf->blocksize_detect_faccumulate_scale.y = 1;
  conf->blocksize_detect_faccumulate_scale.z = 1;
  log_add(conf->log_file, "INFO", 1,  "The configuration of detect_faccumulate_scale kernel is (%d, %d, %d) and (%d, %d, %d), naccumulate is %"PRIu64"",
	  conf->gridsize_detect_faccumulate_scale.x, conf->gridsize_detect_faccumulate_scale.y, conf->gridsize_detect_faccumulate_scale.z,
	  conf->blocksize_detect_faccumulate_scale.x, conf->blocksize_detect_faccumulate_scale.y, conf->blocksize_detect_faccumulate_scale.z, conf->naccumulate_scale);

  conf->naccumulate = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx; 
  naccumulate_pow2  = (uint64_t)pow(2.0, floor(log2((double)conf->naccumulate)));
  conf->gridsize_taccumulate.x = conf->nchan_out;
  conf->gridsize_taccumulate.y = 1;
  conf->gridsize_taccumulate.z = 1;
  conf->blocksize_taccumulate.x = (naccumulate_pow2<1024)?naccumulate_pow2:1024;
  conf->blocksize_taccumulate.y = 1;
  conf->blocksize_taccumulate.z = 1;
  log_add(conf->log_file, "INFO", 1,  "The configuration of accumulate kernel is (%d, %d, %d) and (%d, %d, %d), naccumulate is %"PRIu64"",
	  conf->gridsize_taccumulate.x, conf->gridsize_taccumulate.y, conf->gridsize_taccumulate.z,
	  conf->blocksize_taccumulate.x, conf->blocksize_taccumulate.y, conf->blocksize_taccumulate.z, conf->naccumulate);
  
  conf->gridsize_scale.x = 1;
  conf->gridsize_scale.y = 1;
  conf->gridsize_scale.z = 1;
  conf->blocksize_scale.x = conf->nchan_out;
  conf->blocksize_scale.y = 1;
  conf->blocksize_scale.z = 1;
  log_add(conf->log_file, "INFO", 1,  "The configuration of scale kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_scale.x, conf->gridsize_scale.y, conf->gridsize_scale.z,
	  conf->blocksize_scale.x, conf->blocksize_scale.y, conf->blocksize_scale.z);
  
  /* attach to input ring buffer */
  conf->hdu_in = dada_hdu_create(NULL);
  dada_hdu_set_key(conf->hdu_in, conf->key_in);
  if(dada_hdu_connect(conf->hdu_in) < 0)
    {
      log_add(conf->log_file, "ERR", 1,  "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }  
  conf->db_in = (ipcbuf_t *) conf->hdu_in->data_block;
  conf->hdr_in = (ipcbuf_t *) conf->hdu_in->header_block;
  conf->rbufin_size = ipcbuf_get_bufsz(conf->db_in);
  log_add(conf->log_file, "INFO", 1,  "Input buffer block size is %"PRIu64".", conf->rbufin_size);
  if(conf->rbufin_size != (conf->bufin_size * conf->nrepeat_per_blk))  
    {
      log_add(conf->log_file, "ERR", 1,  "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }

  struct timespec start, stop;
  double elapsed_time;
  clock_gettime(CLOCK_REALTIME, &start);
  /* registers the existing host memory range for use by CUDA */
  dada_cuda_dbregister(conf->hdu_in);  // To put this into capture does not improve the memcpy!!!  
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "elapsed_time for dbregister of input ring buffer is %f\n", elapsed_time);
  fflush(stdout);
  
  if(ipcbuf_get_bufsz(conf->hdr_in) != DADA_DEFAULT_HEADER_SIZE)    // This number should match
    {
      log_add(conf->log_file, "ERR", 1,  "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }
  
  /* make ourselves the read client */
  if(dada_hdu_lock_read(conf->hdu_in) < 0)
    {
      log_add(conf->log_file, "ERR", 1,  "Error locking HDU, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }

  /* Prepare output ring buffer */
  conf->hdu_out = dada_hdu_create(NULL);
  dada_hdu_set_key(conf->hdu_out, conf->key_out);
  if(dada_hdu_connect(conf->hdu_out) < 0)
    {
      log_add(conf->log_file, "ERR", 1,  "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }
  conf->db_out = (ipcbuf_t *) conf->hdu_out->data_block;
  conf->hdr_out = (ipcbuf_t *) conf->hdu_out->header_block;
  conf->rbufout_size_filterbank = ipcbuf_get_bufsz(conf->db_out);
  log_add(conf->log_file, "INFO", 1,  "Output buffer block size is %"PRIu64".", conf->rbufout_size_filterbank);
  
  clock_gettime(CLOCK_REALTIME, &start);
  /* registers the existing host memory range for use by CUDA */
  dada_cuda_dbregister(conf->hdu_out);  // To put this into capture does not improve the memcpy!!!
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "elapsed_time for dbregister of output ring buffer is %f\n", elapsed_time);
  fflush(stdout);
  
  if(conf->rbufout_size_filterbank != (conf->bufout_size_filterbank * conf->nrepeat_per_blk))  
    {
      log_add(conf->log_file, "ERR", 1,  "Buffer size mismatch, %"PRIu64"\t%"PRIu64", which happens at \"%s\", line [%d].", conf->rbufout_size_filterbank, conf->bufout_size_filterbank, __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Buffer size mismatch, %"PRIu64"\t%"PRIu64", which happens at \"%s\", line [%d].\n", conf->rbufout_size_filterbank, conf->bufout_size_filterbank, __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }
  
  if(ipcbuf_get_bufsz(conf->hdr_out) != DADA_DEFAULT_HEADER_SIZE)    // This number should match
    {
      log_add(conf->log_file, "ERR", 1,  "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }
  
  /* make ourselves the write client */
  if(dada_hdu_lock_write(conf->hdu_out) < 0)
    {
      log_add(conf->log_file, "ERR", 1,  "Error locking HDU, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }

  if(conf->sod == 0)
    {
      if(ipcbuf_disable_sod(conf->db_out) < 0)
	{
	  log_add(conf->log_file, "ERR", 1,  "Can not write data before start, which happens at \"%s\", line [%d], has to abort.", __FILE__, __LINE__);
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  
	  destroy_baseband2filterbank(*conf);
	  fclose(conf->log_file);
	  exit(EXIT_FAILURE);
	}
    }

  if((conf->spectral2network == 1) || (conf->spectral2disk == 1))
    {
      conf->naccumulate_spectral = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx_spectral;
      conf->nchan_keep_chan_spectral = (int)(conf->cufft_nx_spectral / OVER_SAMP_RATE);
      conf->nchan_in_spectral = conf->nchunk_in_spectral * NCHAN_PER_CHUNK;
      conf->nchan_out_spectral = conf->nchan_keep_chan_spectral * conf->nchan_in_spectral;
      conf->cufft_mod_spectral = (int)(0.5*conf->nchan_keep_chan_spectral);
      conf->scale_dtsz_spectral = NBYTE_SPECTRAL * NDATA_PER_SAMP_FULL * conf->nchan_out_spectral / (double)(NBYTE_BASEBAND * NPOL_BASEBAND * NDIM_BASEBAND * conf->ndf_per_chunk_rbufin * conf->nchan_in * NSAMP_DF * conf->nblk_accumulate); // replace NDATA_PER_SAMP_FULL with conf->pol_type if we do not fill 0 for other pols

      log_add(conf->log_file, "INFO", 1,  "naccumulate_spectral is %"PRIu64"", conf->naccumulate_spectral);
      log_add(conf->log_file, "INFO", 1,  "nchan_out_spectral is %"PRIu64"", conf->nchan_out_spectral);
      log_add(conf->log_file, "INFO", 1,  "We have %d channels input for spectral", conf->nchan_in_spectral);
      log_add(conf->log_file, "INFO", 1,  "The mod to reduce oversampling is %d for spectral", conf->cufft_mod_spectral);
      log_add(conf->log_file, "INFO", 1,  "We will keep %d fine channels for each input channel after FFT for spectral", conf->nchan_keep_chan_spectral);
      log_add(conf->log_file, "INFO", 1,  "The data size rate between spectral and baseband data is %E for spectral", conf->scale_dtsz_spectral);
      
      conf->nsamp_in_spectral = conf->ndf_per_chunk_stream * conf->nchan_in_spectral * NSAMP_DF;
      conf->npol_in_spectral  = conf->nsamp_in_spectral * NPOL_BASEBAND;
      conf->ndata_in_spectral = conf->npol_in_spectral  * NDIM_BASEBAND;
      
      conf->nsamp_keep_spectral = conf->nsamp_in_spectral / OVER_SAMP_RATE;
      conf->npol_keep_spectral  = conf->nsamp_keep_spectral * NPOL_BASEBAND;
      conf->ndata_keep_spectral = conf->npol_keep_spectral  * NDIM_BASEBAND;
      
      conf->nsamp_out_spectral = conf->nsamp_keep_spectral / conf->naccumulate_spectral;
      conf->ndata_out_spectral = conf->nsamp_out_spectral  * NDATA_PER_SAMP_RT;
            
      nx        = conf->cufft_nx_spectral;
      batch     = conf->npol_in_spectral / conf->cufft_nx_spectral;
      
      iembed    = nx;
      istride   = 1;
      idist     = nx;
      
      oembed    = nx;
      ostride   = 1;
      odist     = nx;

      conf->fft_plans_spectral = NULL;
      conf->fft_plans_spectral = (cufftHandle *)malloc(conf->nstream * sizeof(cufftHandle));
      for(i = 0; i < conf->nstream; i ++)
	{
	  CufftSafeCall(cufftPlanMany(&conf->fft_plans_spectral[i],
				      CUFFT_RANK, &nx, &iembed,
				      istride, idist, &oembed,
				      ostride, odist, CUFFT_C2C, batch));
	  CufftSafeCall(cufftSetStream(conf->fft_plans_spectral[i], conf->streams[i]));
	}

      conf->sbufout_size_spectral = conf->ndata_out_spectral * NBYTE_SPECTRAL;
      conf->sbufrt1_size_spectral = conf->npol_in_spectral * NBYTE_CUFFT_COMPLEX;
      conf->sbufrt2_size_spectral = conf->npol_keep_spectral * NBYTE_CUFFT_COMPLEX;
      
      conf->bufin_size_spectral   = conf->nstream * conf->sbufin_size_spectral;
      conf->bufout_size_spectral  = conf->nstream * conf->sbufout_size_spectral;      
      conf->bufrt1_size_spectral  = conf->nstream * conf->sbufrt1_size_spectral;
      conf->bufrt2_size_spectral  = conf->nstream * conf->sbufrt2_size_spectral;
      
      conf->hbufin_offset_spectral = conf->sbufin_size_spectral;
      conf->dbufin_offset_spectral = conf->sbufin_size_spectral / (NBYTE_BASEBAND * NPOL_BASEBAND * NDIM_BASEBAND);
      conf->bufrt1_offset_spectral = conf->sbufrt1_size_spectral / NBYTE_CUFFT_COMPLEX;
      conf->bufrt2_offset_spectral = conf->sbufrt2_size_spectral / NBYTE_CUFFT_COMPLEX;
      
      conf->dbufout_offset_spectral = conf->sbufout_size_spectral / NBYTE_SPECTRAL;
      
      conf->dbuf_out_spectral = NULL;
      conf->buf_rt1_spectral = NULL;
      conf->buf_rt2_spectral = NULL;
      CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out_spectral, conf->bufout_size_spectral));
      CudaSafeCall(cudaMalloc((void **)&conf->buf_rt1_spectral, conf->bufrt1_size_spectral));
      CudaSafeCall(cudaMalloc((void **)&conf->buf_rt2_spectral, conf->bufrt2_size_spectral));
      log_add(conf->log_file, "INFO", 1,  "Spectral out size is %"PRIu64", spectral rt1 size is %"PRIu64" and spectral rt2 size is %"PRIu64"", conf->bufout_size_spectral, conf->bufrt1_size_spectral, conf->bufrt2_size_spectral);
      
      /* Prepare the setup of kernels */
      conf->gridsize_swap_select_transpose_pft1.x = ceil(conf->cufft_nx_spectral / (double)TILE_DIM);  
      conf->gridsize_swap_select_transpose_pft1.y = ceil(conf->ndf_per_chunk_stream * NSAMP_DF / (double) (conf->cufft_nx_spectral * TILE_DIM));
      conf->gridsize_swap_select_transpose_pft1.z = conf->nchan_in_spectral;
      conf->blocksize_swap_select_transpose_pft1.x = TILE_DIM;
      conf->blocksize_swap_select_transpose_pft1.y = NROWBLOCK_TRANS;
      conf->blocksize_swap_select_transpose_pft1.z = 1;
      log_add(conf->log_file, "INFO", 1, 
	      "The configuration of swap_select_transpose_pft1 kernel is (%d, %d, %d) and (%d, %d, %d)",
	      conf->gridsize_swap_select_transpose_pft1.x,
	      conf->gridsize_swap_select_transpose_pft1.y,
	      conf->gridsize_swap_select_transpose_pft1.z,
	      conf->blocksize_swap_select_transpose_pft1.x,
	      conf->blocksize_swap_select_transpose_pft1.y,
	      conf->blocksize_swap_select_transpose_pft1.z);
      
      naccumulate_pow2 = (uint64_t)pow(2.0, floor(log2((double)conf->naccumulate_spectral)));
      conf->gridsize_spectral_taccumulate.x = conf->nchan_in_spectral;
      conf->gridsize_spectral_taccumulate.y = conf->nchan_keep_chan_spectral;
      conf->gridsize_spectral_taccumulate.z = 1;
      conf->blocksize_spectral_taccumulate.x = (naccumulate_pow2<1024)?naccumulate_pow2:1024;
      conf->blocksize_spectral_taccumulate.y = 1;
      conf->blocksize_spectral_taccumulate.z = 1; 
      log_add(conf->log_file, "INFO", 1, 
	      "The configuration of spectral_taccumulate kernel is (%d, %d, %d) and (%d, %d, %d)",
	      conf->gridsize_spectral_taccumulate.x,
	      conf->gridsize_spectral_taccumulate.y,
	      conf->gridsize_spectral_taccumulate.z,
	      conf->blocksize_spectral_taccumulate.x,
	      conf->blocksize_spectral_taccumulate.y,
	      conf->blocksize_spectral_taccumulate.z);        
      
      conf->gridsize_saccumulate.x = NDATA_PER_SAMP_RT;
      conf->gridsize_saccumulate.y = conf->nchan_keep_chan_spectral;
      conf->gridsize_saccumulate.z = 1;
      conf->blocksize_saccumulate.x = conf->nchan_in_spectral;
      conf->blocksize_saccumulate.y = 1;
      conf->blocksize_saccumulate.z = 1; 
      log_add(conf->log_file, "INFO", 1, 
	      "The configuration of saccumulate kernel is (%d, %d, %d) and (%d, %d, %d)",
	      conf->gridsize_saccumulate.x,
	      conf->gridsize_saccumulate.y,
	      conf->gridsize_saccumulate.z,
	      conf->blocksize_saccumulate.x,
	      conf->blocksize_saccumulate.y,
	      conf->blocksize_saccumulate.z);        
      
      if(conf->spectral2disk == 1)
	{
	  conf->hdu_out_spectral = dada_hdu_create(NULL);
	  dada_hdu_set_key(conf->hdu_out_spectral, conf->key_out_spectral);
	  if(dada_hdu_connect(conf->hdu_out_spectral) < 0)
	    {
	      log_add(conf->log_file, "ERR", 1,  "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      
	      destroy_baseband2filterbank(*conf);
	      fclose(conf->log_file);
	      exit(EXIT_FAILURE);    
	    }
	  conf->db_out_spectral = (ipcbuf_t *) conf->hdu_out_spectral->data_block;
	  conf->hdr_out_spectral = (ipcbuf_t *) conf->hdu_out_spectral->header_block;
	  conf->rbufout_size_spectral = ipcbuf_get_bufsz(conf->db_out_spectral);
	  log_add(conf->log_file, "INFO", 1,  "Output buffer block size is %"PRIu64".", conf->rbufout_size_spectral);
      
	  clock_gettime(CLOCK_REALTIME, &start);
	  dada_cuda_dbregister(conf->hdu_out_spectral); // To put this into capture does not improve the memcpy!!!
	  clock_gettime(CLOCK_REALTIME, &stop);
	  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
	  fprintf(stdout, "elapse_time for dbregister of output ring buffer is %f\n", elapsed_time);
	  fflush(stdout);
	  
	  if(conf->rbufout_size_spectral != conf->nsamp_out_spectral * NDATA_PER_SAMP_FULL * NBYTE_SPECTRAL)
	    {
	      // replace NDATA_PER_SAMP_FULL with conf->pol_type if we do not fill 0 for other pols
	      log_add(conf->log_file, "ERR", 1,  "Buffer size mismatch, %"PRIu64" vs %"PRIu64", which happens at \"%s\", line [%d].", conf->rbufout_size_spectral, conf->nsamp_out_spectral * NDATA_PER_SAMP_FULL * NBYTE_SPECTRAL, __FILE__, __LINE__);
	      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Buffer size mismatch, %"PRIu64" vs %"PRIu64", which happens at \"%s\", line [%d].\n", conf->rbufout_size_spectral, conf->nsamp_out_spectral * NDATA_PER_SAMP_FULL * NBYTE_SPECTRAL, __FILE__, __LINE__);
	      
	      destroy_baseband2filterbank(*conf);
	      fclose(conf->log_file);
	      exit(EXIT_FAILURE);    
	    }
	  
	  if(ipcbuf_get_bufsz(conf->hdr_out_spectral) != DADA_DEFAULT_HEADER_SIZE)    // This number should match
	    {
	      log_add(conf->log_file, "ERR", 1,  "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      
	      destroy_baseband2filterbank(*conf);
	      fclose(conf->log_file);
	      exit(EXIT_FAILURE);    
	    }
	  
	  /* make ourselves the write client */
	  if(dada_hdu_lock_write(conf->hdu_out_spectral) < 0)
	    {
	      log_add(conf->log_file, "ERR", 1,  "Error locking HDU, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      
	      destroy_baseband2filterbank(*conf);
	      fclose(conf->log_file);
	      exit(EXIT_FAILURE);
	    }
	  
	  if(conf->sod_spectral == 0)
	    {
	      if(ipcbuf_disable_sod(conf->db_out_spectral) < 0)
		{
		  log_add(conf->log_file, "ERR", 1,  "Can not write data before start, which happens at \"%s\", line [%d], has to abort.", __FILE__, __LINE__);
		  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
		  
		  destroy_baseband2filterbank(*conf);
		  fclose(conf->log_file);
		  exit(EXIT_FAILURE);
		}
	    }
	}
      if(conf->spectral2network == 1)
	{	  
	  conf->nchunk_network_spectral = conf->nchan_in_spectral; // We send spectral of one input channel per udp packet;
	  conf->nchan_per_chunk_network_spectral = conf->nchan_keep_chan_spectral;
	  conf->dtsz_network_spectral = NBYTE_FLOAT * conf->nchan_per_chunk_network_spectral;
	  conf->pktsz_network_spectral     = conf->dtsz_network_spectral + 3 * NBYTE_FLOAT + 6 * NBYTE_INT + FITS_TIME_STAMP_LEN;
	  log_add(conf->log_file, "INFO", 1,  "Spectral data will be sent with %d frequency chunks for each pol.", conf->nchunk_network_spectral);
	  log_add(conf->log_file, "INFO", 1,  "Spectral data will be sent with %d frequency channels in each frequency chunks.", conf->nchan_per_chunk_network_spectral);
	  log_add(conf->log_file, "INFO", 1,  "Size of spectral data in  each network packet is %d bytes.", conf->dtsz_network_spectral);
	  log_add(conf->log_file, "INFO", 1,  "Size of each network packet is %d bytes.", conf->pktsz_network_spectral);
	  
	  queue_fits_spectral = create_queue(10 * conf->nchunk_network_spectral * NDATA_PER_SAMP_FULL);
	}
    }

  fprintf(stdout, "BASEBAND2FILTERBANK_READY\n");  // Ready to take data from ring buffer, just before the header thing
  fflush(stdout);
  log_add(conf->log_file, "INFO", 1,  "BASEBAND2FILTERBANK_READY");
  
  if(read_dada_header(conf))
    {
      log_add(conf->log_file, "ERR", 1,  "header read failed, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: header read failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "read_dada_header done");
  
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

  //for(i = 0; i < conf.rbufin_size; i += conf.bufin_size)
  for(i = 0; i < conf.nrepeat_per_blk; i ++)
    {
      for (j = 0; j < conf.nstream; j++)
	{
	  //hbufin_offset = j * conf.hbufin_offset + i;
	  hbufin_offset = (i * conf.nstream + j) * conf.hbufin_offset;
	  dbufin_offset = j * conf.dbufin_offset; 
	  bufrt1_offset = j * conf.bufrt1_offset;
	  bufrt2_offset = j * conf.bufrt2_offset;
	  
	  /* Copy data into device */
	  CudaSafeCall(cudaMemcpyAsync(&conf.dbuf_in[dbufin_offset], &conf.cbuf_in[hbufin_offset], conf.sbufin_size, cudaMemcpyHostToDevice, conf.streams[j]));

	  /* Unpack raw data into cufftComplex array */
	  unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, conf.streams[j]>>>(&conf.dbuf_in[dbufin_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp_in);
	  CudaSafeKernelLaunch();
	  
	  /* Do forward FFT */
	  CufftSafeCall(cufftExecC2C(conf.fft_plans[j], &conf.buf_rt1[bufrt1_offset], &conf.buf_rt1[bufrt1_offset], CUFFT_FORWARD));
	  swap_select_transpose_ptf_kernel<<<gridsize_swap_select_transpose, blocksize_swap_select_transpose, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp_in, conf.nsamp_keep, conf.cufft_nx, conf.cufft_mod, conf.nchan_keep_chan, conf.nchan_keep_band, conf.nchan_edge);
	  CudaSafeKernelLaunch();
	  
	  switch (blocksize_detect_faccumulate_pad_transpose.x )
	    {
	    case 1024:
	      detect_faccumulate_pad_transpose1_kernel<1024><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp_keep, conf.naccumulate_pad);
	      break;
	    case 512:
	      detect_faccumulate_pad_transpose1_kernel< 512><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp_keep, conf.naccumulate_pad);
	      break;			     
	    case 256:
	      detect_faccumulate_pad_transpose1_kernel< 256><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp_keep, conf.naccumulate_pad);
	      break;			     
	    case 128:
	      detect_faccumulate_pad_transpose1_kernel< 128><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp_keep, conf.naccumulate_pad);
	      break;			     
	    case 64:
	      detect_faccumulate_pad_transpose1_kernel<  64><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp_keep, conf.naccumulate_pad);
	      break;			     
	    case 32:
	      detect_faccumulate_pad_transpose1_kernel<  32><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp_keep, conf.naccumulate_pad);
	      break;			     
	    case 16:
	      detect_faccumulate_pad_transpose1_kernel<  16><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp_keep, conf.naccumulate_pad);
	      break;			     
	    case 8:
	      detect_faccumulate_pad_transpose1_kernel<   8><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp_keep, conf.naccumulate_pad);
	      break;			     
	    case 4:
	      detect_faccumulate_pad_transpose1_kernel<   4><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp_keep, conf.naccumulate_pad);
	      break;			     
	    case 2:
	      detect_faccumulate_pad_transpose1_kernel<   2><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp_keep, conf.naccumulate_pad);
	      break;			     
	    case 1:
	      detect_faccumulate_pad_transpose1_kernel<   1><<<gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose.x * NBYTE_FLOAT, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp_keep, conf.naccumulate_pad);
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
  sprintf(fname, "%s/%s_baseband2filterbank.scl", conf.dir, conf.utc_start);
  fp = fopen(fname, "w");
  if(fp == NULL)
    {
      log_add(conf.log_file, "ERR", 1,  "Can not open scale file, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Can not open scale file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
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
      if(conf.fft_plans[i])
	CufftSafeCall(cufftDestroy(conf.fft_plans[i]));
    }
  if(conf.fft_plans)
    free(conf.fft_plans);
  log_add(conf.log_file, "INFO", 1,  "destroy filterbank fft plan done");

  if((conf.spectral2disk == 1) || (conf.spectral2network == 1))
    {
      if(conf.dbuf_out_spectral)
	cudaFree(conf.dbuf_out_spectral);
      if(conf.buf_rt1_spectral)
	cudaFree(conf.buf_rt1_spectral);
      if(conf.buf_rt2_spectral)
	cudaFree(conf.buf_rt2_spectral);
      for(i = 0; i < conf.nstream; i++)
	{
	  if(conf.fft_plans_spectral[i])
	    CufftSafeCall(cufftDestroy(conf.fft_plans_spectral[i]));
	}
      if(conf.fft_plans_spectral)
	free(conf.fft_plans_spectral);

      if(conf.spectral2network == 1)
	destroy_queue(*queue_fits_spectral);
    }
  if(conf.monitor)
    {
      destroy_queue(*queue_fits_monitor);
      if(conf.dbuf_out_monitor1)
	cudaFree(conf.dbuf_out_monitor1);
      if(conf.dbuf_out_monitor2)
	cudaFree(conf.dbuf_out_monitor2);
      if(conf.dbuf_out_monitor3)
	cudaFree(conf.dbuf_out_monitor3);
    }
  if(conf.buf_rt1)
    cudaFree(conf.buf_rt1);
  if(conf.buf_rt2)
    cudaFree(conf.buf_rt2);
  if(conf.dbuf_in)
    cudaFree(conf.dbuf_in);
  if(conf.dbuf_out_filterbank)
    cudaFree(conf.dbuf_out_filterbank);
  if(conf.offset_scale_h)
    cudaFreeHost(conf.offset_scale_h);
  if(conf.offset_scale_d)
    cudaFree(conf.offset_scale_d);
  log_add(conf.log_file, "INFO", 1,  "Free cuda memory done");

  if(conf.db_in)
    {
      dada_cuda_dbunregister(conf.hdu_in);
      dada_hdu_unlock_read(conf.hdu_in);
      dada_hdu_destroy(conf.hdu_in);
      fprintf(stdout, "HERE db_in free\n");
      fflush(stdout);
    }
  if(conf.db_out)
    {
      dada_cuda_dbunregister(conf.hdu_out);
      dada_hdu_unlock_write(conf.hdu_out);
      dada_hdu_destroy(conf.hdu_out);
      fprintf(stdout, "HERE db_out free\n");
      fflush(stdout);
    }  
  if(conf.spectral2disk && conf.db_out_spectral)
    {
      dada_cuda_dbunregister(conf.hdu_out_spectral);
      dada_hdu_unlock_write(conf.hdu_out_spectral);
      dada_hdu_destroy(conf.hdu_out_spectral);
      
      fprintf(stdout, "HERE db_out_spectral free\n");
      fflush(stdout);
    }  
  log_add(conf.log_file, "INFO", 1,  "destory hdu done");  

  for(i = 0; i < conf.nstream; i++)
    {
      if(conf.streams[i])
	CudaSafeCall(cudaStreamDestroy(conf.streams[i]));
    }
  if(conf.streams)
    free(conf.streams);
  log_add(conf.log_file, "INFO", 1,  "destroy stream done");
  
  CudaSafeCall(cudaProfilerStop());
  CudaSafeCall(cudaDeviceReset());
  
  return EXIT_SUCCESS;
}

int examine_record_arguments(conf_t conf, char **argv, int argc)
{
  int i;
  char command_line[MSTR_LEN] = {'\0'};
  log_add(conf.log_file, "INFO", 1,  "BASEBAND2FILTERBANK HERE");
  
  /* Log the input */
  strcpy(command_line, argv[0]);
  for(i = 1; i < argc; i++)
    {
      strcat(command_line, " ");
      strcat(command_line, argv[i]);
    }
  log_add(conf.log_file, "INFO", 1,  "The command line is \"%s\"", command_line);
  log_add(conf.log_file, "INFO", 1,  "The input ring buffer key is %x", conf.key_in); 
  log_add(conf.log_file, "INFO", 1,  "The output ring buffer key is %x", conf.key_out);

  if((conf.spectral2disk == 1) || (conf.spectral2network == 1))
    {
      if((conf.start_chunk + conf.nchunk_in_spectral) > conf.nchunk_in)
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: start_nchunk + nchunk_in_spectral % is larger than nchunk_in %d, which happens at \"%s\", line [%d], has to abort\n", conf.start_chunk + conf.nchunk_in_spectral, conf.nchunk_in, __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1,  "start_nchunk + nchunk_in_spectral % is larger than nchunk_in %d, which happens at \"%s\", line [%d], has to abort", conf.start_chunk + conf.nchunk_in_spectral, conf.nchunk_in, __FILE__, __LINE__);
      
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      else if(conf.start_chunk<0)
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: start_nchunk should not be negative, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.start_chunk, __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1,  "start_nchunk should not be negative, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.start_chunk, __FILE__, __LINE__);
      
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      else if (conf.nchunk_in_spectral<=0)	
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: nchunk_in_spectral should be positive, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.nchunk_in_spectral, __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1,  "nchunk_in_spectral should be positive, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.nchunk_in_spectral, __FILE__, __LINE__);
      
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      else
	log_add(conf.log_file, "ERR", 1,  "start_chunk is %d and nchunk_in_spectral is %d", conf.start_chunk, conf.nchunk_in_spectral);
      
      if(!((conf.ptype_spectral == 1) || (conf.ptype_spectral == 2) || (conf.ptype_spectral == 4)))
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: ptype_spectral should be 1, 2 or 4, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.ptype_spectral, __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1,  "ptype_spectral should be 1, 2 or 4, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.ptype_spectral, __FILE__, __LINE__);
      
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      else
	log_add(conf.log_file, "INFO", 1,  "ptype_spectral is %d", conf.ptype_spectral);
      
      if(conf.nblk_accumulate == -1)
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: nblk_accumulate is unset, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1,  "nblk_accumulate is unset, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      log_add(conf.log_file, "INFO", 1,  "We will average %d buffer blocks for spectral output", conf.nblk_accumulate);
      
      if(conf.cufft_nx_spectral<=0)    
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: cufft_nx_spectral shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.cufft_nx, __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1,  "cufft_nx_spectral shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.cufft_nx, __FILE__, __LINE__);
	  
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      log_add(conf.log_file, "INFO", 1,  "We use %d points FFT for spectral output", conf.cufft_nx_spectral);
      
      if(conf.spectral2disk == 1)
	{
	  log_add(conf.log_file, "INFO", 1,  "We will send spectral data with ring buffer");
	  
	  if(conf.sod_spectral == 1)
	    log_add(conf.log_file, "INFO", 1,  "The spectral data is enabled at the beginning");
	  else if(conf.sod_spectral == 0)
	    log_add(conf.log_file, "INFO", 1,  "The spectral data is NOT enabled at the beginning");
	  else
	    {
	      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: The sod should be 0 or 1 when we use ring buffer to send spectral data, but it is -1, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	      log_add(conf.log_file, "ERR", 1,  "The sod should be 0 or 1 when we use ring buffer to send spectral data, but it is -1, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	      
	      log_close(conf.log_file);
	      exit(EXIT_FAILURE);
	    }
	  log_add(conf.log_file, "INFO", 1,  "The key for the spectral ring buffer is %x", conf.key_out);
	}
      
      if(conf.spectral2network == 1)
	{
	  log_add(conf.log_file, "INFO", 1,  "We will send spectral data with network");
	  
	  if(strstr(conf.ip_spectral, "unset"))
	    {
	      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: We are going to send spectral data with network interface, but no ip is given, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	      log_add(conf.log_file, "ERR", 1,  "We are going to send spectral data with network interface, but no ip is given, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	      
	      log_close(conf.log_file);
	      exit(EXIT_FAILURE);
	    }
	  if(conf.port_spectral == -1)
	    {
	      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: We are going to send spectral data with network interface, but no port is given, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	      log_add(conf.log_file, "ERR", 1,  "We are going to send spectral data with network interface, but no port is given, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	      
	      log_close(conf.log_file);
	      exit(EXIT_FAILURE);
	    }
	  else
	    log_add(conf.log_file, "INFO", 1,  "The network interface for the spectral data is %s_%d", conf.ip_spectral, conf.port_spectral);        
	}
    }
  
  if(conf.monitor == 1)
    {   
      if(!((conf.ptype_monitor == 1) || (conf.ptype_monitor == 2) || (conf.ptype_monitor == 4)))
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: ptype_monitor should be 1, 2 or 4, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.ptype_monitor, __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1,  "ptype_monitor should be 1, 2 or 4, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.ptype_monitor, __FILE__, __LINE__);
	  
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      else
	log_add(conf.log_file, "INFO", 1,  "ptype_monitor is %d", conf.ptype_monitor);
            
      if(conf.port_monitor == -1)
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: monitor port shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.port_monitor, __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1,  "monitor port shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.port_monitor, __FILE__, __LINE__);
	  
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      if(strstr(conf.ip_monitor, "unset"))
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: monitor ip is unset, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1,  "monitor ip is unset, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      log_add(conf.log_file, "INFO", 1,  "We will send monitor data to %s:%d", conf.ip_monitor, conf.port_monitor); 
    }  
  else
    log_add(conf.log_file, "INFO", 1,  "We will not send monitor data to FITSwriter interface");
  
  if(conf.ndf_per_chunk_rbufin == 0)
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: ndf_per_chunk_rbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_rbufin, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "ndf_per_chunk_rbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort", conf.ndf_per_chunk_rbufin, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "Each input ring buffer block has %"PRIu64" packets per frequency chunk", conf.ndf_per_chunk_rbufin); 

  if(conf.nstream <= 0)
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: nstream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.nstream, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "nstream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.nstream, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "%d streams run on GPU", conf.nstream);
  
  if(conf.ndf_per_chunk_stream == 0)
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: ndf_per_chunk_stream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_stream, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "ndf_per_chunk_stream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.ndf_per_chunk_stream, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "Each stream process %d packets per frequency chunk", conf.ndf_per_chunk_stream);

  log_add(conf.log_file, "INFO", 1,  "The runtime information is %s", conf.dir);  // Checked already
  
  if(conf.sod == 1)
    log_add(conf.log_file, "INFO", 1,  "The filterbank data is enabled at the beginning");
  else if(conf.sod == 0)
    log_add(conf.log_file, "INFO", 1,  "The filterbank data is NOT enabled at the beginning");
  else
    { 
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: The SOD is not set, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "The SOD is not set, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  if(conf.nchunk_in<=0 || conf.nchunk_in>NCHUNK_FULL_BEAM)    
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: nchunk_in shoule be in (0 %d], but it is %d, which happens at \"%s\", line [%d], has to abort\n", NCHUNK_FULL_BEAM, conf.nchunk_in, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "nchunk_in shoule be in (0 %d], but it is %d, which happens at \"%s\", line [%d], has to abort", NCHUNK_FULL_BEAM, conf.nchunk_in, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }  
  log_add(conf.log_file, "INFO", 1,  "%d chunks of input data", conf.nchunk_in);

  if(conf.cufft_nx<=0)    
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: cufft_nx shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.cufft_nx, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "cufft_nx shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.cufft_nx, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "We use %d points FFT", conf.cufft_nx);
  
  if(conf.nchan_out <= 0)
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: nchan_out should be positive, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.nchan_out, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "nchan_out should be positive, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.nchan_out, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }

  if((log2((double)conf.nchan_out) - floor(log2((double)conf.nchan_out))) != 0)
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: nchan_out should be power of 2, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.nchan_out, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "nchan_out should be power of 2, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.nchan_out, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "We output %d channels", conf.nchan_out);
  
  return EXIT_SUCCESS;
}

int read_dada_header(conf_t *conf)
{  
  uint64_t hdrsz;
  
  conf->hdrbuf_in  = ipcbuf_get_next_read(conf->hdr_in, &hdrsz);  
  if (!conf->hdrbuf_in)
    {
      log_add(conf->log_file, "ERR", 1,  "Error getting header_buf, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if(hdrsz != DADA_DEFAULT_HEADER_SIZE)
    {
      log_add(conf->log_file, "ERR", 1,  "Header size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Header size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  
  if(ascii_header_get(conf->hdrbuf_in, "PICOSECONDS", "%"SCNu64"", &(conf->picoseconds)) < 0)
    {
      log_add(conf->log_file, "ERR", 1,  "Error getting PICOSECONDS, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting PICOSECONDS, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      log_close(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "PICOSECONDS from DADA header is %"PRIu64"", conf->picoseconds);
  
  if(ascii_header_get(conf->hdrbuf_in, "FREQ", "%lf", &(conf->cfreq_band)) < 0)
    {
      log_add(conf->log_file, "ERR", 1,  "Error egtting FREQ, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting FREQ, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      log_close(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "FREQ from DADA header is %f", conf->cfreq_band);
  
  if (ascii_header_get(conf->hdrbuf_in, "FILE_SIZE", "%"SCNu64"", &conf->file_size_in) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error getting FILE_SIZE, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }   
  log_add(conf->log_file, "INFO", 1,  "FILE_SIZE from DADA header is %"PRIu64"", conf->file_size_in);
  
  if (ascii_header_get(conf->hdrbuf_in, "BYTES_PER_SECOND", "%"SCNu64"", &conf->bytes_per_second_in) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "BYTES_PER_SECOND from DADA header is %"PRIu64"", conf->bytes_per_second_in);
  
  if (ascii_header_get(conf->hdrbuf_in, "TSAMP", "%lf", &conf->tsamp_in) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error getting TSAMP, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "TSAMP from DADA header is %f", conf->tsamp_in);
  
  /* Get utc_start from hdrin */
  if (ascii_header_get(conf->hdrbuf_in, "UTC_START", "%s", conf->utc_start) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error getting UTC_START, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "UTC_START from DADA header is %s", conf->utc_start);
    
  if (ascii_header_get(conf->hdrbuf_in, "RECEIVER", "%d", &conf->beam_index) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error getting RECEIVER, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting RECEIVER, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "RECEIVER from DADA header is %d", conf->beam_index);
  
  if(ipcbuf_mark_cleared (conf->hdr_in))  // We are the only one reader, so that we can clear it after read;
    {
      log_add(conf->log_file, "ERR", 1,  "Error header_clear, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error header_clear, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  return EXIT_SUCCESS;
}

int register_dada_header(conf_t *conf)
{
  char *hdrbuf_out = NULL;
  uint64_t file_size, bytes_per_second;
  
  hdrbuf_out = ipcbuf_get_next_write(conf->hdr_out);
  if (!hdrbuf_out)
    {
      log_add(conf->log_file, "ERR", 1,  "Error getting header_buf, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }  
  memcpy(hdrbuf_out, conf->hdrbuf_in, DADA_DEFAULT_HEADER_SIZE); // Pass the header
  
  file_size = (uint64_t)(conf->file_size_in * conf->scale_dtsz);
  bytes_per_second = (uint64_t)(conf->bytes_per_second_in * conf->scale_dtsz);
  
  if (ascii_header_set(hdrbuf_out, "NCHAN", "%d", conf->nchan_out) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error setting NCHAN, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting NCHAN, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "NCHAN to DADA header is %d", conf->nchan_out);

  conf->tsamp_out_filterbank = conf->tsamp_in * conf->cufft_nx;
  if (ascii_header_set(hdrbuf_out, "TSAMP", "%f", conf->tsamp_out_filterbank) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error setting TSAMP, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "TSAMP to DADA header is %f microseconds", conf->tsamp_out_filterbank);
  
  if (ascii_header_set(hdrbuf_out, "NBIT", "%d", NBIT_FILTERBANK) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting NBIT, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "NBIT to DADA header is %d", NBIT_FILTERBANK);
  
  if (ascii_header_set(hdrbuf_out, "NDIM", "%d", NDIM_FILTERBANK) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error setting NDIM, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting NDIM, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "NDIM to DADA header is %d", NDIM_FILTERBANK);
  
  if (ascii_header_set(hdrbuf_out, "NPOL", "%d", NPOL_FILTERBANK) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error setting NPOL, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting NPOL, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "NPOL to DADA header is %d", NPOL_FILTERBANK);
  
  if (ascii_header_set(hdrbuf_out, "FILE_SIZE", "%"PRIu64"", file_size) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: BASEBAND2FILTERBANK_ERROR:\tError setting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "FILE_SIZE to DADA header is %"PRIu64"", file_size);
  
  if (ascii_header_set(hdrbuf_out, "BW", "%f", -conf->bandwidth) < 0)  // Reverse frequency order
    //if (ascii_header_set(hdrbuf_out, "BW", "%f", conf->bandwidth) < 0)  // Reverse frequency order
    {
      log_add(conf->log_file, "ERR", 1,  "Error setting BW, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting BW, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "BW to DADA header is %f", -conf->bandwidth);
  
  if (ascii_header_set(hdrbuf_out, "BYTES_PER_SECOND", "%"PRIu64"", bytes_per_second) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "BYTES_PER_SECOND to DADA header is %"PRIu64"", bytes_per_second);
  
  /* donot set header parameters anymore */
  if (ipcbuf_mark_filled (conf->hdr_out, DADA_DEFAULT_HEADER_SIZE) < 0)
    {
      log_add(conf->log_file, "ERR", 1,  "Error header_fill, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }

  return EXIT_SUCCESS;
}

int register_dada_header_spectral(conf_t *conf)
{
  char *hdrbuf_out = NULL;
  uint64_t file_size, bytes_per_second;
  
  hdrbuf_out = ipcbuf_get_next_write(conf->hdr_out_spectral);
  if (!hdrbuf_out)
    {
      log_add(conf->log_file, "ERR", 1,  "Error getting header_buf, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }  
  memcpy(hdrbuf_out, conf->hdrbuf_in, DADA_DEFAULT_HEADER_SIZE); // Pass the header
  
  file_size = (uint64_t)(conf->file_size_in * conf->scale_dtsz_spectral);
  bytes_per_second = (uint64_t)(conf->bytes_per_second_in * conf->scale_dtsz_spectral);
  
  if(ascii_header_set(hdrbuf_out, "FREQ", "%.6f", conf->cfreq_spectral) < 0)
    {
      log_add(conf->log_file, "ERR", 1,  "Error setting FREQ, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "CAPTURE_ERROR: Error setting FREQ, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      log_close(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "FREQ to DADA header is %f", conf->cfreq_spectral);
  
  if (ascii_header_set(hdrbuf_out, "NCHAN", "%d", conf->nchan_out_spectral) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error setting NCHAN, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting NCHAN, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "NCHAN to DADA header is %d", conf->nchan_out_spectral);

  conf->tsamp_out_spectral = conf->tsamp_in * conf->ndf_per_chunk_rbufin * NSAMP_DF * conf->nblk_accumulate;
  if (ascii_header_set(hdrbuf_out, "TSAMP", "%f", conf->tsamp_out_spectral) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error setting TSAMP, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "TSAMP to DADA header is %f microseconds", conf->tsamp_out_spectral);
  
  if (ascii_header_set(hdrbuf_out, "NBIT", "%d", NBIT_SPECTRAL) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting NBIT, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "NBIT to DADA header is %d", NBIT_SPECTRAL);
  
  if (ascii_header_set(hdrbuf_out, "NDIM", "%d", conf->ndim_spectral) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error setting NDIM, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting NDIM, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "NDIM to DADA header is %d", conf->ndim_spectral);
  
  if (ascii_header_set(hdrbuf_out, "NPOL", "%d", conf->npol_spectral) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error setting NPOL, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting NPOL, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "NPOL to DADA header is %d", conf->npol_spectral);
  
  if (ascii_header_set(hdrbuf_out, "FILE_SIZE", "%"PRIu64"", file_size) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: BASEBAND2FILTERBANK_ERROR:\tError setting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "FILE_SIZE to DADA header is %"PRIu64"", file_size);
  
  if (ascii_header_set(hdrbuf_out, "BW", "%f", (double)conf->nchan_in_spectral) < 0)  // Reverse frequency order
    {
      log_add(conf->log_file, "ERR", 1,  "Error setting BW, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting BW, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "BW to DADA header is %f", (double)conf->nchan_in_spectral);
  
  if (ascii_header_set(hdrbuf_out, "BYTES_PER_SECOND", "%"PRIu64"", bytes_per_second) < 0)  
    {
      log_add(conf->log_file, "ERR", 1,  "Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1,  "BYTES_PER_SECOND to DADA header is %"PRIu64"", bytes_per_second);
  
  /* donot set header parameters anymore */
  if (ipcbuf_mark_filled (conf->hdr_out_spectral, DADA_DEFAULT_HEADER_SIZE) < 0)
    {
      log_add(conf->log_file, "ERR", 1,  "Error header_fill, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2filterbank(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }

  return EXIT_SUCCESS;
}


void *spectral_sendto(void *conf)
{
  conf_t *baseband2filterbank_conf = (conf_t *)conf;
  double sendto_period;
  unsigned int sleep_time;
  fits_t fits;
  int enable = 1, sock;
  struct sockaddr_in sa;
  socklen_t tolen = sizeof(sa);
  
  if((sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: socket creation failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      log_add(baseband2filterbank_conf->log_file, "ERR", 1,  "socket creation failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      quit = 2;
      pthread_exit(NULL);
    }
  memset((char *) &sa, 0, sizeof(sa));
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(baseband2filterbank_conf->port_spectral);
  sa.sin_addr.s_addr = inet_addr(baseband2filterbank_conf->ip_spectral);
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
  
  sendto_period = baseband2filterbank_conf->tsamp_in *
    baseband2filterbank_conf->ndf_per_chunk_rbufin *
    NSAMP_DF * baseband2filterbank_conf->nblk_accumulate;
  sleep_time = (unsigned int)(0.90 * sendto_period / (baseband2filterbank_conf->nchunk_network_spectral * NDATA_PER_SAMP_FULL)); // To be safe, do not use 100% cycle, in microseconds
  fprintf(stdout, "baseband2filterbank_conf->tsamp_in is %f microseconds, sendto_period is %f microseconds and sleep_time is %d microseconds [spectral]\n", baseband2filterbank_conf->tsamp_in, sendto_period, sleep_time);
  log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "sendto_period is %f microseconds and sleep_time is %d microseconds, [spectral].", sendto_period, sleep_time);

  int index = 0;
  while((!quit) || (!is_empty(queue_fits_spectral)))
    {      
      while((!quit) && (is_empty(queue_fits_spectral))) // Wait until we get data or quit if error
	usleep(sleep_time);
      
      //fprintf(stdout, "HERE sending data for spectral, %d\n", index);
      //fflush(stdout);
      index ++;
      if(dequeue(queue_fits_spectral, &fits))
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: queue is empty on spectral,  which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  log_add(baseband2filterbank_conf->log_file, "ERR", 1,  "queue is empty on spectral,  which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	  close(sock);
	  pthread_exit(NULL);
	  quit = 2;
	}
      if(fits.nchan != 0) // Rough check data is there
	{
	  if(sendto(sock,
		    (void *)&fits,
		    baseband2filterbank_conf->pktsz_network_spectral,
		    0,
		    (struct sockaddr *)&sa,
		    tolen) == -1)
	    {
	      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: sendto() failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      log_add(baseband2filterbank_conf->log_file, "ERR", 1,  "sendto() failed, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	      
	      close(sock);
	      quit = 2;
	      pthread_exit(NULL);
	    }
	  usleep(sleep_time);
	}
      else
	{
	  fprintf(stdout, "We got a bad spectral packet\n");
	  fflush(stdout);
	  log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "One bad spectral packet", __FILE__, __LINE__);
	}
    }
    
  close(sock);
  quit = 1;
  pthread_exit(NULL);
}

void *monitor_sendto(void *conf)
{
  conf_t *baseband2filterbank_conf = (conf_t *)conf;
  double sendto_period;
  unsigned int sleep_time;
  fits_t fits;
  int enable = 1, sock;
  struct sockaddr_in sa;
  socklen_t tolen = sizeof(sa);
  
  if((sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: socket creation failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      log_add(baseband2filterbank_conf->log_file, "ERR", 1,  "socket creation failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      quit = 2;
      pthread_exit(NULL);
    }
  memset((char *) &sa, 0, sizeof(sa));
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(baseband2filterbank_conf->port_monitor);
  sa.sin_addr.s_addr = inet_addr(baseband2filterbank_conf->ip_monitor);
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
  
  sendto_period = baseband2filterbank_conf->tsamp_in *
    baseband2filterbank_conf->ndf_per_chunk_rbufin *
    NSAMP_DF;
  sleep_time = (unsigned int)(0.90 * sendto_period / (baseband2filterbank_conf->neth_per_blk)); // To be safe, do not use 100% cycle, in microseconds
  fprintf(stdout, "sendto_period is %f microseconds, sleep_time is %d microseconds, [monitor]\n", sendto_period, sleep_time);
  log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "sendto_period is %f microseconds and sleep_time is %d microseconds, [monitor].", sendto_period, sleep_time);

  int index = 0;
  while((!quit) || (!is_empty(queue_fits_monitor)))
    {
      while((!quit) && (is_empty(queue_fits_monitor))) // Wait until we get data or quit if error
	usleep(sleep_time);
      
      if(dequeue(queue_fits_monitor, &fits))
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: queue is empty on monitor,  which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  log_add(baseband2filterbank_conf->log_file, "ERR", 1,  "queue is empty on monitor,  which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	  close(sock);
	  pthread_exit(NULL);
	  quit = 2;
	}
      //fprintf(stdout, "HERE sending data for monitor, %d\n", index);
      //fflush(stdout);
      index++;
      if(fits.nchan != 0) // Rough check data is there
	{
	  if(sendto(sock,
		    (void *)&fits,
		    baseband2filterbank_conf->pktsz_network,
		    0,
		    (struct sockaddr *)&sa,
		    tolen) == -1)
	    {
	      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: sendto() failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      log_add(baseband2filterbank_conf->log_file, "ERR", 1,  "sendto() failed, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	      
	      close(sock);
	      quit = 2;
	      pthread_exit(NULL);
	    }
	  usleep(sleep_time);
	}
      else
	{
	  fprintf(stdout, "We got a bad monitor packet\n");
	  fflush(stdout);
	  log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "One bad monitor packet.", __FILE__, __LINE__);
	}
    }
  close(sock);
  quit = 1;
  pthread_exit(NULL);
}

int threads(conf_t conf)
{  
  int i, ret[3], nthread = 0;
  pthread_t thread[3];
  
  ret[0] = pthread_create(&thread[0], NULL, do_baseband2filterbank, (void *)&conf);
  nthread ++;
  if(conf.monitor == 1)
    {      
      ret[1] = pthread_create(&thread[1], NULL, monitor_sendto, (void *)&conf);
      nthread ++;
    }
  if(conf.spectral2network == 1)
    {      
      ret[2] = pthread_create(&thread[2], NULL, spectral_sendto, (void *)&conf);
      nthread ++;
    }
  
  for(i = 0; i < nthread; i++)   // Join threads and unbind cpus
    pthread_join(thread[i], NULL);
  
  log_add(conf.log_file, "INFO", 1,  "Join threads? The last quit is %d", quit);
  
  return EXIT_SUCCESS;
}

void *do_baseband2filterbank(void *conf)
{
  conf_t *baseband2filterbank_conf = (conf_t *)conf;
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
  uint64_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset, hbufout_offset_filterbank, dbufout_offset_filterbank, dbufout_offset_monitor1, dbufout_offset_monitor2, dbufout_offset_monitor3, bufrt1_offset_spectral, bufrt2_offset_spectral, dbufout_offset_spectral;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_transpose, blocksize_transpose;
  dim3 gridsize_swap_select_transpose, blocksize_swap_select_transpose;
  dim3 gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale;
  dim3 gridsize_taccumulate, blocksize_taccumulate;
  dim3 gridsize_spectral_taccumulate, blocksize_spectral_taccumulate;
  dim3 gridsize_saccumulate, blocksize_saccumulate;
  dim3 gridsize_swap_select_transpose_pft1, blocksize_swap_select_transpose_pft1;
  
  uint64_t cbufsz;
  int first = 1;
  double chan_width_monitor; 
  double time_res_blk, time_offset = 0;
  double time_res_monitor, time_res_spectral;
  struct tm tm_stamp;
  char time_stamp_monitor[MSTR_LEN];
  char time_stamp_spectral[MSTR_LEN];
  double time_stamp_monitor_f;
  time_t time_stamp_monitor_i;
  double time_stamp_spectral_f;
  time_t time_stamp_spectral_i;
  uint64_t memcpy_offset;
  int nblk_accumulate = 0;
  int eth_index;
  fits_t *fits_monitor, fits_spectral;
  
  gridsize_unpack                      = baseband2filterbank_conf->gridsize_unpack;
  blocksize_unpack                     = baseband2filterbank_conf->blocksize_unpack;
  gridsize_taccumulate                 = baseband2filterbank_conf->gridsize_taccumulate;
  blocksize_taccumulate                = baseband2filterbank_conf->blocksize_taccumulate;
  gridsize_transpose                   = baseband2filterbank_conf->gridsize_transpose;
  blocksize_transpose                  = baseband2filterbank_conf->blocksize_transpose;
  gridsize_detect_faccumulate_scale    = baseband2filterbank_conf->gridsize_detect_faccumulate_scale ;
  blocksize_detect_faccumulate_scale   = baseband2filterbank_conf->blocksize_detect_faccumulate_scale ;
  gridsize_swap_select_transpose       = baseband2filterbank_conf->gridsize_swap_select_transpose;   
  blocksize_swap_select_transpose      = baseband2filterbank_conf->blocksize_swap_select_transpose;
  gridsize_spectral_taccumulate        = baseband2filterbank_conf->gridsize_spectral_taccumulate;
  blocksize_spectral_taccumulate       = baseband2filterbank_conf->blocksize_spectral_taccumulate;
  gridsize_saccumulate                 = baseband2filterbank_conf->gridsize_saccumulate;
  blocksize_saccumulate                = baseband2filterbank_conf->blocksize_saccumulate;
  gridsize_swap_select_transpose_pft1  = baseband2filterbank_conf->gridsize_swap_select_transpose_pft1;
  blocksize_swap_select_transpose_pft1 = baseband2filterbank_conf->blocksize_swap_select_transpose_pft1;
  
  gridsize_saccumulate        = baseband2filterbank_conf->gridsize_saccumulate; 
  blocksize_saccumulate       = baseband2filterbank_conf->blocksize_saccumulate;
  
  time_res_blk = baseband2filterbank_conf->tsamp_in * baseband2filterbank_conf->ndf_per_chunk_rbufin * NSAMP_DF / 1.0E6; // This has to be after read_register_header, in seconds
  if(baseband2filterbank_conf->monitor == 1)
    {
      time_res_monitor = baseband2filterbank_conf->tsamp_in * baseband2filterbank_conf->ndf_per_chunk_stream * NSAMP_DF / 1.0E6; // This has to be after read_register_header, in seconds
      strptime(baseband2filterbank_conf->utc_start, DADA_TIMESTR, &tm_stamp);
      time_stamp_monitor_f = mktime(&tm_stamp) + baseband2filterbank_conf->picoseconds / 1.0E12 + 0.5 * time_res_monitor;
      chan_width_monitor = baseband2filterbank_conf->bandwidth/baseband2filterbank_conf->nchan_out;
      
      fits_monitor = (fits_t *)malloc(baseband2filterbank_conf->neth_per_blk * sizeof(fits_t));
      for(i = 0; i < baseband2filterbank_conf->neth_per_blk; i++)
	cudaHostRegister ((void *) fits_monitor[i].data, sizeof(fits_monitor[i].data), 0);
    }  
  if(baseband2filterbank_conf->sod == 1)
    {
      if(register_dada_header(baseband2filterbank_conf))
	{
	  log_add(baseband2filterbank_conf->log_file, "ERR", 1,  "header register failed, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

	  quit = 2;
	  if(baseband2filterbank_conf->monitor == 1)
	    {	     
	      for(i = 0; i < baseband2filterbank_conf->neth_per_blk; i++)
		cudaHostUnregister((void *) fits_monitor[i].data);
	      free(fits_monitor);
	    }
	  if(baseband2filterbank_conf->spectral2network == 1)
	    cudaHostUnregister ((void *) fits_spectral.data);
	  pthread_exit(NULL);
	}
      log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "register_dada_header done");
    }

  if((baseband2filterbank_conf->spectral2disk == 1) || (baseband2filterbank_conf->spectral2network == 1))
    {
      //CudaSafeCall(cudaMemset((void *)baseband2filterbank_conf->dbuf_out_spectral, 0, sizeof(baseband2filterbank_conf->dbuf_out_spectral)));// We have to clear the memory for this parameter
      CudaSafeCall(cudaMemset((void *)baseband2filterbank_conf->dbuf_out_spectral, 0, baseband2filterbank_conf->bufout_size_spectral));// We have to clear the memory for this parameter
      baseband2filterbank_conf->cfreq_spectral = baseband2filterbank_conf->cfreq_band + (baseband2filterbank_conf->start_chunk + 0.5 * baseband2filterbank_conf->nchunk_in_spectral - 0.5 * baseband2filterbank_conf->nchunk_in) * NCHAN_PER_CHUNK;
      if((baseband2filterbank_conf->spectral2disk == 1) && (baseband2filterbank_conf->sod_spectral == 1))
	{
	  if(register_dada_header_spectral(baseband2filterbank_conf))
	    {
	      log_add(baseband2filterbank_conf->log_file, "ERR", 1,  "header register failed, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      quit = 2;
	      if(baseband2filterbank_conf->monitor == 1)
		{	     
		  for(i = 0; i < baseband2filterbank_conf->neth_per_blk; i++)
		    cudaHostUnregister((void *) fits_monitor[i].data);
		  free(fits_monitor);
		}
	      pthread_exit(NULL);
	    }
	}
      if(baseband2filterbank_conf->spectral2network == 1)
	{
	  cudaHostRegister ((void *) fits_spectral.data, sizeof(fits_spectral.data), 0);
	  strptime(baseband2filterbank_conf->utc_start, DADA_TIMESTR, &tm_stamp);
	  time_res_spectral = time_res_blk * baseband2filterbank_conf->nblk_accumulate;
	  time_stamp_spectral_f = mktime(&tm_stamp) + baseband2filterbank_conf->picoseconds / 1.0E12 + 0.5 * time_res_spectral;
	}
    }
  
  while(!ipcbuf_eod(baseband2filterbank_conf->db_in) && (!quit))
    {
      log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "before getting new buffer block");
      baseband2filterbank_conf->cbuf_in  = ipcbuf_get_next_read(baseband2filterbank_conf->db_in, &cbufsz);
      baseband2filterbank_conf->cbuf_out = ipcbuf_get_next_write(baseband2filterbank_conf->db_out);
      log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "after getting new buffer block");
      
      if(baseband2filterbank_conf->spectral2disk == 1)
	{
	  baseband2filterbank_conf->cbuf_out_spectral = ipcbuf_get_next_write(baseband2filterbank_conf->db_out_spectral);
	  log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "after getting new buffer block");
	}
      /* Get scale of data */
      if(first)
      	{
      	  first = 0;
      	  offset_scale(*baseband2filterbank_conf);
	  log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "offset_scale done");  // I may need to put this part before while and make the first output buffer block empty
	}

      for(i = 0; i < baseband2filterbank_conf->nrepeat_per_blk; i ++)
	{
	  for(j = 0; j < baseband2filterbank_conf->nstream; j++)
	    {
	      hbufin_offset = (i * baseband2filterbank_conf->nstream + j) * baseband2filterbank_conf->hbufin_offset;// + i * baseband2filterbank_conf->bufin_size;
	      dbufin_offset = j * baseband2filterbank_conf->dbufin_offset; 
	      bufrt1_offset = j * baseband2filterbank_conf->bufrt1_offset;
	      bufrt2_offset = j * baseband2filterbank_conf->bufrt2_offset;
	      dbufout_offset_filterbank = j * baseband2filterbank_conf->dbufout_offset_filterbank;
	      dbufout_offset_monitor1 = j * baseband2filterbank_conf->dbufout_offset_monitor1;
	      dbufout_offset_monitor2 = j * baseband2filterbank_conf->dbufout_offset_monitor2;
	      dbufout_offset_monitor3 = j * baseband2filterbank_conf->dbufout_offset_monitor3;
	      hbufout_offset_filterbank = (i * baseband2filterbank_conf->nstream + j) * baseband2filterbank_conf->hbufout_offset_filterbank;// + i * baseband2filterbank_conf->bufout_size_filterbank;
	      
	      /* Copy data into device */
	      CudaSafeCall(cudaMemcpyAsync(&baseband2filterbank_conf->dbuf_in[dbufin_offset], &baseband2filterbank_conf->cbuf_in[hbufin_offset], baseband2filterbank_conf->sbufin_size, cudaMemcpyHostToDevice, baseband2filterbank_conf->streams[j]));

	      /* Unpack raw data into cufftComplex array */
	      if((baseband2filterbank_conf->spectral2network == 1) || (baseband2filterbank_conf->spectral2disk == 1))
		{
		  bufrt1_offset_spectral = j * baseband2filterbank_conf->bufrt1_offset_spectral;
		  dbufout_offset_spectral = j * baseband2filterbank_conf->dbufout_offset_spectral;
		  bufrt2_offset_spectral = j * baseband2filterbank_conf->bufrt2_offset_spectral;
		  
		  unpack1_kernel<<<gridsize_unpack, blocksize_unpack, 0, baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->dbuf_in[dbufin_offset], &baseband2filterbank_conf->buf_rt1[bufrt1_offset], baseband2filterbank_conf->nsamp_in, &baseband2filterbank_conf->buf_rt1_spectral[bufrt1_offset_spectral], baseband2filterbank_conf->nsamp_in_spectral, baseband2filterbank_conf->start_chunk, baseband2filterbank_conf->nchunk_in_spectral);
		  CudaSafeKernelLaunch();
		  
		  /* Do forward FFT */
		  CufftSafeCall(cufftExecC2C(baseband2filterbank_conf->fft_plans_spectral[j],
					     &baseband2filterbank_conf->buf_rt1_spectral[bufrt1_offset_spectral],
					     &baseband2filterbank_conf->buf_rt1_spectral[bufrt1_offset_spectral],
					     CUFFT_FORWARD));
		  
		  /* from PFTF order to PFT order, also remove channel edge */
		  swap_select_transpose_pft1_kernel
		    <<<gridsize_swap_select_transpose_pft1,
		    blocksize_swap_select_transpose_pft1,
		    0,
		    baseband2filterbank_conf->streams[j]>>>
		    (&baseband2filterbank_conf->buf_rt1_spectral[bufrt1_offset_spectral],
		     &baseband2filterbank_conf->buf_rt2_spectral[bufrt2_offset_spectral],
		     baseband2filterbank_conf->cufft_nx_spectral,
		     baseband2filterbank_conf->ndf_per_chunk_stream * NSAMP_DF / baseband2filterbank_conf->cufft_nx_spectral,
		     baseband2filterbank_conf->nsamp_in_spectral,
		     baseband2filterbank_conf->nsamp_keep_spectral,
		     baseband2filterbank_conf->cufft_nx_spectral,
		     baseband2filterbank_conf->cufft_mod_spectral,
		     baseband2filterbank_conf->nchan_keep_chan_spectral);
		  CudaSafeKernelLaunch();
		  
		  /* Convert to required pol and accumulate in time */
		  switch(blocksize_spectral_taccumulate.x)
		    {
		    case 1024:
		      spectral_taccumulate_kernel
			<1024>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2_spectral[bufrt2_offset_spectral],
			 &baseband2filterbank_conf->dbuf_out_spectral[dbufout_offset_spectral],
			 baseband2filterbank_conf->nsamp_keep_spectral,
			 baseband2filterbank_conf->nsamp_out_spectral,
			 baseband2filterbank_conf->naccumulate_spectral);
		      break;
		      
		    case 512:
		      spectral_taccumulate_kernel
			< 512>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2_spectral[bufrt2_offset_spectral],
			 &baseband2filterbank_conf->dbuf_out_spectral[dbufout_offset_spectral],
			 baseband2filterbank_conf->nsamp_keep_spectral,
			 baseband2filterbank_conf->nsamp_out_spectral,
			 baseband2filterbank_conf->naccumulate_spectral);
		      break;
		      
		    case 256:
		      spectral_taccumulate_kernel
			< 256>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2_spectral[bufrt2_offset_spectral],
			 &baseband2filterbank_conf->dbuf_out_spectral[dbufout_offset_spectral],
			 baseband2filterbank_conf->nsamp_keep_spectral,
			 baseband2filterbank_conf->nsamp_out_spectral,
			 baseband2filterbank_conf->naccumulate_spectral);
		      break;
		      
		    case 128:
		      spectral_taccumulate_kernel
			< 128>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2_spectral[bufrt2_offset_spectral],
			 &baseband2filterbank_conf->dbuf_out_spectral[dbufout_offset_spectral],
			 baseband2filterbank_conf->nsamp_keep_spectral,
			 baseband2filterbank_conf->nsamp_out_spectral,
			 baseband2filterbank_conf->naccumulate_spectral);
		      break;
		      
		    case  64:
		      spectral_taccumulate_kernel
			<  64>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2_spectral[bufrt2_offset_spectral],
			 &baseband2filterbank_conf->dbuf_out_spectral[dbufout_offset_spectral],
			 baseband2filterbank_conf->nsamp_keep_spectral,
			 baseband2filterbank_conf->nsamp_out_spectral,
			 baseband2filterbank_conf->naccumulate_spectral);
		      break;
		      
		    case  32:
		      spectral_taccumulate_kernel
			<  32>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2_spectral[bufrt2_offset_spectral],
			 &baseband2filterbank_conf->dbuf_out_spectral[dbufout_offset_spectral],
			 baseband2filterbank_conf->nsamp_keep_spectral,
			 baseband2filterbank_conf->nsamp_out_spectral,
			 baseband2filterbank_conf->naccumulate_spectral);
		      break;
		      
		    case  16:
		      spectral_taccumulate_kernel
			<  16>	
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2_spectral[bufrt2_offset_spectral],
			 &baseband2filterbank_conf->dbuf_out_spectral[dbufout_offset_spectral],
			 baseband2filterbank_conf->nsamp_keep_spectral,
			 baseband2filterbank_conf->nsamp_out_spectral,
			 baseband2filterbank_conf->naccumulate_spectral);
		      break;
		      
		    case  8:
		      spectral_taccumulate_kernel
			<   8>	
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2_spectral[bufrt2_offset_spectral],
			 &baseband2filterbank_conf->dbuf_out_spectral[dbufout_offset_spectral],
			 baseband2filterbank_conf->nsamp_keep_spectral,
			 baseband2filterbank_conf->nsamp_out_spectral,
			 baseband2filterbank_conf->naccumulate_spectral);
		      break;
		      
		    case  4:
		      spectral_taccumulate_kernel
			<   4>	
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2_spectral[bufrt2_offset_spectral],
			 &baseband2filterbank_conf->dbuf_out_spectral[dbufout_offset_spectral],
			 baseband2filterbank_conf->nsamp_keep_spectral,
			 baseband2filterbank_conf->nsamp_out_spectral,
			 baseband2filterbank_conf->naccumulate_spectral);
		      break;
		      
		    case  2:
		      spectral_taccumulate_kernel
			<   2>	
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2_spectral[bufrt2_offset_spectral],
			 &baseband2filterbank_conf->dbuf_out_spectral[dbufout_offset_spectral],
			 baseband2filterbank_conf->nsamp_keep_spectral,
			 baseband2filterbank_conf->nsamp_out_spectral,
			 baseband2filterbank_conf->naccumulate_spectral);
		      break;
		      
		    case  1:
		      spectral_taccumulate_kernel
			<   1>	
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2_spectral[bufrt2_offset_spectral],
			 &baseband2filterbank_conf->dbuf_out_spectral[dbufout_offset_spectral],
			 baseband2filterbank_conf->nsamp_keep_spectral,
			 baseband2filterbank_conf->nsamp_out_spectral,
			 baseband2filterbank_conf->naccumulate_spectral);
		      break;
		    }
		  CudaSafeKernelLaunch();
		}
	      else
		unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->dbuf_in[dbufin_offset], &baseband2filterbank_conf->buf_rt1[bufrt1_offset], baseband2filterbank_conf->nsamp_in);
	      CudaSafeKernelLaunch();
	      
	      /* Do forward FFT */
	      CufftSafeCall(cufftExecC2C(baseband2filterbank_conf->fft_plans[j], &baseband2filterbank_conf->buf_rt1[bufrt1_offset], &baseband2filterbank_conf->buf_rt1[bufrt1_offset], CUFFT_FORWARD));

	      swap_select_transpose_ptf_kernel<<<gridsize_swap_select_transpose, blocksize_swap_select_transpose, 0, baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->buf_rt1[bufrt1_offset], &baseband2filterbank_conf->buf_rt2[bufrt2_offset], baseband2filterbank_conf->nsamp_in, baseband2filterbank_conf->nsamp_keep, baseband2filterbank_conf->cufft_nx, baseband2filterbank_conf->cufft_mod, baseband2filterbank_conf->nchan_keep_chan, baseband2filterbank_conf->nchan_keep_band, baseband2filterbank_conf->nchan_edge);
	      CudaSafeKernelLaunch();

	      if(baseband2filterbank_conf->monitor == 1)
		{
		  switch (blocksize_detect_faccumulate_scale.x)
		    {
		    case 1024:
		      detect_faccumulate_scale2_spectral_faccumulate_kernel
		    <1024>
			<<<gridsize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2[bufrt2_offset],
			 &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank],
			 &baseband2filterbank_conf->dbuf_out_monitor1[dbufout_offset_monitor1],
			 baseband2filterbank_conf->nsamp_keep,
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->naccumulate_scale,
			 baseband2filterbank_conf->offset_scale_d);
		      break;
		      
		    case 512:
		      detect_faccumulate_scale2_spectral_faccumulate_kernel
			< 512>
			<<<gridsize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2[bufrt2_offset],
			 &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank],
			 &baseband2filterbank_conf->dbuf_out_monitor1[dbufout_offset_monitor1],
			 baseband2filterbank_conf->nsamp_keep,
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->naccumulate_scale,
			 baseband2filterbank_conf->offset_scale_d);
		      break;
		      
		    case 256:
		      detect_faccumulate_scale2_spectral_faccumulate_kernel
			< 256>
			<<<gridsize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2[bufrt2_offset],
			 &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank],
			 &baseband2filterbank_conf->dbuf_out_monitor1[dbufout_offset_monitor1],
			 baseband2filterbank_conf->nsamp_keep,
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->naccumulate_scale,
			 baseband2filterbank_conf->offset_scale_d);
		      break;
		      
		    case 128:
		      detect_faccumulate_scale2_spectral_faccumulate_kernel
			< 128>
			<<<gridsize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2[bufrt2_offset],
			 &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank],
			 &baseband2filterbank_conf->dbuf_out_monitor1[dbufout_offset_monitor1],
			 baseband2filterbank_conf->nsamp_keep,
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->naccumulate_scale,
			 baseband2filterbank_conf->offset_scale_d);
		      break;																															                                  
		    case 64:
		      detect_faccumulate_scale2_spectral_faccumulate_kernel
			<  64>
			<<<gridsize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2[bufrt2_offset],
			 &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank],
			 &baseband2filterbank_conf->dbuf_out_monitor1[dbufout_offset_monitor1],
			 baseband2filterbank_conf->nsamp_keep,
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->naccumulate_scale,
			 baseband2filterbank_conf->offset_scale_d);
		      break;																															                                  
		    case 32:
		      detect_faccumulate_scale2_spectral_faccumulate_kernel
			<  32>
			<<<gridsize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2[bufrt2_offset],
			 &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank],
			 &baseband2filterbank_conf->dbuf_out_monitor1[dbufout_offset_monitor1],
			 baseband2filterbank_conf->nsamp_keep,
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->naccumulate_scale,
			 baseband2filterbank_conf->offset_scale_d);
		      break;
		      
		    case 16:
		      detect_faccumulate_scale2_spectral_faccumulate_kernel
			<  16>
			<<<gridsize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2[bufrt2_offset],
			 &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank],
			 &baseband2filterbank_conf->dbuf_out_monitor1[dbufout_offset_monitor1],
			 baseband2filterbank_conf->nsamp_keep,
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->naccumulate_scale,
			 baseband2filterbank_conf->offset_scale_d);
		      break;
		      
		    case 8:
		      detect_faccumulate_scale2_spectral_faccumulate_kernel
			<   8>
			<<<gridsize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2[bufrt2_offset],
			 &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank],
			 &baseband2filterbank_conf->dbuf_out_monitor1[dbufout_offset_monitor1],
			 baseband2filterbank_conf->nsamp_keep,
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->naccumulate_scale,
			 baseband2filterbank_conf->offset_scale_d);
		      break;
		      
		    case 4:
		      detect_faccumulate_scale2_spectral_faccumulate_kernel
			<   4>
			<<<gridsize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2[bufrt2_offset],
			 &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank],
			 &baseband2filterbank_conf->dbuf_out_monitor1[dbufout_offset_monitor1],
			 baseband2filterbank_conf->nsamp_keep,
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->naccumulate_scale,
			 baseband2filterbank_conf->offset_scale_d);
		      break;
		      
		    case 2:
		      detect_faccumulate_scale2_spectral_faccumulate_kernel
			<   2>
			<<<gridsize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2[bufrt2_offset],
			 &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank],
			 &baseband2filterbank_conf->dbuf_out_monitor1[dbufout_offset_monitor1],
			 baseband2filterbank_conf->nsamp_keep,
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->naccumulate_scale,
			 baseband2filterbank_conf->offset_scale_d);
		      break;
		      
		    case 1:
		      detect_faccumulate_scale2_spectral_faccumulate_kernel
			<   1>
			<<<gridsize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale,
			blocksize_detect_faccumulate_scale.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->buf_rt2[bufrt2_offset],
			 &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank],
			 &baseband2filterbank_conf->dbuf_out_monitor1[dbufout_offset_monitor1],
			 baseband2filterbank_conf->nsamp_keep,
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->naccumulate_scale,
			 baseband2filterbank_conf->offset_scale_d);
		      break;
		    }
		  CudaSafeKernelLaunch();
		  
		  /* Further process for monitor information */
		  transpose_kernel<<<gridsize_transpose, blocksize_transpose, 0, baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->dbuf_out_monitor1[dbufout_offset_monitor1], &baseband2filterbank_conf->dbuf_out_monitor2[dbufout_offset_monitor2], baseband2filterbank_conf->nsamp_filterbank, baseband2filterbank_conf->n_transpose, baseband2filterbank_conf->m_transpose);
		  CudaSafeKernelLaunch();
		  
		  switch (blocksize_taccumulate.x)
		    {
		    case 1024:
		      accumulate_float_kernel
			<1024>
			<<<gridsize_taccumulate,
			blocksize_taccumulate,
			blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->dbuf_out_monitor2[dbufout_offset_monitor2],
			 &baseband2filterbank_conf->dbuf_out_monitor3[dbufout_offset_monitor3],
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->nchan_out,
			 baseband2filterbank_conf->naccumulate);
		      break;
		      
		    case 512:
		      accumulate_float_kernel
			< 512>
			<<<gridsize_taccumulate,
			blocksize_taccumulate,
			blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->dbuf_out_monitor2[dbufout_offset_monitor2],
			 &baseband2filterbank_conf->dbuf_out_monitor3[dbufout_offset_monitor3],
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->nchan_out,
			 baseband2filterbank_conf->naccumulate);
		      break;
		      
		    case 256:
		      accumulate_float_kernel
			< 256>
			<<<gridsize_taccumulate,
			blocksize_taccumulate,
			blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->dbuf_out_monitor2[dbufout_offset_monitor2],
			 &baseband2filterbank_conf->dbuf_out_monitor3[dbufout_offset_monitor3],
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->nchan_out,
			 baseband2filterbank_conf->naccumulate);
		      break;
		      
		    case 128:
		      accumulate_float_kernel
			< 128>
			<<<gridsize_taccumulate,
			blocksize_taccumulate,
			blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->dbuf_out_monitor2[dbufout_offset_monitor2],
			 &baseband2filterbank_conf->dbuf_out_monitor3[dbufout_offset_monitor3],
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->nchan_out,
			 baseband2filterbank_conf->naccumulate);
		      break;
		      
		    case 64:
		      accumulate_float_kernel
			<  64>
			<<<gridsize_taccumulate,
			blocksize_taccumulate,
			blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->dbuf_out_monitor2[dbufout_offset_monitor2],
			 &baseband2filterbank_conf->dbuf_out_monitor3[dbufout_offset_monitor3],
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->nchan_out,
			 baseband2filterbank_conf->naccumulate);
		      break;
		      
		    case 32:
		      accumulate_float_kernel
			<  32>
			<<<gridsize_taccumulate,
			blocksize_taccumulate,
			blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->dbuf_out_monitor2[dbufout_offset_monitor2],
			 &baseband2filterbank_conf->dbuf_out_monitor3[dbufout_offset_monitor3],
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->nchan_out,
			 baseband2filterbank_conf->naccumulate);
		      break;
		      
		    case 16:
		      accumulate_float_kernel
			<  16>
			<<<gridsize_taccumulate,
			blocksize_taccumulate,
			blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->dbuf_out_monitor2[dbufout_offset_monitor2],
			 &baseband2filterbank_conf->dbuf_out_monitor3[dbufout_offset_monitor3],
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->nchan_out,
			 baseband2filterbank_conf->naccumulate);
		      break;
		      
		    case 8:
		      accumulate_float_kernel
			<   8>
			<<<gridsize_taccumulate,
			blocksize_taccumulate,
			blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->dbuf_out_monitor2[dbufout_offset_monitor2],
			 &baseband2filterbank_conf->dbuf_out_monitor3[dbufout_offset_monitor3],
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->nchan_out,
			 baseband2filterbank_conf->naccumulate);
		      break;
		      
		    case 4:
		      accumulate_float_kernel
			<   4>
			<<<gridsize_taccumulate,
			blocksize_taccumulate,
			blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->dbuf_out_monitor2[dbufout_offset_monitor2],
			 &baseband2filterbank_conf->dbuf_out_monitor3[dbufout_offset_monitor3],
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->nchan_out,
			 baseband2filterbank_conf->naccumulate);
		      break;
		      
		    case 2:
		      accumulate_float_kernel
			<   2>
			<<<gridsize_taccumulate,
			blocksize_taccumulate,
			blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->dbuf_out_monitor2[dbufout_offset_monitor2],
			 &baseband2filterbank_conf->dbuf_out_monitor3[dbufout_offset_monitor3],
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->nchan_out,
			 baseband2filterbank_conf->naccumulate);
		      break;
		      
		    case 1:
		      accumulate_float_kernel
			<   1>
			<<<gridsize_taccumulate,
			blocksize_taccumulate,
			blocksize_taccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			baseband2filterbank_conf->streams[j]>>>
			(&baseband2filterbank_conf->dbuf_out_monitor2[dbufout_offset_monitor2],
			 &baseband2filterbank_conf->dbuf_out_monitor3[dbufout_offset_monitor3],
			 baseband2filterbank_conf->nsamp_filterbank,
			 baseband2filterbank_conf->nchan_out,
			 baseband2filterbank_conf->naccumulate);
		      break;
		    }
		  CudaSafeKernelLaunch();
		  
		  /* Setup ethernet packets */
		  time_stamp_monitor_i = (time_t)time_stamp_monitor_f;
		  strftime(time_stamp_monitor, FITS_TIME_STAMP_LEN, FITS_TIMESTR, gmtime(&time_stamp_monitor_i)); 
		  sprintf(time_stamp_monitor, "%s.%04dUTC ", time_stamp_monitor, (int)((time_stamp_monitor_f - time_stamp_monitor_i) * 1E4 + 0.5));
		  for(k = 0; k < NDATA_PER_SAMP_FULL; k++)
		    {
		      eth_index = i * baseband2filterbank_conf->nstream * NDATA_PER_SAMP_FULL + j * NDATA_PER_SAMP_FULL + k;
		      strncpy(fits_monitor[eth_index].time_stamp, time_stamp_monitor, FITS_TIME_STAMP_LEN);		  
		      fits_monitor[eth_index].tsamp = time_res_monitor;
		      fits_monitor[eth_index].nchan = baseband2filterbank_conf->nchan_out;
		      fits_monitor[eth_index].chan_width = chan_width_monitor;
		      fits_monitor[eth_index].pol_type = baseband2filterbank_conf->ptype_monitor;
		      fits_monitor[eth_index].pol_index = k;
		      fits_monitor[eth_index].beam_index  = baseband2filterbank_conf->beam_index;
		      fits_monitor[eth_index].center_freq = baseband2filterbank_conf->cfreq_band;
		      fits_monitor[eth_index].nchunk = 1;
		      fits_monitor[eth_index].chunk_index = 0;
		      
		      if(fits_monitor[eth_index].nchan == 0)
			{
			  fprintf(stdout, "We get a bad monitor packet before queue\n");
			  log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "We get a bad monitor packet before queue");
			  fflush(stdout);
			}
		      
		      if(k < baseband2filterbank_conf->ptype_monitor)
			{
			  if(baseband2filterbank_conf->ptype_monitor == 2)
			    {
			      CudaSafeCall(cudaMemcpyAsync(fits_monitor[eth_index].data,
							   &baseband2filterbank_conf->dbuf_out_monitor3[dbufout_offset_monitor3 +
									   baseband2filterbank_conf->nchan_out  *
									   (NDATA_PER_SAMP_FULL + k)],
							   baseband2filterbank_conf->dtsz_network,
							   cudaMemcpyDeviceToHost,
							   baseband2filterbank_conf->streams[j]));
			    }
			  else
			    CudaSafeCall(cudaMemcpyAsync(fits_monitor[eth_index].data,
							 &baseband2filterbank_conf->dbuf_out_monitor3[dbufout_offset_monitor3 +
									 k * baseband2filterbank_conf->nchan_out],
							 baseband2filterbank_conf->dtsz_network,
							 cudaMemcpyDeviceToHost,
							 baseband2filterbank_conf->streams[j]));
			}
		    }
		  time_stamp_monitor_f += time_res_monitor;
		}
	      else
	      	{		  
		  switch (blocksize_detect_faccumulate_scale.x)
		    {
		    case 1024:
		      detect_faccumulate_scale2_kernel<1024><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->buf_rt2[bufrt2_offset], &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank], baseband2filterbank_conf->nsamp_keep, baseband2filterbank_conf->naccumulate_scale, baseband2filterbank_conf->offset_scale_d);
		      break;
		    case 512:
		      detect_faccumulate_scale2_kernel< 512><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->buf_rt2[bufrt2_offset], &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank], baseband2filterbank_conf->nsamp_keep, baseband2filterbank_conf->naccumulate_scale, baseband2filterbank_conf->offset_scale_d);
		      break;
		    case 256:
		      detect_faccumulate_scale2_kernel< 256><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->buf_rt2[bufrt2_offset], &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank], baseband2filterbank_conf->nsamp_keep, baseband2filterbank_conf->naccumulate_scale, baseband2filterbank_conf->offset_scale_d);
		      break;
		    case 128:
		      detect_faccumulate_scale2_kernel< 128><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->buf_rt2[bufrt2_offset], &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank], baseband2filterbank_conf->nsamp_keep, baseband2filterbank_conf->naccumulate_scale, baseband2filterbank_conf->offset_scale_d);
		      break;
		    case 64:
		      detect_faccumulate_scale2_kernel<  64><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->buf_rt2[bufrt2_offset], &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank], baseband2filterbank_conf->nsamp_keep, baseband2filterbank_conf->naccumulate_scale, baseband2filterbank_conf->offset_scale_d);
		      break;
		    case 32:
		      detect_faccumulate_scale2_kernel<  32><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->buf_rt2[bufrt2_offset], &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank], baseband2filterbank_conf->nsamp_keep, baseband2filterbank_conf->naccumulate_scale, baseband2filterbank_conf->offset_scale_d);
		      break;
		    case 16:
		      detect_faccumulate_scale2_kernel<  16><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->buf_rt2[bufrt2_offset], &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank], baseband2filterbank_conf->nsamp_keep, baseband2filterbank_conf->naccumulate_scale, baseband2filterbank_conf->offset_scale_d);
		      break;
		    case 8:
		      detect_faccumulate_scale2_kernel<   8><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->buf_rt2[bufrt2_offset], &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank], baseband2filterbank_conf->nsamp_keep, baseband2filterbank_conf->naccumulate_scale, baseband2filterbank_conf->offset_scale_d);
		      break;
		    case 4:
		      detect_faccumulate_scale2_kernel<   4><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->buf_rt2[bufrt2_offset], &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank], baseband2filterbank_conf->nsamp_keep, baseband2filterbank_conf->naccumulate_scale, baseband2filterbank_conf->offset_scale_d);
		      break;
		    case 2:
		      detect_faccumulate_scale2_kernel<   2><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->buf_rt2[bufrt2_offset], &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank], baseband2filterbank_conf->nsamp_keep, baseband2filterbank_conf->naccumulate_scale, baseband2filterbank_conf->offset_scale_d);
		      break;
		    case 1:
		      detect_faccumulate_scale2_kernel<   1><<<gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale.x * sizeof(float), baseband2filterbank_conf->streams[j]>>>(&baseband2filterbank_conf->buf_rt2[bufrt2_offset], &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank], baseband2filterbank_conf->nsamp_keep, baseband2filterbank_conf->naccumulate_scale, baseband2filterbank_conf->offset_scale_d);
		      break;
		    }
		  CudaSafeKernelLaunch(); 
	      	}	
	      CudaSafeCall(cudaMemcpyAsync(&baseband2filterbank_conf->cbuf_out[hbufout_offset_filterbank], &baseband2filterbank_conf->dbuf_out_filterbank[dbufout_offset_filterbank], baseband2filterbank_conf->sbufout_size_filterbank, cudaMemcpyDeviceToHost, baseband2filterbank_conf->streams[j]));
	    }
	}
      CudaSynchronizeCall(); // Sync here is for multiple streams

      //if((baseband2filterbank_conf->spectral2disk == 1) || (baseband2filterbank_conf->spectral2network == 1))
      //	{
      //	  saccumulate_kernel
      //	    <<<gridsize_saccumulate,
      //	    blocksize_saccumulate>>>
      //	    (baseband2filterbank_conf->dbuf_out_spectral,
      //	     baseband2filterbank_conf->ndata_out_spectral,
      //	     baseband2filterbank_conf->nstream);  
      //	  CudaSafeKernelLaunch();
      //	}
      log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "before closing old buffer block");
      ipcbuf_mark_filled(baseband2filterbank_conf->db_out, (uint64_t)(cbufsz * baseband2filterbank_conf->scale_dtsz));
      //ipcbuf_mark_filled(baseband2filterbank_conf->db_out, baseband2filterbank_conf->bufout_size_filterbank * baseband2filterbank_conf->nrepeat_per_blk);
      //ipcbuf_mark_filled(baseband2filterbank_conf->db_out, baseband2filterbank_conf->rbufout_size_filterbank);
      ipcbuf_mark_cleared(baseband2filterbank_conf->db_in);
      log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "after closing old buffer block");

      time_offset += time_res_blk;
      fprintf(stdout, "BASEBAND2FILTERBANK, finished %f seconds data\n", time_offset);
      log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "finished %f seconds data", time_offset);
      fflush(stdout);
      nblk_accumulate++;
      
      if(baseband2filterbank_conf->monitor == 1)
	{
	  for(i = 0; i < baseband2filterbank_conf->neth_per_blk; i++)
	    enqueue(queue_fits_monitor, fits_monitor[i]); // Put the FITS into the queue
	}
      
      if(nblk_accumulate == baseband2filterbank_conf->nblk_accumulate)
	{
	  if((baseband2filterbank_conf->spectral2disk == 1) || (baseband2filterbank_conf->spectral2network == 1))
	    {
	      saccumulate_kernel
		<<<gridsize_saccumulate,
		blocksize_saccumulate>>>
		(baseband2filterbank_conf->dbuf_out_spectral,
		 baseband2filterbank_conf->ndata_out_spectral,
		 baseband2filterbank_conf->nstream);  
	      CudaSafeKernelLaunch();
	    }
	  if(baseband2filterbank_conf->spectral2disk == 1)
	    {
	      if(baseband2filterbank_conf->ptype_spectral == 2)
		CudaSafeCall(cudaMemcpy(baseband2filterbank_conf->cbuf_out_spectral,
					&baseband2filterbank_conf->dbuf_out_spectral[baseband2filterbank_conf->nsamp_out_spectral  * NDATA_PER_SAMP_FULL],
					2 * baseband2filterbank_conf->nsamp_out_spectral * NBYTE_SPECTRAL,
					cudaMemcpyDeviceToHost));
	      else
		CudaSafeCall(cudaMemcpy(baseband2filterbank_conf->cbuf_out_spectral,
					baseband2filterbank_conf->dbuf_out_spectral,
					baseband2filterbank_conf->nsamp_out_spectral  * baseband2filterbank_conf->ptype_spectral * NBYTE_SPECTRAL,
					cudaMemcpyDeviceToHost));
	    }
	  if(baseband2filterbank_conf->spectral2network == 1)
	    {
	      time_stamp_spectral_i = (time_t)time_stamp_spectral_f;
	      strftime(time_stamp_spectral,
		       FITS_TIME_STAMP_LEN,
		       FITS_TIMESTR,
		       gmtime(&time_stamp_spectral_i));    // String start time without fraction second
	      sprintf(time_stamp_spectral,
		      "%s.%04dUTC ",
		      time_stamp_spectral,
		      (int)((time_stamp_spectral_f - time_stamp_spectral_i) * 1E4 + 0.5));// To put the fraction part in and make sure that it rounds to closest integer
	      for(i = 0; i < NDATA_PER_SAMP_FULL; i++)
		{
		  for(j = 0; j < baseband2filterbank_conf->nchunk_network_spectral; j++)
		    {
		      memset(fits_spectral.data, 0x00, sizeof(fits_spectral.data));
		      
		      strncpy(fits_spectral.time_stamp, time_stamp_spectral, FITS_TIME_STAMP_LEN);		      	      
		      fits_spectral.pol_index = i;	      
		      fits_spectral.nchan = baseband2filterbank_conf->nchan_out_spectral;
		      fits_spectral.pol_type = baseband2filterbank_conf->ptype_spectral;
		      fits_spectral.nchunk   = baseband2filterbank_conf->nchan_in_spectral;	  
		      fits_spectral.chan_width = baseband2filterbank_conf->nchan_in_spectral / (double)baseband2filterbank_conf->nchan_out_spectral;
		      fits_spectral.center_freq = baseband2filterbank_conf->cfreq_spectral;
		      fits_spectral.tsamp = time_res_spectral;
		      fits_spectral.beam_index = baseband2filterbank_conf->beam_index;
	  
		      if(fits_spectral.nchan == 0)
			{
			  fprintf(stdout, "We get a bad spectral packet before queue\n");
			  log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "We get a bad spectral packet before queue");
			  fflush(stdout);
			}
		      
		      memcpy_offset = i * baseband2filterbank_conf->nchan_out_spectral +
			j * baseband2filterbank_conf->nchan_per_chunk_network_spectral;
		      fits_spectral.chunk_index = j;
		      if(i < baseband2filterbank_conf->ptype_spectral)
			{
			  if(baseband2filterbank_conf->ptype_spectral == 2)
			    CudaSafeCall(cudaMemcpy(fits_spectral.data,
						    &baseband2filterbank_conf->dbuf_out_spectral[baseband2filterbank_conf->nsamp_out_spectral  * NDATA_PER_SAMP_FULL + memcpy_offset],
						    baseband2filterbank_conf->dtsz_network_spectral,
						    cudaMemcpyDeviceToHost));
			  else
			    CudaSafeCall(cudaMemcpy(fits_spectral.data,
						    &baseband2filterbank_conf->dbuf_out_spectral[memcpy_offset],
						    baseband2filterbank_conf->dtsz_network_spectral,
						    cudaMemcpyDeviceToHost));
			}
		      enqueue(queue_fits_spectral, fits_spectral); // Put the FITS into the queue
		    }
		}
	      time_stamp_spectral_f += fits_spectral.tsamp;
	    }
	  
	  if(baseband2filterbank_conf->spectral2disk == 1)
	    {
	      log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "before closing old buffer block");
	      ipcbuf_mark_filled(baseband2filterbank_conf->db_out_spectral, (uint64_t)(baseband2filterbank_conf->nblk_accumulate * cbufsz * baseband2filterbank_conf->scale_dtsz));
	      //ipcbuf_mark_filled(baseband2filterbank_conf->db_out_spectral, baseband2filterbank_conf->bufout_size * baseband2filterbank_conf->nrepeat_per_blk);
	      //ipcbuf_mark_filled(baseband2filterbank_conf->db_out_spectral, baseband2filterbank_conf->rbufout_size_spectral);
	    }
	  
	  nblk_accumulate = 0;
	  //CudaSafeCall(cudaMemset((void *)baseband2filterbank_conf->dbuf_out_spectral, 0, sizeof(baseband2filterbank_conf->dbuf_out_spectral)));// We have to clear the memory for this parameter
	  CudaSafeCall(cudaMemset((void *)baseband2filterbank_conf->dbuf_out_spectral, 0, baseband2filterbank_conf->bufout_size_spectral));// We have to clear the memory for this parameter
	}
    }
  
  log_add(baseband2filterbank_conf->log_file, "INFO", 1,  "FINISH the process");

  quit = 1;
  
  if(baseband2filterbank_conf->monitor)
    {
      for(i = 0; i < baseband2filterbank_conf->neth_per_blk; i++)
	cudaHostUnregister ((void *) fits_monitor[i].data);
      free(fits_monitor);
    }
  if(baseband2filterbank_conf->spectral2network == 1)
    cudaHostUnregister ((void *) fits_spectral.data);
  pthread_exit(NULL);
}
