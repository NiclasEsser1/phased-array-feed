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
#include <sys/types.h>
#include <cuda_profiler_api.h>

#include "baseband2spectral.cuh"
#include "cudautil.cuh"
#include "kernel.cuh"
#include "log.h"
#include "constants.h"

extern pthread_mutex_t log_mutex;

int default_arguments(conf_t *conf)
{
  memset(conf->dir, 0x00, sizeof(conf->dir));
  sprintf(conf->dir, "unset"); // Default with "unset"
  memset(conf->ip, 0x00, sizeof(conf->ip));
  sprintf(conf->ip, "unset"); // Default with "unset"
  
  memset(conf->ip_monitor, 0x00, sizeof(conf->ip_monitor));
  sprintf(conf->ip_monitor, "unset"); // Default with "unset"

  conf->port_monitor = -1;
  conf->monitor = 0; // default no monitor
  conf->ptype_monitor = -1;

  conf->ndf_per_chunk_rbufin = 0; // Default with an impossible value
  conf->nstream              = -1;// Default with an impossible value
  conf->ndf_per_chunk_stream = 0; // Default with an impossible value
  conf->sod = -1;                 // Default no SOD at the beginning
  conf->nchunk_in = -1;
  conf->port = -1;
  conf->cufft_nx = -1;
  conf->output_network = -1;
  conf->pol_type = -1;
  conf->ndim_out = -1;
  conf->npol_out = -1;
  conf->nblk_accumulate = -1;
  
  return EXIT_SUCCESS;
}

int initialize_baseband2spectral(conf_t *conf)
{
  int i;
  int iembed, istride, idist, oembed, ostride, odist, batch, nx;
  uint64_t naccumulate_pow2;

  CudaSafeCall(cudaProfilerStart());
    
  /* Prepare parameters */
  conf->naccumulate     = conf->ndf_per_chunk_stream * NSAMP_DF / conf->cufft_nx;
  conf->nrepeat_per_blk = conf->ndf_per_chunk_rbufin / (conf->ndf_per_chunk_stream * conf->nstream);
  conf->nchan_in        = conf->nchunk_in * NCHAN_PER_CHUNK;
  conf->nchan_keep_chan = (int)(conf->cufft_nx / OVER_SAMP_RATE);
  conf->nchan_out       = conf->nchan_in * conf->nchan_keep_chan;
  conf->cufft_mod       = (int)(0.5 * conf->nchan_keep_chan);
  conf->scale_dtsz      = NBYTE_SPECTRAL * NDATA_PER_SAMP_FULL * conf->nchan_out / (double)(NBYTE_BASEBAND * NPOL_BASEBAND * NDIM_BASEBAND * conf->ndf_per_chunk_rbufin * conf->nchan_in * NSAMP_DF * conf->nblk_accumulate); // replace NDATA_PER_SAMP_FULL with conf->pol_type if we do not fill 0 for other pols

  log_add(conf->log_file, "INFO", 1, log_mutex, "We have %d channels input", conf->nchan_in);
  log_add(conf->log_file, "INFO", 1, log_mutex, "The mod to reduce oversampling is %d", conf->cufft_mod);
  log_add(conf->log_file, "INFO", 1, log_mutex, "We will keep %d fine channels for each input channel after FFT", conf->nchan_keep_chan);
  log_add(conf->log_file, "INFO", 1, log_mutex, "The data size rate between spectral and baseband data is %E", conf->scale_dtsz);
  log_add(conf->log_file, "INFO", 1, log_mutex, "%d run to finish one ring buffer block", conf->nrepeat_per_blk);

  /* Prepare monitor */
  conf->fits_monitor = NULL;
  if(conf->monitor == 1)
    {
      conf->nseg_per_blk = conf->nstream * conf->nrepeat_per_blk;
      conf->neth_per_blk = conf->nseg_per_blk * NDATA_PER_SAMP_FULL;
      conf->fits_monitor = (fits_t *)malloc(conf->neth_per_blk * sizeof(fits_t));
      for(i = 0; i < conf->neth_per_blk; i++)
      	{
      	  memset(conf->fits_monitor[i].data, 0x00, sizeof(conf->fits_monitor[i].data));
      	  cudaHostRegister ((void *) conf->fits_monitor[i].data, sizeof(conf->fits_monitor[i].data), 0);
      	}
      log_add(conf->log_file, "INFO", 1, log_mutex, "%d network packets are requied for each buffer block", conf->neth_per_blk);
      
      conf->dtsz_network_monitor  = NBYTE_FLOAT * conf->nchan_in;
      conf->pktsz_network_monitor = conf->dtsz_network_monitor + 3 * NBYTE_FLOAT + 6 * NBYTE_INT + FITS_TIME_STAMP_LEN;
      log_add(conf->log_file, "INFO", 1, log_mutex, "Network data size for monitor is %d", conf->dtsz_network_monitor);
      log_add(conf->log_file, "INFO", 1, log_mutex, "Network packet size for monitor is %d", conf->pktsz_network_monitor); 
    }
  
  /* Prepare buffer, stream and fft plan for process */
  conf->nsamp_in      = conf->ndf_per_chunk_stream * conf->nchan_in * NSAMP_DF;
  conf->npol_in       = conf->nsamp_in * NPOL_BASEBAND;
  conf->ndata_in      = conf->npol_in  * NDIM_BASEBAND;
  
  conf->nsamp_keep      = conf->nsamp_in / OVER_SAMP_RATE;
  conf->npol_keep       = conf->nsamp_keep * NPOL_BASEBAND;
  conf->ndata_keep      = conf->npol_keep  * NDIM_BASEBAND;

  conf->nsamp_out      = conf->nsamp_keep / conf->naccumulate;
  conf->ndata_out      = conf->nsamp_out  * NDATA_PER_SAMP_RT;
  
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
      CufftSafeCall(cufftPlanMany(&conf->fft_plans[i],
				  CUFFT_RANK, &nx, &iembed,
				  istride, idist, &oembed,
				  ostride, odist, CUFFT_C2C, batch));
      CufftSafeCall(cufftSetStream(conf->fft_plans[i], conf->streams[i]));
    }
  
  conf->sbufin_size  = conf->ndata_in * NBYTE_BASEBAND;
  conf->sbufout_size = conf->ndata_out * NBYTE_SPECTRAL;
  conf->sbufout_size_monitor1 = conf->ndata_out * NBYTE_FLOAT * NDATA_PER_SAMP_RT;
  conf->sbufout_size_monitor2 = conf->nchan_in * NBYTE_FLOAT * NDATA_PER_SAMP_RT;
  
  conf->bufin_size   = conf->nstream * conf->sbufin_size;
  conf->bufout_size  = conf->nstream * conf->sbufout_size;
  conf->bufout_size_monitor1 = conf->nstream * conf->sbufout_size_monitor1;
  conf->bufout_size_monitor2 = conf->nstream * conf->sbufout_size_monitor2;
  
  conf->sbufrt1_size = conf->npol_in * NBYTE_CUFFT_COMPLEX;
  conf->sbufrt2_size = conf->npol_keep * NBYTE_CUFFT_COMPLEX;
  conf->bufrt1_size  = conf->nstream * conf->sbufrt1_size;
  conf->bufrt2_size  = conf->nstream * conf->sbufrt2_size;
    
  conf->hbufin_offset = conf->sbufin_size;
  conf->dbufin_offset = conf->sbufin_size / (NBYTE_BASEBAND * NPOL_BASEBAND * NDIM_BASEBAND);
  conf->bufrt1_offset = conf->sbufrt1_size / NBYTE_CUFFT_COMPLEX;
  conf->bufrt2_offset = conf->sbufrt2_size / NBYTE_CUFFT_COMPLEX;
  
  conf->dbufout_offset = conf->sbufout_size / NBYTE_SPECTRAL;
  conf->dbufout_offset_monitor1 = conf->sbufout_size_monitor1 / NBYTE_FLOAT;
  conf->dbufout_offset_monitor2 = conf->sbufout_size_monitor2 / NBYTE_FLOAT;
  
  conf->dbuf_in = NULL;
  conf->dbuf_out = NULL;
  conf->buf_rt1 = NULL;
  conf->buf_rt2 = NULL;
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_in, conf->bufin_size));  
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out, conf->bufout_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt1, conf->bufrt1_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt2, conf->bufrt2_size));
  
  if(conf->monitor == 1)
    {      
      conf->dbuf_out_monitor1 = NULL;
      conf->dbuf_out_monitor2 = NULL;
      CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out_monitor1, conf->bufout_size_monitor1));
      log_add(conf->log_file, "INFO", 1, log_mutex, "bufout_size_monitor1 is %"PRIu64"", conf->bufout_size_monitor1);      
      CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out_monitor2, conf->bufout_size_monitor2));
      log_add(conf->log_file, "INFO", 1, log_mutex, "bufout_size_monitor1 is %"PRIu64"", conf->bufout_size_monitor2);      
    }
  
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
  
  conf->gridsize_swap_select_transpose_pft1.x = ceil(conf->cufft_nx / (double)TILE_DIM);  
  conf->gridsize_swap_select_transpose_pft1.y = ceil(conf->ndf_per_chunk_stream * NSAMP_DF / (double) (conf->cufft_nx * TILE_DIM));
  conf->gridsize_swap_select_transpose_pft1.z = conf->nchan_in;
  conf->blocksize_swap_select_transpose_pft1.x = TILE_DIM;
  conf->blocksize_swap_select_transpose_pft1.y = NROWBLOCK_TRANS;
  conf->blocksize_swap_select_transpose_pft1.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex,
	  "The configuration of swap_select_transpose_pft1 kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_swap_select_transpose_pft1.x,
	  conf->gridsize_swap_select_transpose_pft1.y,
	  conf->gridsize_swap_select_transpose_pft1.z,
	  conf->blocksize_swap_select_transpose_pft1.x,
	  conf->blocksize_swap_select_transpose_pft1.y,
	  conf->blocksize_swap_select_transpose_pft1.z);

  naccumulate_pow2 = (uint64_t)pow(2.0, floor(log2((double)conf->naccumulate)));
  conf->gridsize_spectral_taccumulate.x = conf->nchan_in;
  conf->gridsize_spectral_taccumulate.y = conf->nchan_keep_chan;
  conf->gridsize_spectral_taccumulate.z = 1;
  conf->blocksize_spectral_taccumulate.x = (naccumulate_pow2<1024)?naccumulate_pow2:1024;
  conf->blocksize_spectral_taccumulate.y = 1;
  conf->blocksize_spectral_taccumulate.z = 1; 
  log_add(conf->log_file, "INFO", 1, log_mutex,
	  "The configuration of spectral_taccumulate kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_spectral_taccumulate.x,
	  conf->gridsize_spectral_taccumulate.y,
	  conf->gridsize_spectral_taccumulate.z,
	  conf->blocksize_spectral_taccumulate.x,
	  conf->blocksize_spectral_taccumulate.y,
	  conf->blocksize_spectral_taccumulate.z);

  naccumulate_pow2 = (uint64_t)pow(2.0, floor(log2((double)conf->nchan_keep_chan)));
  conf->gridsize_faccumulate.x = conf->nchan_in;
  conf->gridsize_faccumulate.y = 1;
  conf->gridsize_faccumulate.z = 1;
  conf->blocksize_faccumulate.x = naccumulate_pow2;
  conf->blocksize_faccumulate.y = 1;
  conf->blocksize_faccumulate.z = 1;
  log_add(conf->log_file, "INFO", 1, log_mutex, "The configuration of frequency accumulate kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_faccumulate.x, conf->gridsize_faccumulate.y, conf->gridsize_faccumulate.z,
	  conf->blocksize_faccumulate.x, conf->blocksize_faccumulate.y, conf->blocksize_faccumulate.z);

  conf->gridsize_saccumulate.x = NDATA_PER_SAMP_RT;
  conf->gridsize_saccumulate.y = conf->nchan_keep_chan;
  conf->gridsize_saccumulate.z = 1;
  conf->blocksize_saccumulate.x = conf->nchan_in;
  conf->blocksize_saccumulate.y = 1;
  conf->blocksize_saccumulate.z = 1; 
  log_add(conf->log_file, "INFO", 1, log_mutex,
	  "The configuration of saccumulate kernel is (%d, %d, %d) and (%d, %d, %d)",
	  conf->gridsize_saccumulate.x,
	  conf->gridsize_saccumulate.y,
	  conf->gridsize_saccumulate.z,
	  conf->blocksize_saccumulate.x,
	  conf->blocksize_saccumulate.y,
	  conf->blocksize_saccumulate.z);        

  /* attach to input ring buffer */
  conf->hdu_in = dada_hdu_create(NULL);
  dada_hdu_set_key(conf->hdu_in, conf->key_in);
  if(dada_hdu_connect(conf->hdu_in) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex,
	      "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }  
  conf->db_in = (ipcbuf_t *) conf->hdu_in->data_block;
  conf->rbufin_size = ipcbuf_get_bufsz(conf->db_in);
  log_add(conf->log_file, "INFO", 1, log_mutex, "Input buffer block size is %"PRIu64".", conf->rbufin_size);
  if(conf->rbufin_size != (conf->bufin_size * conf->nrepeat_per_blk))  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }

  struct timespec start, stop;
  double elapsed_time;

  /* registers the existing host memory range for use by CUDA */
  clock_gettime(CLOCK_REALTIME, &start);
  dada_cuda_dbregister(conf->hdu_in); // To put this into capture does not improve the memcpy!!!
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "elapse_time for dbregister of input ring buffer is %f\n", elapsed_time);
  fflush(stdout);
  
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_in->header_block);  
  if(conf->hdrsz != DADA_HDRSZ)    // This number should match
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);    
    }
  
  /* make ourselves the read client */
  if(dada_hdu_lock_read(conf->hdu_in) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error locking HDU, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }

  /* Prepare output ring buffer */
  if(conf->output_network==0)
    {
      conf->hdu_out = dada_hdu_create(NULL);
      dada_hdu_set_key(conf->hdu_out, conf->key_out);
      if(dada_hdu_connect(conf->hdu_out) < 0)
	{
	  log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  
	  destroy_baseband2spectral(*conf);
	  fclose(conf->log_file);
	  exit(EXIT_FAILURE);    
	}
      conf->db_out = (ipcbuf_t *) conf->hdu_out->data_block;
      conf->rbufout_size = ipcbuf_get_bufsz(conf->db_out);
      log_add(conf->log_file, "INFO", 1, log_mutex, "Output buffer block size is %"PRIu64".", conf->rbufout_size);
      
      clock_gettime(CLOCK_REALTIME, &start);
      dada_cuda_dbregister(conf->hdu_out); // To put this into capture does not improve the memcpy!!!
      clock_gettime(CLOCK_REALTIME, &stop);
      elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
      fprintf(stdout, "elapse_time for dbregister of output ring buffer is %f\n", elapsed_time);
      fflush(stdout);
      
      if(conf->rbufout_size != conf->nsamp_out * NDATA_PER_SAMP_FULL * NBYTE_SPECTRAL)
	{
	  // replace NDATA_PER_SAMP_FULL with conf->pol_type if we do not fill 0 for other pols
	  log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, %"PRIu64" vs %"PRIu64", which happens at \"%s\", line [%d].", conf->rbufout_size, conf->nsamp_out * NDATA_PER_SAMP_FULL * NBYTE_SPECTRAL, __FILE__, __LINE__);
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Buffer size mismatch, %"PRIu64" vs %"PRIu64", which happens at \"%s\", line [%d].\n", conf->rbufout_size, conf->nsamp_out * NDATA_PER_SAMP_FULL * NBYTE_SPECTRAL, __FILE__, __LINE__);
	  
	  destroy_baseband2spectral(*conf);
	  fclose(conf->log_file);
	  exit(EXIT_FAILURE);    
	}
      
      conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_out->header_block);  
      if(conf->hdrsz != DADA_HDRSZ)    // This number should match
	{
	  log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  
	  destroy_baseband2spectral(*conf);
	  fclose(conf->log_file);
	  exit(EXIT_FAILURE);    
	}
      
      /* make ourselves the write client */
      if(dada_hdu_lock_write(conf->hdu_out) < 0)
	{
	  log_add(conf->log_file, "ERR", 1, log_mutex, "Error locking HDU, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  
	  destroy_baseband2spectral(*conf);
	  fclose(conf->log_file);
	  exit(EXIT_FAILURE);
	}
      
      if(conf->sod == 0)
	{
	  if(ipcbuf_disable_sod(conf->db_out) < 0)
	    {
	      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.", __FILE__, __LINE__);
	      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      
	      destroy_baseband2spectral(*conf);
	      fclose(conf->log_file);
	      exit(EXIT_FAILURE);
	    }
	}
    }
  if(conf->output_network == 1)
    {
      conf->nchunk_network = conf->nchan_in; // We send spectral of one input channel per udp packet;
      conf->nchan_per_chunk_network = conf->nchan_out/conf->nchunk_network;
      conf->dtsz_network = NBYTE_FLOAT * conf->nchan_per_chunk_network;
      conf->pktsz_network     = conf->dtsz_network + 3 * NBYTE_FLOAT + 6 * NBYTE_INT + FITS_TIME_STAMP_LEN;
      log_add(conf->log_file, "INFO", 1, log_mutex, "Spectral data will be sent with %d frequency chunks for each pol.", conf->nchunk_network);
      log_add(conf->log_file, "INFO", 1, log_mutex, "Spectral data will be sent with %d frequency channels in each frequency chunks.", conf->nchan_per_chunk_network);
      log_add(conf->log_file, "INFO", 1, log_mutex, "Size of spectral data in  each network packet is %d bytes.", conf->dtsz_network);
      log_add(conf->log_file, "INFO", 1, log_mutex, "Size of each network packet is %d bytes.", conf->pktsz_network);
    }
  
  return EXIT_SUCCESS;
}

int baseband2spectral(conf_t conf)
{
  uint64_t i, j, k;
  int nblk_accumulate = 0;
  uint64_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset, dbufout_offset, dbufout_offset_monitor1, dbufout_offset_monitor2;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_pft1, blocksize_swap_select_transpose_pft1;
  dim3 gridsize_spectral_taccumulate, blocksize_spectral_taccumulate;
  dim3 gridsize_saccumulate, blocksize_saccumulate;
  dim3 gridsize_faccumulate, blocksize_faccumulate;
  uint64_t cbufsz;
  double time_res_blk, time_offset = 0, time_res_stream;
  int eth_index;
  char time_stamp_monitor[MSTR_LEN];
  double time_stamp_monitor_f;
  time_t time_stamp_monitor_i;
  uint64_t memcpy_offset;
  double time_stamp_f;
  time_t time_stamp_i;
  
  struct sockaddr_in sa_udp, sa_udp_monitor;
  socklen_t tolen = sizeof(sa_udp);
  int sock_udp, sock_udp_monitor;
  int enable = 1;
  struct tm tm_stamp;
  fits_t fits;
  
  gridsize_unpack                      = conf.gridsize_unpack;
  blocksize_unpack                     = conf.blocksize_unpack;
  gridsize_swap_select_transpose_pft1  = conf.gridsize_swap_select_transpose_pft1;   
  blocksize_swap_select_transpose_pft1 = conf.blocksize_swap_select_transpose_pft1;
  gridsize_spectral_taccumulate        = conf.gridsize_spectral_taccumulate; 
  blocksize_spectral_taccumulate       = conf.blocksize_spectral_taccumulate;
  gridsize_saccumulate        = conf.gridsize_saccumulate; 
  blocksize_saccumulate       = conf.blocksize_saccumulate;
  gridsize_faccumulate        = conf.gridsize_faccumulate; 
  blocksize_faccumulate       = conf.blocksize_faccumulate;
  
  if(read_dada_header(&conf))
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "header read failed, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: header read failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2spectral(conf);
      fclose(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "read_dada_header done");
  
  time_res_blk = conf.tsamp_in * conf.ndf_per_chunk_rbufin * NSAMP_DF / 1.0E6; // This has to be after read_dada_header, in seconds
  if(conf.monitor == 1)
    {
      time_res_stream = conf.tsamp_in * conf.ndf_per_chunk_stream * NSAMP_DF / 1.0E6; // This has to be after read_register_header, in seconds
      strptime(conf.utc_start, DADA_TIMESTR, &tm_stamp);
      time_stamp_monitor_f = mktime(&tm_stamp) + conf.picoseconds / 1.0E12 + 0.5 * time_res_stream;

      
      if((sock_udp_monitor = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: socket creation failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1, log_mutex, "socket creation failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  
	  destroy_baseband2spectral(conf);
	  fclose(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      memset((char *) &sa_udp_monitor, 0, sizeof(sa_udp_monitor));
      sa_udp_monitor.sin_family      = AF_INET;
      sa_udp_monitor.sin_port        = htons(conf.port_monitor);
      sa_udp_monitor.sin_addr.s_addr = inet_addr(conf.ip_monitor);
      setsockopt(sock_udp_monitor, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
    }
  if(conf.output_network == 0)
    {
      if(register_dada_header(&conf))
	{
	  log_add(conf.log_file, "ERR", 1, log_mutex, "header register failed, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: header register failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  
	  destroy_baseband2spectral(conf);
	  fclose(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      log_add(conf.log_file, "INFO", 1, log_mutex, "register_dada_header done");
    }
  if(conf.output_network == 1)
    {      
      fits.nchan = conf.nchan_out;
      fits.pol_type = conf.pol_type;
      fits.nchunk   = conf.nchan_in;
      memset(fits.data, 0x00, sizeof(fits.data));
	
      fits.chan_width = conf.bandwidth / (double)conf.nchan_out;
      
      fits.center_freq = conf.center_freq;
      fits.tsamp = time_res_blk * conf.nblk_accumulate;
      fits.beam_index = conf.beam_index;
      strptime(conf.utc_start, DADA_TIMESTR, &tm_stamp);
      time_stamp_f = mktime(&tm_stamp) + conf.picoseconds / 1.0E12 + 0.5 * fits.tsamp;
      
      if((sock_udp = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
	{
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: socket creation failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1, log_mutex, "socket creation failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  
	  destroy_baseband2spectral(conf);
	  fclose(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      memset((char *) &sa_udp, 0, sizeof(sa_udp));
      sa_udp.sin_family      = AF_INET;
      sa_udp.sin_port        = htons(conf.port);
      sa_udp.sin_addr.s_addr = inet_addr(conf.ip);
      setsockopt(sock_udp, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
    }
  fprintf(stdout, "HERE AFTER REGISTER HEADER\n");
  
  log_add(conf.log_file, "INFO", 1, log_mutex, "register_dada_header done");
  
  CudaSafeCall(cudaMemset((void *)conf.dbuf_out, 0, sizeof(conf.dbuf_out)));// We have to clear the memory for this parameter
  while(!ipcbuf_eod(conf.db_in))
    {
      log_add(conf.log_file, "INFO", 1, log_mutex, "before getting new buffer block");
      conf.cbuf_in  = ipcbuf_get_next_read(conf.db_in, &cbufsz);
      if(conf.output_network == 0)
	{
	  conf.cbuf_out = ipcbuf_get_next_write(conf.db_out);
	  log_add(conf.log_file, "INFO", 1, log_mutex, "after getting new buffer block");
	}
      
      for(i = 0; i < conf.nrepeat_per_blk; i ++)
	{
	  for(j = 0; j < conf.nstream; j++)
	    {
	      hbufin_offset = (i * conf.nstream + j) * conf.hbufin_offset;// + i * conf.bufin_size;
	      dbufin_offset = j * conf.dbufin_offset; 
	      bufrt1_offset = j * conf.bufrt1_offset;
	      bufrt2_offset = j * conf.bufrt2_offset;
	      dbufout_offset = j * conf.dbufout_offset;
	      dbufout_offset_monitor1 = j * conf.dbufout_offset_monitor1;
	      dbufout_offset_monitor2 = j * conf.dbufout_offset_monitor2;
		
	      /* Copy data into device */
	      CudaSafeCall(cudaMemcpyAsync(&conf.dbuf_in[dbufin_offset],
					   &conf.cbuf_in[hbufin_offset],
					   conf.sbufin_size,
					   cudaMemcpyHostToDevice,
					   conf.streams[j]));
	      
	      /* Unpack raw data into cufftComplex array */
	      unpack_kernel
		<<<gridsize_unpack,
		blocksize_unpack,
		0,
		conf.streams[j]>>>
		(&conf.dbuf_in[dbufin_offset],
		 &conf.buf_rt1[bufrt1_offset],
		 conf.nsamp_in);
	      CudaSafeKernelLaunch();
	      
	      /* Do forward FFT */
	      CufftSafeCall(cufftExecC2C(conf.fft_plans[j],
					 &conf.buf_rt1[bufrt1_offset],
					 &conf.buf_rt1[bufrt1_offset],
					 CUFFT_FORWARD));
	      
	      /* from PFTF order to PFT order, also remove channel edge */
	      swap_select_transpose_pft1_kernel
		<<<gridsize_swap_select_transpose_pft1,
		blocksize_swap_select_transpose_pft1,
		0,
		conf.streams[j]>>>
		(&conf.buf_rt1[bufrt1_offset],
		 &conf.buf_rt2[bufrt2_offset],
		 conf.cufft_nx,
		 conf.ndf_per_chunk_stream * NSAMP_DF / conf.cufft_nx,
		 conf.nsamp_in,
		 conf.nsamp_keep,
		 conf.cufft_nx,
		 conf.cufft_mod,
		 conf.nchan_keep_chan);
	      CudaSafeKernelLaunch();

	      if(conf.monitor == 1)
		{
		  log_add(conf.log_file, "INFO", 1, log_mutex, "IN the processing loop with monitor");
		  /* Convert to required pol and accumulate in time */
		  switch(blocksize_spectral_taccumulate.x)
		    {
		    case 1024:
		      spectral_taccumulate_dual_kernel
			<1024>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
		      //(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 &conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case 512:
		      spectral_taccumulate_dual_kernel
			< 512>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 &conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case 256:
		      spectral_taccumulate_dual_kernel
			< 256>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 &conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case 128:
		      spectral_taccumulate_dual_kernel
			< 128>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 &conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case  64:
		      spectral_taccumulate_dual_kernel
			<  64>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 &conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case  32:
		      spectral_taccumulate_dual_kernel
			<  32>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 &conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case  16:
		      spectral_taccumulate_dual_kernel
			<  16>		    
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 &conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case  8:
		      spectral_taccumulate_dual_kernel
			<   8>		    
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 &conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case  4:
		      spectral_taccumulate_dual_kernel
			<   4>		    		    
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 &conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case  2:
		      spectral_taccumulate_dual_kernel
			<   2>		    		    		    
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 &conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case  1:
		      spectral_taccumulate_dual_kernel
			<   1>		    		    		    
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 &conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		    }
		  CudaSafeKernelLaunch();
		  log_add(conf.log_file, "INFO", 1, log_mutex, "IN the processing loop with monitor, first kernel done");
		  
		  /* Frequency accumulate kernel here */		  
		  switch (blocksize_faccumulate.x)
		    {
		    case 1024:
		      accumulate_float_kernel
			<1024>
			<<<gridsize_faccumulate,
			blocksize_faccumulate,
			blocksize_faccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			conf.streams[j]>>>
			(&conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 &conf.dbuf_out_monitor2[dbufout_offset_monitor2],
			 conf.nsamp_out,
			 conf.nchan_in,
			 conf.nchan_keep_chan);
		      break;
		      
		    case 512:
		      accumulate_float_kernel
			< 512>
			<<<gridsize_faccumulate,
			blocksize_faccumulate,
			blocksize_faccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			conf.streams[j]>>>
			(&conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 &conf.dbuf_out_monitor2[dbufout_offset_monitor2],
			 conf.nsamp_out,
			 conf.nchan_in,
			 conf.nchan_keep_chan);
		      break;
		      
		    case 256:
		      accumulate_float_kernel
			< 256>
			<<<gridsize_faccumulate,
			blocksize_faccumulate,
			blocksize_faccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			conf.streams[j]>>>
			(&conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 &conf.dbuf_out_monitor2[dbufout_offset_monitor2],
			 conf.nsamp_out,
			 conf.nchan_in,
			 conf.nchan_keep_chan);
		      break;
		      
		    case 128:
		      accumulate_float_kernel
			< 128>
			<<<gridsize_faccumulate,
			blocksize_faccumulate,
			blocksize_faccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			conf.streams[j]>>>
			(&conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 &conf.dbuf_out_monitor2[dbufout_offset_monitor2],
			 conf.nsamp_out,
			 conf.nchan_in,
			 conf.nchan_keep_chan);
		      break;
		      
		    case 64:
		      accumulate_float_kernel
			<  64>
			<<<gridsize_faccumulate,
			blocksize_faccumulate,
			blocksize_faccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			conf.streams[j]>>>
			(&conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 &conf.dbuf_out_monitor2[dbufout_offset_monitor2],
			 conf.nsamp_out,
			 conf.nchan_in,
			 conf.nchan_keep_chan);
		      break;
		      
		    case 32:
		      accumulate_float_kernel
			<  32>
			<<<gridsize_faccumulate,
			blocksize_faccumulate,
			blocksize_faccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			conf.streams[j]>>>
			(&conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 &conf.dbuf_out_monitor2[dbufout_offset_monitor2],
			 conf.nsamp_out,
			 conf.nchan_in,
			 conf.nchan_keep_chan);
		      break;
		      
		    case 16:
		      accumulate_float_kernel
			<  16>
			<<<gridsize_faccumulate,
			blocksize_faccumulate,
			blocksize_faccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			conf.streams[j]>>>
			(&conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 &conf.dbuf_out_monitor2[dbufout_offset_monitor2],
			 conf.nsamp_out,
			 conf.nchan_in,
			 conf.nchan_keep_chan);
		      break;
		      
		    case 8:
		      accumulate_float_kernel
			<   8>
			<<<gridsize_faccumulate,
			blocksize_faccumulate,
			blocksize_faccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			conf.streams[j]>>>
			(&conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 &conf.dbuf_out_monitor2[dbufout_offset_monitor2],
			 conf.nsamp_out,
			 conf.nchan_in,
			 conf.nchan_keep_chan);
		      break;
		      
		    case 4:
		      accumulate_float_kernel
			<   4>
			<<<gridsize_faccumulate,
			blocksize_faccumulate,
			blocksize_faccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			conf.streams[j]>>>
			(&conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 &conf.dbuf_out_monitor2[dbufout_offset_monitor2],
			 conf.nsamp_out,
			 conf.nchan_in,
			 conf.nchan_keep_chan);
		      break;
		      
		    case 2:
		      accumulate_float_kernel
			<   2>
			<<<gridsize_faccumulate,
			blocksize_faccumulate,
			blocksize_faccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			conf.streams[j]>>>
			(&conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 &conf.dbuf_out_monitor2[dbufout_offset_monitor2],
			 conf.nsamp_out,
			 conf.nchan_in,
			 conf.nchan_keep_chan);
		      break;
		      
		    case 1:
		      accumulate_float_kernel
			<   1>
			<<<gridsize_faccumulate,
			blocksize_faccumulate,
			blocksize_faccumulate.x * NBYTE_FLOAT * NDATA_PER_SAMP_RT,
			conf.streams[j]>>>
			(&conf.dbuf_out_monitor1[dbufout_offset_monitor1],
			 &conf.dbuf_out_monitor2[dbufout_offset_monitor2],
			 conf.nsamp_out,
			 conf.nchan_in,
			 conf.nchan_keep_chan);
		      break;
		    }
		  CudaSafeKernelLaunch();
		  log_add(conf.log_file, "INFO", 1, log_mutex, "IN the processing loop with monitor, second kernel done");
		  
		  /* Setup ethernet packets */
		  time_stamp_monitor_i = (time_t)time_stamp_monitor_f;
		  strftime(time_stamp_monitor, FITS_TIME_STAMP_LEN, FITS_TIMESTR, gmtime(&time_stamp_monitor_i)); 
		  sprintf(time_stamp_monitor, "%s.%04dUTC ", time_stamp_monitor, (int)((time_stamp_monitor_f - time_stamp_monitor_i) * 1E4 + 0.5));
		  for(k = 0; k < NDATA_PER_SAMP_FULL; k++)
		    {
		      log_add(conf.log_file, "INFO", 1, log_mutex, "IN the data packing loop1");
		      eth_index = i * conf.nstream * NDATA_PER_SAMP_FULL + j * NDATA_PER_SAMP_FULL + k;
		      log_add(conf.log_file, "INFO", 1, log_mutex, "IN the data packing loop2");
		      
		      strncpy(conf.fits_monitor[eth_index].time_stamp, time_stamp_monitor, FITS_TIME_STAMP_LEN);
		      log_add(conf.log_file, "INFO", 1, log_mutex, "IN the data packing loop3");
		      conf.fits_monitor[eth_index].tsamp = time_res_stream;
		      log_add(conf.log_file, "INFO", 1, log_mutex, "IN the data packing loop4");
		      conf.fits_monitor[eth_index].nchan = conf.nchan_in;
		      log_add(conf.log_file, "INFO", 1, log_mutex, "IN the data packing loop5");
		      conf.fits_monitor[eth_index].chan_width = conf.bandwidth/conf.nchan_in;
		      log_add(conf.log_file, "INFO", 1, log_mutex, "IN the data packing loop6");
		      conf.fits_monitor[eth_index].pol_type = conf.ptype_monitor;
		      log_add(conf.log_file, "INFO", 1, log_mutex, "IN the data packing loop7");
		      conf.fits_monitor[eth_index].pol_index = k;
		      log_add(conf.log_file, "INFO", 1, log_mutex, "IN the data packing loop8");
		      conf.fits_monitor[eth_index].beam_index  = conf.beam_index;
		      log_add(conf.log_file, "INFO", 1, log_mutex, "IN the data packing loop9");
		      conf.fits_monitor[eth_index].center_freq = conf.center_freq;
		      log_add(conf.log_file, "INFO", 1, log_mutex, "IN the data packing loop10");
		      conf.fits_monitor[eth_index].nchunk = 1;
		      log_add(conf.log_file, "INFO", 1, log_mutex, "IN the data packing loop11");
		      conf.fits_monitor[eth_index].chunk_index = 0;
		      log_add(conf.log_file, "INFO", 1, log_mutex, "IN the data packing loop12");
		      
		      if(k < conf.ptype_monitor)
			{
			  log_add(conf.log_file, "INFO", 1, log_mutex, "IN the data packing loop13");
			  if(conf.ptype_monitor == 2)
			    {
			      CudaSafeCall(cudaMemcpyAsync(conf.fits_monitor[eth_index].data,
							   &conf.dbuf_out_monitor2[dbufout_offset_monitor2 +
									   conf.nchan_in  *
									   (NDATA_PER_SAMP_FULL + k)],
							   conf.dtsz_network_monitor,
							   cudaMemcpyDeviceToHost,
							   conf.streams[j]));
			    }
			  else
			    CudaSafeCall(cudaMemcpyAsync(conf.fits_monitor[eth_index].data,
							 &conf.dbuf_out_monitor2[dbufout_offset_monitor2 +
									 k * conf.nchan_in],
							 conf.dtsz_network_monitor,
							 cudaMemcpyDeviceToHost,
							 conf.streams[j]));
			}
		      log_add(conf.log_file, "INFO", 1, log_mutex, "IN the data packing loop14");
		    }
		  time_stamp_monitor_f += time_res_stream;
		  log_add(conf.log_file, "INFO", 1, log_mutex, "IN the processing loop with monitor, data packing done");
		}
	      else
		{
		  /* Convert to required pol and accumulate in time */
		  switch(blocksize_spectral_taccumulate.x)
		    {
		    case 1024:
		      spectral_taccumulate_kernel
			<1024>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
		      //(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case 512:
		      spectral_taccumulate_kernel
			< 512>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case 256:
		      spectral_taccumulate_kernel
			< 256>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case 128:
		      spectral_taccumulate_kernel
			< 128>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case  64:
		      spectral_taccumulate_kernel
			<  64>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case  32:
		      spectral_taccumulate_kernel
			<  32>
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case  16:
		      spectral_taccumulate_kernel
			<  16>		    
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case  8:
		      spectral_taccumulate_kernel
			<   8>		    
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case  4:
		      spectral_taccumulate_kernel
			<   4>		    		    
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case  2:
		      spectral_taccumulate_kernel
			<   2>		    		    		    
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		      
		    case  1:
		      spectral_taccumulate_kernel
			<   1>		    		    		    
			<<<gridsize_spectral_taccumulate,
			blocksize_spectral_taccumulate,
			blocksize_spectral_taccumulate.x * NDATA_PER_SAMP_RT * NBYTE_SPECTRAL,
			conf.streams[j]>>>
			//(&conf.buf_rt2[conf.bufrt2_offset],
			(&conf.buf_rt2[bufrt2_offset],
			 &conf.dbuf_out[dbufout_offset],
			 conf.nsamp_keep,
			 conf.nsamp_out,
			 conf.naccumulate);
		      break;
		    }
		  CudaSafeKernelLaunch();
		}
	    }
	}
      CudaSynchronizeCall(); // Sync here is for multiple streams
      
      saccumulate_kernel
	<<<conf.gridsize_saccumulate,
	conf.blocksize_saccumulate>>>
	(conf.dbuf_out,
	 conf.ndata_out,
	 conf.nstream);  
      CudaSafeKernelLaunch();
      
      ipcbuf_mark_cleared(conf.db_in);
      log_add(conf.log_file, "INFO", 1, log_mutex, "after closing old buffer block");      
      nblk_accumulate++;
      
      /* Send all packets from the previous buffer block with one go */
      if(conf.monitor == 1)
	{
	  log_add(conf.log_file, "INFO", 1, log_mutex, "monitor bfore sending monitor data");
	  for(i = 0; i < conf.neth_per_blk; i++)
	    {
	      if(sendto(sock_udp_monitor, (void *)&conf.fits_monitor[i], conf.pktsz_network_monitor, 0, (struct sockaddr *)&sa_udp_monitor, tolen) == -1)
		{
		  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: sendto() failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
		  log_add(conf.log_file, "ERR", 1, log_mutex, "sendto() failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
		  
		  destroy_baseband2spectral(conf);
		  fclose(conf.log_file);
		  exit(EXIT_FAILURE);
		}
	    }
	  log_add(conf.log_file, "INFO", 1, log_mutex, "monitor data sending done");
	}

      if(nblk_accumulate == conf.nblk_accumulate)
	{
	  if(conf.output_network == 0)
	    {
	      if(conf.pol_type == 2)
		CudaSafeCall(cudaMemcpy(conf.cbuf_out,
					&conf.dbuf_out[conf.nsamp_out  * NDATA_PER_SAMP_FULL],
					2 * conf.nsamp_out * NBYTE_SPECTRAL,
					cudaMemcpyDeviceToHost));
	      else
		CudaSafeCall(cudaMemcpy(conf.cbuf_out,
					conf.dbuf_out,
					conf.nsamp_out  * conf.pol_type * NBYTE_SPECTRAL,
					cudaMemcpyDeviceToHost));
	    }
	  if(conf.output_network == 1)
	    {
	      time_stamp_i = (time_t)time_stamp_f;
	      strftime(fits.time_stamp,
		       FITS_TIME_STAMP_LEN,
		       FITS_TIMESTR,
		       gmtime(&time_stamp_i));    // String start time without fraction second
	      sprintf(fits.time_stamp,
		      "%s.%04dUTC ",
		      fits.time_stamp,
		      (int)((time_stamp_f - time_stamp_i) * 1E4 + 0.5));// To put the fraction part in and make sure that it rounds to closest integer
	      
	      for(i = 0; i < NDATA_PER_SAMP_FULL; i++)
		{
		  fits.pol_index = i;
		  for(j = 0; j < conf.nchunk_network; j++)
		    {
		      memset(fits.data, 0x00, sizeof(fits.data));
		      memcpy_offset = i * fits.nchan +
			j * conf.nchan_per_chunk_network;
		      fits.chunk_index = j;
		      if(i < conf.pol_type)
			{
			  if(conf.pol_type == 2)
			    CudaSafeCall(cudaMemcpy(fits.data,
						    &conf.dbuf_out[conf.nsamp_out  * NDATA_PER_SAMP_FULL + memcpy_offset],
						    conf.dtsz_network,
						    cudaMemcpyDeviceToHost));
			  else
			    CudaSafeCall(cudaMemcpy(fits.data,
						    &conf.dbuf_out[memcpy_offset],
						    conf.dtsz_network,
						    cudaMemcpyDeviceToHost));
			}
		      if(sendto(sock_udp,
				(void *)&fits,
				conf.pktsz_network,
				0,
				(struct sockaddr *)&sa_udp,
				tolen) == -1)
			{
			  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: sendto() failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
			  log_add(conf.log_file, "ERR", 1, log_mutex, "sendto() failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
			  
			  destroy_baseband2spectral(conf);
			  fclose(conf.log_file);
			  exit(EXIT_FAILURE);
			}
		    }
		}
	      time_stamp_f += fits.tsamp;
	    }
	  
	  if(conf.output_network == 0)
	    {
	      log_add(conf.log_file, "INFO", 1, log_mutex, "before closing old buffer block");
	      ipcbuf_mark_filled(conf.db_out, (uint64_t)(conf.nblk_accumulate * cbufsz * conf.scale_dtsz));
	      //ipcbuf_mark_filled(conf.db_out, conf.bufout_size * conf.nrepeat_per_blk);
	      //ipcbuf_mark_filled(conf.db_out, conf.rbufout_size);
	    }
	  
	  nblk_accumulate = 0;
	  CudaSafeCall(cudaMemset((void *)conf.dbuf_out, 0, sizeof(conf.dbuf_out)));// We have to clear the memory for this parameter
	}
      
      time_offset += time_res_blk;
      fprintf(stdout, "BASEBAND2SPECTRAL, finished %f seconds data\n", time_offset);
      log_add(conf.log_file, "INFO", 1, log_mutex, "finished %f seconds data", time_offset);
      fflush(stdout);
    }
  CudaSynchronizeCall(); // Sync here is for multiple streams
  
  log_add(conf.log_file, "INFO", 1, log_mutex, "FINISH the process");

  return EXIT_SUCCESS;
}

int destroy_baseband2spectral(conf_t conf)
{
  int i;
  for (i = 0; i < conf.nstream; i++)
    {
      if(conf.fft_plans[i])
	{
	  fprintf(stdout, "HERE before cufftDestroy\n");
	  fflush(stdout);
	  CufftSafeCall(cufftDestroy(conf.fft_plans[i]));
	}
    }
  if(conf.fft_plans)
    free(conf.fft_plans);
  log_add(conf.log_file, "INFO", 1, log_mutex, "destroy fft plan and stream done");

  if(conf.monitor)
    {
      if(conf.fits_monitor)
	{
	  for(i = 0; i < conf.neth_per_blk; i++)
	    cudaHostUnregister ((void *) conf.fits_monitor[i].data);
	  free(conf.fits_monitor);
	}
      if(conf.dbuf_out_monitor1)
	cudaFree(conf.dbuf_out_monitor1);
      if(conf.dbuf_out_monitor2)
	cudaFree(conf.dbuf_out_monitor2);
    }
    
  if(conf.dbuf_in)
    cudaFree(conf.dbuf_in);
  if(conf.dbuf_out)
    cudaFree(conf.dbuf_out);
  if(conf.buf_rt1)
    cudaFree(conf.buf_rt1);
  if(conf.buf_rt2)
    cudaFree(conf.buf_rt2);
  log_add(conf.log_file, "INFO", 1, log_mutex, "Free cuda memory done");

  if(conf.db_in)
    {
      dada_cuda_dbunregister(conf.hdu_in);
      dada_hdu_unlock_read(conf.hdu_in);
      dada_hdu_destroy(conf.hdu_in);
    }  
  if(conf.db_out && (!conf.output_network))
    {
      dada_cuda_dbunregister(conf.hdu_out);
      dada_hdu_unlock_write(conf.hdu_out);
      dada_hdu_destroy(conf.hdu_out);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "destory hdu done");  
  
  for(i = 0; i < conf.nstream; i++)
    {
      if(conf.streams[i])
	CudaSafeCall(cudaStreamDestroy(conf.streams[i]));
    }
  if(conf.streams)
    free(conf.streams);
  log_add(conf.log_file, "INFO", 1, log_mutex, "destroy stream done");
  
  /* Cleanup GPU and return profile if it is the case */
  CudaSafeCall(cudaProfilerStop());
  CudaSafeCall(cudaDeviceReset());
  
  return EXIT_SUCCESS;
}

int read_dada_header(conf_t *conf)
{  
  uint64_t hdrsz;
  
  conf->hdrbuf_in  = ipcbuf_get_next_read(conf->hdu_in->header_block, &hdrsz);  
  if (!conf->hdrbuf_in)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting header_buf, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if(hdrsz != DADA_HDRSZ)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Header size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Header size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  
  if(ascii_header_get(conf->hdrbuf_in, "BW", "%lf", &(conf->bandwidth)) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting BW, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting BW, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      log_close(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "BW from DADA header is %f MHz", conf->bandwidth);
  
  if(ascii_header_get(conf->hdrbuf_in, "PICOSECONDS", "%"SCNu64"", &(conf->picoseconds)) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting PICOSECONDS, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting PICOSECONDS, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      log_close(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "PICOSECONDS from DADA header is %"PRIu64"", conf->picoseconds);
  
  if(ascii_header_get(conf->hdrbuf_in, "FREQ", "%lf", &(conf->center_freq)) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error egtting FREQ, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting FREQ, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      log_close(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "FREQ from DADA header is %f", conf->center_freq);
  
  if (ascii_header_get(conf->hdrbuf_in, "FILE_SIZE", "%"SCNu64"", &conf->file_size_in) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting FILE_SIZE, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }   
  log_add(conf->log_file, "INFO", 1, log_mutex, "FILE_SIZE from DADA header is %"PRIu64"", conf->file_size_in);
  
  if (ascii_header_get(conf->hdrbuf_in, "BYTES_PER_SECOND", "%"SCNu64"", &conf->bytes_per_second_in) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "BYTES_PER_SECOND from DADA header is %"PRIu64"", conf->bytes_per_second_in);
  
  if (ascii_header_get(conf->hdrbuf_in, "TSAMP", "%lf", &conf->tsamp_in) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting TSAMP, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "TSAMP from DADA header is %f", conf->tsamp_in);
  
  /* Get utc_start from hdrin */
  if (ascii_header_get(conf->hdrbuf_in, "UTC_START", "%s", conf->utc_start) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting UTC_START, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);      

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "UTC_START from DADA header is %s", conf->utc_start);

  if (ascii_header_get(conf->hdrbuf_in, "RECEIVER", "%d", &conf->beam_index) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error getting RECEIVER, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting RECEIVER, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "RECEIVER from DADA header is %d", conf->beam_index);
      
  if(ipcbuf_mark_cleared (conf->hdu_in->header_block))  // We are the only one reader, so that we can clear it after read;
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error header_clear, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error header_clear, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
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
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }  
  memcpy(hdrbuf_out, conf->hdrbuf_in, DADA_HDRSZ); // Pass the header
  
  file_size = (uint64_t)(conf->file_size_in * conf->scale_dtsz);
  bytes_per_second = (uint64_t)(conf->bytes_per_second_in * conf->scale_dtsz);
  
  if (ascii_header_set(hdrbuf_out, "NCHAN", "%d", conf->nchan_out) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting NCHAN, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error setting NCHAN, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "NCHAN to DADA header is %d", conf->nchan_out);
  
  conf->tsamp_out = conf->tsamp_in * conf->ndf_per_chunk_rbufin * NSAMP_DF * conf->nblk_accumulate;
  if (ascii_header_set(hdrbuf_out, "TSAMP", "%f", conf->tsamp_out) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting TSAMP, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error setting TSAMP, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "TSAMP to DADA header is %f microseconds", conf->tsamp_out);
  
  if (ascii_header_set(hdrbuf_out, "NBIT", "%d", NBIT_SPECTRAL) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error setting NBIT, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "NBIT to DADA header is %d", NBIT_SPECTRAL);
  
  if (ascii_header_set(hdrbuf_out, "NDIM", "%d", conf->ndim_out) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting NDIM, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error setting NDIM, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "NDIM to DADA header is %d", conf->ndim_out);
  
  if (ascii_header_set(hdrbuf_out, "NPOL", "%d", conf->npol_out) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting NPOL, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error setting NPOL, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "NPOL to DADA header is %d", conf->npol_out);
  
  if (ascii_header_set(hdrbuf_out, "FILE_SIZE", "%"PRIu64"", file_size) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: BASEBAND2SPECTRAL_ERROR:\tError setting FILE_SIZE, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "FILE_SIZE to DADA header is %"PRIu64"", file_size);
  
  if (ascii_header_set(hdrbuf_out, "BYTES_PER_SECOND", "%"PRIu64"", bytes_per_second) < 0)  
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error setting BYTES_PER_SECOND, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf->log_file, "INFO", 1, log_mutex, "BYTES_PER_SECOND to DADA header is %"PRIu64"", bytes_per_second);
  
  /* donot set header parameters anymore */
  if (ipcbuf_mark_filled (conf->hdu_out->header_block, DADA_HDRSZ) < 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error header_fill, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      destroy_baseband2spectral(*conf);
      fclose(conf->log_file);
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

  if(conf.monitor == 1)
    {   
      if(!((conf.ptype_monitor == 1) || (conf.ptype_monitor == 2) || (conf.ptype_monitor == 4)))
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: ptype_monitor should be 1, 2 or 4, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.ptype_monitor, __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1, log_mutex, "ptype_monitor should be 1, 2 or 4, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.ptype_monitor, __FILE__, __LINE__);
	  
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      else
	log_add(conf.log_file, "INFO", 1, log_mutex, "ptype_monitor is %d", conf.ptype_monitor);
            
      if(conf.port_monitor == -1)
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: monitor port shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.port_monitor, __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1, log_mutex, "monitor port shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.port_monitor, __FILE__, __LINE__);
	  
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      if(strstr(conf.ip_monitor, "unset"))
	{
	  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: monitor ip is unset, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1, log_mutex, "monitor ip is unset, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      log_add(conf.log_file, "INFO", 1, log_mutex, "We will send monitor data to %s:%d", conf.ip_monitor, conf.port_monitor); 
    }  
  else
    log_add(conf.log_file, "INFO", 1, log_mutex, "We will not send monitor data to FITSwriter interface");
  
  if(conf.ndf_per_chunk_rbufin == 0)
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: ndf_per_chunk_rbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_rbufin, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "ndf_per_chunk_rbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort", conf.ndf_per_chunk_rbufin, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "Each input ring buffer block has %"PRIu64" packets per frequency chunk", conf.ndf_per_chunk_rbufin); 

  if(conf.nblk_accumulate == -1)
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: nblk_accumulate is unset, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "nblk_accumulate is unset, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "We will average %d buffer blocks", conf.nblk_accumulate); 

  if(conf.nstream <= 0)
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: nstream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.nstream, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "nstream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.nstream, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "%d streams run on GPU", conf.nstream);
  
  if(conf.ndf_per_chunk_stream == 0)
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: ndf_per_chunk_stream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_stream, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "ndf_per_chunk_stream shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.ndf_per_chunk_stream, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "Each stream process %d packets per frequency chunk", conf.ndf_per_chunk_stream);

  log_add(conf.log_file, "INFO", 1, log_mutex, "The runtime information is %s", conf.dir);  // Checked already

  if(conf.output_network == 0)
    {
      log_add(conf.log_file, "INFO", 1, log_mutex, "We will send spectral data with ring buffer");
      if(conf.sod == 1)
	log_add(conf.log_file, "INFO", 1, log_mutex, "The spectral data is enabled at the beginning");
      else if(conf.sod == 0)
	log_add(conf.log_file, "INFO", 1, log_mutex, "The spectral data is NOT enabled at the beginning");
      else
	{
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: The sod should be 0 or 1 when we use ring buffer to send spectral data, but it is -1, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1, log_mutex, "The sod should be 0 or 1 when we use ring buffer to send spectral data, but it is -1, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      log_add(conf.log_file, "INFO", 1, log_mutex, "The key for the spectral ring buffer is %x", conf.key_out);  
    }
  if(conf.output_network == 1)
    {
      log_add(conf.log_file, "INFO", 1, log_mutex, "We will send spectral data with network interface");
      if(strstr(conf.ip, "unset"))
	{
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: We are going to send spectral data with network interface, but no ip is given, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1, log_mutex, "We are going to send spectral data with network interface, but no ip is given, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      if(conf.port == -1)
	{
	  fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: We are going to send spectral data with network interface, but no port is given, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  log_add(conf.log_file, "ERR", 1, log_mutex, "We are going to send spectral data with network interface, but no port is given, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  
	  log_close(conf.log_file);
	  exit(EXIT_FAILURE);
	}
      else
	log_add(conf.log_file, "INFO", 1, log_mutex, "The network interface for the spectral data is %s_%d", conf.ip, conf.port);  
    }
  if(conf.output_network == -1)
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: The method to send spectral data is not configured, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "The method to send spectral data is not configured, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  
  if(conf.nchunk_in<=0 || conf.nchunk_in>NCHUNK_FULL_BEAM)    
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: nchunk_in shoule be in (0 %d], but it is %d, which happens at \"%s\", line [%d], has to abort\n", NCHUNK_FULL_BEAM, conf.nchunk_in, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "nchunk_in shoule be in (0 %d], but it is %d, which happens at \"%s\", line [%d], has to abort", NCHUNK_FULL_BEAM, conf.nchunk_in, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }  
  log_add(conf.log_file, "INFO", 1, log_mutex, "%d chunks of input data", conf.nchunk_in);

  if(conf.cufft_nx<=0)    
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: cufft_nx shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.cufft_nx, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "cufft_nx shoule be a positive number, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.cufft_nx, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "We use %d points FFT", conf.cufft_nx);

  if(!((conf.pol_type == 1) || (conf.pol_type == 2) || (conf.pol_type == 4)))
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: pol_type should be 1, 2 or 4, but it is %d, which happens at \"%s\", line [%d], has to abort\n", conf.pol_type, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "pol_type should be 1, 2 or 4, but it is %d, which happens at \"%s\", line [%d], has to abort", conf.pol_type, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "pol_type is %d", conf.pol_type, __FILE__, __LINE__);
  log_add(conf.log_file, "INFO", 1, log_mutex, "npol_out is %d", conf.npol_out, __FILE__, __LINE__);
  log_add(conf.log_file, "INFO", 1, log_mutex, "ndim_out is %d", conf.ndim_out, __FILE__, __LINE__);
  
  return EXIT_SUCCESS;
}