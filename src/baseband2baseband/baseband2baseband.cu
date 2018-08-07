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

int init_process(conf_t *conf)
{
  CudaSafeCall(cudaSetDevice(conf->device_id));
  
  int i;
  int iembed1, istride1, idist1, oembed1, ostride1, odist1, batch1, nx1;
  int iembed2, istride2, idist2, oembed2, ostride2, odist2, batch2, nx2;
  ipcbuf_t *db = NULL;
  
  /* Prepare buffer, stream and fft plan for process */
  conf->sclndim_fold = conf->rbufin_ndf * NSAMP_DF * NPOL_SAMP * NDIM_POL; // Only works when two polarisations has similar power level
  conf->nsamp1       = conf->stream_ndf * NCHK_NIC * NCHAN_CHK * NSAMP_DF;
  conf->npol1        = conf->nsamp1 * NPOL_SAMP;
  conf->ndata1       = conf->npol1  * NDIM_POL;
		     
  conf->nsamp2       = conf->nsamp1 * OSAMP_RATEI / NCHAN_RATEI;
  conf->npol2        = conf->nsamp2 * NPOL_SAMP;
  conf->ndata2       = conf->npol2  * NDIM_POL;

  conf->sclndim_search = conf->rbufin_ndf * NSAMP_DF / CUFFT_NX1;   // We do not average in time and here we work on detected data;
  conf->nsamp3         = conf->nsamp2 * NCHAN_SEARCH / NCHAN_KEEP2;
  conf->npol3          = conf->nsamp3;
  conf->ndata3         = conf->nsamp3;  
    
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
  
  conf->sbufin_size         = conf->ndata1 * NBYTE_IN;
  conf->sbufout_size_fold   = conf->ndata2 * NBYTE_OUT_FOLD;
  conf->sbufout_size_search = conf->ndata3 * NBYTE_OUT_SEARCH;
  
  conf->bufin_size          = conf->nstream * conf->sbufin_size;
  conf->bufout_size_fold    = conf->nstream * conf->sbufout_size_fold;
  conf->bufout_size_search  = conf->nstream * conf->sbufout_size_search;
  
  conf->sbufrt1_size = conf->npol1 * NBYTE_RT;
  conf->sbufrt2_size = conf->npol2 * NBYTE_RT;
  conf->bufrt1_size  = conf->nstream * conf->sbufrt1_size;
  conf->bufrt2_size  = conf->nstream * conf->sbufrt2_size;
    
  //conf->hbufin_offset = conf->sbufin_size / sizeof(char);
  conf->hbufin_offset = conf->sbufin_size;
  conf->dbufin_offset = conf->sbufin_size / sizeof(int64_t);
  conf->bufrt1_offset = conf->sbufrt1_size / sizeof(cufftComplex);
  conf->bufrt2_offset = conf->sbufrt2_size / sizeof(cufftComplex);
  
  conf->dbufout_offset_fold   = conf->sbufout_size_fold / NBYTE_OUT_FOLD;
  //conf->hbufout_offset_fold   = conf->sbufout_size_fold / sizeof(char);
  conf->hbufout_offset_fold   = conf->sbufout_size_fold;
  conf->dbufout_offset_search = conf->sbufout_size_search / NBYTE_OUT_SEARCH;
  //conf->hbufout_offset_search = conf->sbufout_size_search / sizeof(char);
  conf->hbufout_offset_search = conf->sbufout_size_search;

  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_in, conf->bufin_size));
  
#ifdef FOLD_MODE
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out_fold, conf->bufout_size_fold));       
  CudaSafeCall(cudaMalloc((void **)&conf->ddat_offs_fold, NCHAN_FOLD * sizeof(float)));
  CudaSafeCall(cudaMalloc((void **)&conf->dsquare_mean_fold, NCHAN_FOLD * sizeof(float)));
  CudaSafeCall(cudaMalloc((void **)&conf->ddat_scl_fold, NCHAN_FOLD * sizeof(float)));
      
  CudaSafeCall(cudaMemset((void *)conf->ddat_offs_fold, 0, NCHAN_FOLD * sizeof(float)));   // We have to clear the memory for this parameter
  CudaSafeCall(cudaMemset((void *)conf->dsquare_mean_fold, 0, NCHAN_FOLD * sizeof(float)));// We have to clear the memory for this parameter
  
  CudaSafeCall(cudaMallocHost((void **)&conf->hdat_scl_fold, NCHAN_FOLD * sizeof(float)));   // Malloc host memory to receive data from device
  CudaSafeCall(cudaMallocHost((void **)&conf->hdat_offs_fold, NCHAN_FOLD * sizeof(float)));   // Malloc host memory to receive data from device
  CudaSafeCall(cudaMallocHost((void **)&conf->hsquare_mean_fold, NCHAN_FOLD * sizeof(float)));   // Malloc host memory to receive data from device
#else
  CudaSafeCall(cudaMalloc((void **)&conf->dbuf_out_search, conf->bufout_size_search));
  CudaSafeCall(cudaMalloc((void **)&conf->ddat_offs_search, NCHAN_SEARCH * sizeof(float)));
  CudaSafeCall(cudaMalloc((void **)&conf->dsquare_mean_search, NCHAN_SEARCH * sizeof(float)));
  CudaSafeCall(cudaMalloc((void **)&conf->ddat_scl_search, NCHAN_SEARCH * sizeof(float)));
  
  CudaSafeCall(cudaMemset((void *)conf->ddat_offs_search, 0, NCHAN_SEARCH * sizeof(float)));   // We have to clear the memory for this parameter
  CudaSafeCall(cudaMemset((void *)conf->dsquare_mean_search, 0, NCHAN_SEARCH * sizeof(float)));// We have to clear the memory for this parameter
  
  CudaSafeCall(cudaMallocHost((void **)&conf->hdat_scl_search, NCHAN_SEARCH * sizeof(float)));   // Malloc host memory to receive data from device
  CudaSafeCall(cudaMallocHost((void **)&conf->hdat_offs_search, NCHAN_SEARCH * sizeof(float)));   // Malloc host memory to receive data from device
  CudaSafeCall(cudaMallocHost((void **)&conf->hsquare_mean_search, NCHAN_SEARCH * sizeof(float)));   // Malloc host memory to receive data from device
#endif
  
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt1, conf->bufrt1_size));
  CudaSafeCall(cudaMalloc((void **)&conf->buf_rt2, conf->bufrt2_size)); 

  /* Prepare the setup of kernels */
  conf->gridsize_unpack.x = conf->stream_ndf;
  conf->gridsize_unpack.y = NCHK_NIC;
  conf->gridsize_unpack.z = 1;
  conf->blocksize_unpack.x = NSAMP_DF; 
  conf->blocksize_unpack.y = NCHAN_CHK;
  conf->blocksize_unpack.z = 1;
  
  conf->gridsize_swap_select_transpose.x = NCHK_NIC * NCHAN_CHK;
  conf->gridsize_swap_select_transpose.y = conf->stream_ndf * NSAMP_DF / CUFFT_NX1;
  conf->gridsize_swap_select_transpose.z = 1;  
  conf->blocksize_swap_select_transpose.x = CUFFT_NX1;
  conf->blocksize_swap_select_transpose.y = 1;
  conf->blocksize_swap_select_transpose.z = 1;

  conf->gridsize_swap_select_transpose_swap.x = NCHK_NIC * NCHAN_CHK;
  conf->gridsize_swap_select_transpose_swap.y = conf->stream_ndf * NSAMP_DF / CUFFT_NX1;
  conf->gridsize_swap_select_transpose_swap.z = 1;  
  conf->blocksize_swap_select_transpose_swap.x = CUFFT_NX1;
  conf->blocksize_swap_select_transpose_swap.y = 1;
  conf->blocksize_swap_select_transpose_swap.z = 1;
      
  conf->gridsize_swap.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_swap.y = NCHAN_FOLD;
  conf->gridsize_swap.z = 1;
  conf->blocksize_swap.x = CUFFT_NX2;
  conf->blocksize_swap.y = 1;
  conf->blocksize_swap.z = 1;
  
  conf->gridsize_mean_fold.x = 1; 
  conf->gridsize_mean_fold.y = 1; 
  conf->gridsize_mean_fold.z = 1;
  conf->blocksize_mean_fold.x = NCHAN_FOLD; 
  conf->blocksize_mean_fold.y = 1;
  conf->blocksize_mean_fold.z = 1;
  
  conf->gridsize_scale_fold.x = 1;
  conf->gridsize_scale_fold.y = 1;
  conf->gridsize_scale_fold.z = 1;
  conf->blocksize_scale_fold.x = NCHAN_FOLD;
  conf->blocksize_scale_fold.y = 1;
  conf->blocksize_scale_fold.z = 1;
  
  conf->gridsize_transpose_scale.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_scale.y = NCHAN_FOLD;
  conf->gridsize_transpose_scale.z = 1;
  conf->blocksize_transpose_scale.x = CUFFT_NX2;
  conf->blocksize_transpose_scale.y = 1;
  conf->blocksize_transpose_scale.z = 1;
  
  conf->gridsize_transpose_pad.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_pad.y = NCHAN_FOLD;
  conf->gridsize_transpose_pad.z = 1;
  conf->blocksize_transpose_pad.x = CUFFT_NX2;
  conf->blocksize_transpose_pad.y = 1;
  conf->blocksize_transpose_pad.z = 1;

  conf->gridsize_sum1_fold.x = NCHAN_FOLD;
  conf->gridsize_sum1_fold.y = conf->stream_ndf * NPOL_SAMP;
  conf->gridsize_sum1_fold.z = 1;
  //conf->blocksize_sum1_fold.x = NSAMP_DF / 2;  // Here I assume that CUFFT_NX1 is equal to CUFFT_NX2, which is not true in some cases
  conf->blocksize_sum1_fold.x = NSAMP_DF * CUFFT_NX2 / (2 * CUFFT_NX1);  // This is the right setup if CUFFT_NX2 is not equal to CUFFT_NX1
  conf->blocksize_sum1_fold.y = 1;
  conf->blocksize_sum1_fold.z = 1;
  
  conf->gridsize_sum2_fold.x = NCHAN_FOLD;
  conf->gridsize_sum2_fold.y = 1;
  conf->gridsize_sum2_fold.z = 1;
  conf->blocksize_sum2_fold.x = conf->stream_ndf * NPOL_SAMP / 2;
  conf->blocksize_sum2_fold.y = 1;
  conf->blocksize_sum2_fold.z = 1;
  
  conf->gridsize_transpose_scale2.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_scale2.y = 1;
  conf->gridsize_transpose_scale2.z = 1;
  conf->blocksize_transpose_scale2.x = CUFFT_NX2;
  conf->blocksize_transpose_scale2.y = CUFFT_NX2;
  conf->blocksize_transpose_scale2.z = 1;
  
  conf->gridsize_transpose_scale3.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_scale3.y = NCHAN_FOLD / TILE_DIM;
  conf->gridsize_transpose_scale3.z = 1;
  conf->blocksize_transpose_scale3.x = TILE_DIM;
  conf->blocksize_transpose_scale3.y = NROWBLOCK_TRANS;
  conf->blocksize_transpose_scale3.z = 1;
  
  conf->gridsize_transpose_scale4.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_scale4.y = NCHAN_FOLD / TILE_DIM;
  conf->gridsize_transpose_scale4.z = 1;
  conf->blocksize_transpose_scale4.x = TILE_DIM;
  conf->blocksize_transpose_scale4.y = NROWBLOCK_TRANS;
  conf->blocksize_transpose_scale4.z = 1;
  
  conf->gridsize_transpose_float.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1; 
  conf->gridsize_transpose_float.y = NCHAN_FOLD / TILE_DIM;
  conf->gridsize_transpose_float.z = 1;
  conf->blocksize_transpose_float.x = TILE_DIM;
  conf->blocksize_transpose_float.y = NROWBLOCK_TRANS;
  conf->blocksize_transpose_float.z = 1;

  /* Only for search mode */
  conf->gridsize_add_detect_scale.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1;
  conf->gridsize_add_detect_scale.y = NCHAN_SEARCH;
  conf->gridsize_add_detect_scale.z = 1;
  conf->blocksize_add_detect_scale.x = NCHAN_KEEP2/(2 * NCHAN_SEARCH);
  conf->blocksize_add_detect_scale.y = 1;
  conf->blocksize_add_detect_scale.z = 1;
  
  conf->gridsize_add_detect_pad.x = conf->stream_ndf * NSAMP_DF / CUFFT_NX1;
  conf->gridsize_add_detect_pad.y = NCHAN_SEARCH;
  conf->gridsize_add_detect_pad.z = 1;
  conf->blocksize_add_detect_pad.x = NCHAN_KEEP2/(2 * NCHAN_SEARCH);
  conf->blocksize_add_detect_pad.y = 1;
  conf->blocksize_add_detect_pad.z = 1;
  
  conf->gridsize_sum_search.x = NCHAN_SEARCH;
  conf->gridsize_sum_search.y = 1;
  conf->gridsize_sum_search.z = 1;
  conf->blocksize_sum_search.x = conf->stream_ndf * NSAMP_DF / (CUFFT_NX1 * 2); 
  conf->blocksize_sum_search.y = 1;
  conf->blocksize_sum_search.z = 1;
  
  conf->gridsize_mean_search.x = 1; 
  conf->gridsize_mean_search.y = 1; 
  conf->gridsize_mean_search.z = 1;
  conf->blocksize_mean_search.x = NCHAN_SEARCH;
  conf->blocksize_mean_search.y = 1;
  conf->blocksize_mean_search.z = 1;
  
  conf->gridsize_scale_search.x = 1;
  conf->gridsize_scale_search.y = 1;
  conf->gridsize_scale_search.z = 1;
  conf->blocksize_scale_search.x = NCHAN_SEARCH;
  conf->blocksize_scale_search.y = 1;
  conf->blocksize_scale_search.z = 1;
  
  /* attach to input ring buffer */
  conf->hdu_in = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu_in, conf->key_in);
  if(dada_hdu_connect(conf->hdu_in) < 0)
    {
      multilog(runtime_log, LOG_ERR, "could not connect to hdu\n");
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }  
  db = (ipcbuf_t *) conf->hdu_in->data_block;
  conf->rbufin_size = ipcbuf_get_bufsz(db);  
  if(conf->rbufin_size % conf->bufin_size != 0)  
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  
  /* registers the existing host memory range for use by CUDA */
  dada_cuda_dbregister(conf->hdu_in);
        
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_in->header_block);  
  if(conf->hdrsz != DADA_HDR_SIZE)    // This number should match
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
#ifdef FOLD_MODE
  conf->hdu_out_fold = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu_out_fold, conf->key_out_fold);
  if(dada_hdu_connect(conf->hdu_out_fold) < 0)
    {
      multilog(runtime_log, LOG_ERR, "could not connect to hdu\n");
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  db = (ipcbuf_t *) conf->hdu_out_fold->data_block;
  conf->rbufout_size_fold = ipcbuf_get_bufsz(db);
  if(conf->rbufout_size_fold % conf->bufout_size_fold != 0)  
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_out_fold->header_block);  
  if(conf->hdrsz != DADA_HDR_SIZE)    // This number should match
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }  
  /* make ourselves the write client */
  if(dada_hdu_lock_write(conf->hdu_out_fold) < 0)
    {
      multilog(runtime_log, LOG_ERR, "open_hdu: could not lock write\n");
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
#else
  conf->hdu_out_search = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu_out_search, conf->key_out_search);
  if(dada_hdu_connect(conf->hdu_out_search) < 0)
    {
      multilog(runtime_log, LOG_ERR, "could not connect to hdu\n");
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  db = (ipcbuf_t *) conf->hdu_out_search->data_block;
  conf->rbufout_size_search = ipcbuf_get_bufsz(db);
  if(conf->rbufout_size_search % conf->bufout_size_search != 0)  
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  
  conf->hdrsz = ipcbuf_get_bufsz(conf->hdu_out_search->header_block);  
  if(conf->hdrsz != DADA_HDR_SIZE)    // This number should match
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }  
  /* make ourselves the write client */
  if(dada_hdu_lock_write(conf->hdu_out_search) < 0)
    {
      multilog(runtime_log, LOG_ERR, "open_hdu: could not lock write\n");
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
#endif
  
  if(conf->sod)
    {      
      if(ipcbuf_enable_sod(db, 0, 0) < 0)  // We start at the beginning
  	{
	  multilog(runtime_log, LOG_ERR, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  	  fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  	  return EXIT_FAILURE;
  	}
    }
  else
    {
      if(ipcbuf_disable_sod(db) < 0)
  	{
	  multilog(runtime_log, LOG_ERR, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  	  fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  	  return EXIT_FAILURE;
  	}
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

int do_process(conf_t conf)
{
  CudaSafeCall(cudaSetDevice(conf.device_id));
  
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
    4. Add the data in frequency to get NCHAN_SEARCH channels, detect the added data and scale it;
  */
  size_t i, j;
  size_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset, hbufout_offset_fold, dbufout_offset_fold, hbufout_offset_search, dbufout_offset_search;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap;
  dim3 gridsize_swap_select_transpose, blocksize_swap_select_transpose;
  dim3 gridsize_transpose_scale3, blocksize_transpose_scale3;
  dim3 gridsize_transpose_scale4, blocksize_transpose_scale4;
  dim3 gridsize_transpose_float, blocksize_transpose_float;
  dim3 gridsize_add_detect_scale, blocksize_add_detect_scale;
  uint64_t block_id = 0;
  size_t curbufsz;
  
  gridsize_unpack                      = conf.gridsize_unpack;
  blocksize_unpack                     = conf.blocksize_unpack;
  gridsize_swap_select_transpose_swap  = conf.gridsize_swap_select_transpose_swap;   
  blocksize_swap_select_transpose_swap = conf.blocksize_swap_select_transpose_swap;  
  gridsize_transpose_scale3            = conf.gridsize_transpose_scale3;
  blocksize_transpose_scale3           = conf.blocksize_transpose_scale3; 
  gridsize_transpose_scale4            = conf.gridsize_transpose_scale4;
  blocksize_transpose_scale4           = conf.blocksize_transpose_scale4;
  gridsize_transpose_float             = conf.gridsize_transpose_float;
  blocksize_transpose_float            = conf.blocksize_transpose_float;
  gridsize_add_detect_scale            = conf.gridsize_add_detect_scale ;
  blocksize_add_detect_scale           = conf.blocksize_add_detect_scale ;
  gridsize_swap_select_transpose       = conf.gridsize_swap_select_transpose;   
  blocksize_swap_select_transpose      = conf.blocksize_swap_select_transpose;  
  
  /* Get scale of data */
  dat_offs_scl(conf);
#ifdef DEBUG
#ifdef FOLD_MODE
  for(i = 0; i < NCHAN_FOLD; i++)
    fprintf(stdout, "DAT_OFFS:\t%E\tDAT_SCL:\t%E\n", conf.hdat_offs_fold[i], conf.hdat_scl_fold[i]);
#else
  for(i = 0; i < NCHAN_SEARCH; i++)
    fprintf(stdout, "DAT_OFFS:\t%E\tDAT_SCL:\t%E\n", conf.hdat_offs_search[i], conf.hdat_scl_search[i]);
#endif
#endif
  
  /* Do the real job */
#ifdef DEBUG
  double elapsed_time;
  struct timespec start, stop;
  clock_gettime(CLOCK_REALTIME, &start);
#endif

#ifdef FOLD_MODE
  conf.hdu_out_fold->data_block->curbuf = ipcio_open_block_write(conf.hdu_out_fold->data_block, &block_id);   /* Open buffer to write */
#else
  conf.hdu_out_search->data_block->curbuf = ipcio_open_block_write(conf.hdu_out_search->data_block, &block_id);   /* Open buffer to write */
#endif
  
  while(conf.hdu_in->data_block->curbufsz == conf.rbufin_size)
    // The first time we open a block at the scale calculation, we need to make sure that the input ring buffer block is bigger than the block needed for scale calculation
    // Otherwise we have to open couple of blocks to calculate scales and these blocks will dropped after that
    {
      //for(i = 0; i < conf.rbufin_size; i += conf.bufin_size)
      for(i = 0; i < conf.nrun_blk; i ++)
	{
#ifdef DEBUG
	  clock_gettime(CLOCK_REALTIME, &stop);
	  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000000.0L;
	  fprintf(stdout, "elapsed time to read %d data frame steps is %f s\n", conf.stream_ndf * conf.nstream, elapsed_time);
#endif
	  
#ifdef DEBUG
	  float elapsed_event;
	  cudaEvent_t start_event, stop_event;
	  CudaSafeCall(cudaEventCreate(&start_event));
	  CudaSafeCall(cudaEventCreate(&stop_event));
	  CudaSafeCall(cudaEventRecord(start_event));
#endif
	  //fprintf(stdout, "REPEAT HERE\n\n");
	  for(j = 0; j < conf.nstream; j++)
	    {
	      //fprintf(stdout, "STREAM HERE 1\t");
	      hbufin_offset = j * conf.hbufin_offset + i * conf.bufin_size;
	      dbufin_offset = j * conf.dbufin_offset; 
	      bufrt1_offset = j * conf.bufrt1_offset;
	      bufrt2_offset = j * conf.bufrt2_offset;
#ifdef FOLD_MODE
	      dbufout_offset_fold = j * conf.dbufout_offset_fold;
	      hbufout_offset_fold = j * conf.hbufout_offset_fold + i * conf.bufout_size_fold;
#else
	      dbufout_offset_search = j * conf.dbufout_offset_search;
	      hbufout_offset_search = j * conf.hbufout_offset_search + i * conf.bufout_size_search;
#endif
	      
	      /* Copy data into device */
#ifdef DEBUG
	      clock_gettime(CLOCK_REALTIME, &start);
#endif 
	      CudaSafeCall(cudaMemcpyAsync(&conf.dbuf_in[dbufin_offset], &conf.hdu_in->data_block->curbuf[hbufin_offset], conf.sbufin_size, cudaMemcpyHostToDevice, conf.streams[j]));
	      
#ifdef DEBUG
	      clock_gettime(CLOCK_REALTIME, &stop);
	      elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000000.0L;
	      fprintf(stdout, "elapsed time to memcpy %d data frame steps is %f s\n", conf.stream_ndf, elapsed_time);
#endif
	      
	      /* Unpack raw data into cufftComplex array */
	      unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, conf.streams[j]>>>(&conf.dbuf_in[dbufin_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp1);
	      
	      /* Do forward FFT */
	      CufftSafeCall(cufftExecC2C(conf.fft_plans1[j], &conf.buf_rt1[bufrt1_offset], &conf.buf_rt1[bufrt1_offset], CUFFT_FORWARD));

#ifdef FOLD_MODE
	      /* Prepare for inverse FFT */
	      swap_select_transpose_swap_kernel<<<gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2); 
	      /* Do inverse FFT */
	      CufftSafeCall(cufftExecC2C(conf.fft_plans2[j], &conf.buf_rt2[bufrt2_offset], &conf.buf_rt2[bufrt2_offset], CUFFT_INVERSE));
	      /* Get final output */
	      transpose_scale_kernel4<<<gridsize_transpose_scale4, blocksize_transpose_scale4, 0, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out_fold[dbufout_offset_fold], conf.nsamp2, conf.ddat_offs_fold, conf.ddat_scl_fold);   
	      /* Copy the final output to host */
	      CudaSafeCall(cudaMemcpyAsync(&conf.hdu_out_fold->data_block->curbuf[hbufout_offset_fold], &conf.dbuf_out_fold[dbufout_offset_fold], conf.sbufout_size_fold, cudaMemcpyDeviceToHost, conf.streams[j]));
#else
	      swap_select_transpose_kernel<<<gridsize_swap_select_transpose, blocksize_swap_select_transpose, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2); 		  
	      add_detect_scale_kernel<<<gridsize_add_detect_scale, blocksize_add_detect_scale, blocksize_add_detect_scale.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.dbuf_out_search[dbufout_offset_search], conf.nsamp2, conf.ddat_offs_search, conf.ddat_scl_search);
	      CudaSafeCall(cudaMemcpyAsync(&conf.hdu_out_search->data_block->curbuf[hbufout_offset_search], &conf.dbuf_out_search[dbufout_offset_search], conf.sbufout_size_search, cudaMemcpyDeviceToHost, conf.streams[j]));
#endif
	      //fprintf(stdout, "STREAM HERE 2\n");
	    }
	  CudaSynchronizeCall(); // Sync here is for multiple streams
	  
#ifdef DEBUG
	  CudaSafeCall(cudaEventRecord(stop_event));
	  CudaSafeCall(cudaEventSynchronize(stop_event));
	  CudaSafeCall(cudaEventElapsedTime(&elapsed_event, start_event, stop_event));
	  fprintf(stdout, "elapsed time for GPU process of %d data frame steps is %f s\n", conf.stream_ndf * conf.nstream, elapsed_event/1.0E3);
#endif
	}
      
#ifdef DEBUG
      clock_gettime(CLOCK_REALTIME, &start);
#endif      
	  
      /* Close current buffer */
#ifdef FOLD_MODE
      if(ipcio_close_block_write(conf.hdu_out_fold->data_block, conf.rbufout_size_fold) < 0)
	{
	  multilog (runtime_log, LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
	  fprintf(stderr, "close_buffer: ipcio_close_block_write failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
      conf.hdu_out_fold->data_block->curbuf = ipcio_open_block_write(conf.hdu_out_fold->data_block, &block_id);   /* Open buffer to write */
#else
      if(ipcio_close_block_write(conf.hdu_out_search->data_block, conf.rbufout_size_search) < 0)
	{
	  multilog (runtime_log, LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
	  fprintf(stderr, "close_buffer: ipcio_close_block_write failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
      conf.hdu_out_search->data_block->curbuf = ipcio_open_block_write(conf.hdu_out_search->data_block, &block_id);   /* Open buffer to write */
#endif
#ifdef DEBUG
      clock_gettime(CLOCK_REALTIME, &stop);
      elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000000.0L;
      fprintf(stdout, "elapsed time to write %.0f data frame steps is %f\n\n", conf.rbufin_ndf , elapsed_time);
#endif
#ifdef DEBUG
      clock_gettime(CLOCK_REALTIME, &start);
#endif
      
      ipcio_close_block_read(conf.hdu_in->data_block, conf.hdu_in->data_block->curbufsz);
      conf.hdu_in->data_block->curbuf = ipcio_open_block_read(conf.hdu_in->data_block, &curbufsz, &block_id);
    }

  ipcio_close_block_read(conf.hdu_in->data_block, conf.hdu_in->data_block->curbufsz);
#ifdef FOLD_MODE
  if (ipcio_close_block_write(conf.hdu_out_fold->data_block, conf.rbufout_size_fold) < 0)
    {
      multilog (runtime_log, LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
      fprintf(stderr, "close_buffer: ipcio_close_block_write failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
#else
  if (ipcio_close_block_write(conf.hdu_out_search->data_block, conf.rbufout_size_search) < 0)
    {
      multilog (runtime_log, LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
      fprintf(stderr, "close_buffer: ipcio_close_block_write failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
#endif

  return EXIT_SUCCESS;
}

int dat_offs_scl(conf_t conf)
{
  CudaSafeCall(cudaSetDevice(conf.device_id));

  /*
    The procedure for fold mode is:
    1. Get PTFT data as we did at process;
    2. Pad the data;
    3. Add the padded data in time;
    4. Get the mean of the added data;
    5. Get the scale with the mean;

    The procedure for search mode is:
    1. Get PTF data as we did at process;
    2. Add the data in frequency to get NCHAN_SEARCH channels, detect the added data and pad it;
    3. Add the padded data in time;    
    4. Get the mean of the added data;
    5. Get the scale with the mean;
  */
  size_t i, j;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap;
  dim3 gridsize_scale_fold, blocksize_scale_fold;  
  dim3 gridsize_mean_fold, blocksize_mean_fold;
  dim3 gridsize_sum1_fold, blocksize_sum1_fold;
  dim3 gridsize_sum2_fold, blocksize_sum2_fold;
  dim3 gridsize_transpose_pad, blocksize_transpose_pad;
  size_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset;
  size_t curbufsz, block_id;

  dim3 gridsize_swap_select_transpose, blocksize_swap_select_transpose;
  dim3 gridsize_scale_search, blocksize_scale_search;  
  dim3 gridsize_mean_search, blocksize_mean_search;
  dim3 gridsize_sum_search, blocksize_sum_search;
  dim3 gridsize_add_detect_pad, blocksize_add_detect_pad;
  
  char fname[MSTR_LEN];
  FILE *fp=NULL;
    
  gridsize_unpack                      = conf.gridsize_unpack;
  blocksize_unpack                     = conf.blocksize_unpack;
  gridsize_swap_select_transpose_swap  = conf.gridsize_swap_select_transpose_swap;   
  blocksize_swap_select_transpose_swap = conf.blocksize_swap_select_transpose_swap; 
  gridsize_transpose_pad               = conf.gridsize_transpose_pad;
  blocksize_transpose_pad              = conf.blocksize_transpose_pad;
  	         	               						       
  gridsize_sum1_fold              = conf.gridsize_sum1_fold;	       
  blocksize_sum1_fold             = conf.blocksize_sum1_fold;
  gridsize_sum2_fold              = conf.gridsize_sum2_fold;	       
  blocksize_sum2_fold             = conf.blocksize_sum2_fold;
  gridsize_scale_fold             = conf.gridsize_scale_fold;	       
  blocksize_scale_fold            = conf.blocksize_scale_fold;	         					    
  gridsize_mean_fold              = conf.gridsize_mean_fold;	       
  blocksize_mean_fold             = conf.blocksize_mean_fold;

  gridsize_sum_search             = conf.gridsize_sum_search;	       
  blocksize_sum_search            = conf.blocksize_sum_search;
  gridsize_scale_search           = conf.gridsize_scale_search;	       
  blocksize_scale_search          = conf.blocksize_scale_search;	         		           
  gridsize_mean_search            = conf.gridsize_mean_search;	       
  blocksize_mean_search           = conf.blocksize_mean_search;
  gridsize_swap_select_transpose  = conf.gridsize_swap_select_transpose;   
  blocksize_swap_select_transpose = conf.blocksize_swap_select_transpose;
  gridsize_add_detect_pad         = conf.gridsize_add_detect_pad ;
  blocksize_add_detect_pad        = conf.blocksize_add_detect_pad ;
  
  conf.hdu_in->data_block->curbuf = ipcio_open_block_read(conf.hdu_in->data_block, &curbufsz, &block_id);
  if(conf.hdu_in->data_block->curbuf == NULL)
    {
      multilog (runtime_log, LOG_ERR, "Can not get buffer block from input ring buffer, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not get buffer block from input ring buffer, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
#ifdef DEBUG
  float elapsed_event;
  cudaEvent_t start_event, stop_event;
  CudaSafeCall(cudaEventCreate(&start_event));
  CudaSafeCall(cudaEventCreate(&stop_event));
  CudaSafeCall(cudaEventRecord(start_event));
#endif
  
  for(i = 0; i < conf.rbufin_size; i += conf.bufin_size)
    {
      for (j = 0; j < conf.nstream; j++)
	{
	  hbufin_offset = j * conf.hbufin_offset + i;
	  dbufin_offset = j * conf.dbufin_offset; 
	  bufrt1_offset = j * conf.bufrt1_offset;
	  bufrt2_offset = j * conf.bufrt2_offset;
	  
	  /* Copy data into device */
	  CudaSafeCall(cudaMemcpyAsync(&conf.dbuf_in[dbufin_offset], &conf.hdu_in->data_block->curbuf[hbufin_offset], conf.sbufin_size, cudaMemcpyHostToDevice, conf.streams[j]));

	  /* Unpack raw data into cufftComplex array */
	  unpack_kernel<<<gridsize_unpack, blocksize_unpack, 0, conf.streams[j]>>>(&conf.dbuf_in[dbufin_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp1);

	  /* Do forward FFT */
	  CufftSafeCall(cufftExecC2C(conf.fft_plans1[j], &conf.buf_rt1[bufrt1_offset], &conf.buf_rt1[bufrt1_offset], CUFFT_FORWARD));

#ifdef FOLD_MODE
	  /* Prepare for inverse FFT */
	  swap_select_transpose_swap_kernel<<<gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2); 
	  
	  /* Do inverse FFT */
	  CufftSafeCall(cufftExecC2C(conf.fft_plans2[j], &conf.buf_rt2[bufrt2_offset], &conf.buf_rt2[bufrt2_offset], CUFFT_INVERSE));
	  
	  /* Transpose the data from PTFT to FTP for later calculation */
	  transpose_pad_kernel<<<gridsize_transpose_pad, blocksize_transpose_pad, 0, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], conf.nsamp2, &conf.buf_rt1[bufrt1_offset]);
	  
	  /* Get the sum of samples and square of samples */
	  sum_kernel<<<gridsize_sum1_fold, blocksize_sum1_fold, blocksize_sum1_fold.x * sizeof(cufftComplex), conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset]);
	  sum_kernel<<<gridsize_sum2_fold, blocksize_sum2_fold, blocksize_sum2_fold.x * sizeof(cufftComplex), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset]);
#else
	  swap_select_transpose_kernel<<<gridsize_swap_select_transpose, blocksize_swap_select_transpose, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2); 		  
	  add_detect_pad_kernel<<<gridsize_add_detect_pad, blocksize_add_detect_pad, blocksize_add_detect_pad.x * sizeof(float), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset], conf.nsamp2);
	  sum_kernel<<<gridsize_sum_search, blocksize_sum_search, blocksize_sum_search.x * sizeof(cufftComplex), conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset]);
#endif
	}
      CudaSynchronizeCall(); // Sync here is for multiple streams

#ifdef FOLD_MODE
      mean_kernel<<<gridsize_mean_fold, blocksize_mean_fold>>>(conf.buf_rt1, conf.bufrt1_offset, conf.ddat_offs_fold, conf.dsquare_mean_fold, conf.nstream, conf.sclndim_fold);
#else
      mean_kernel<<<gridsize_mean_search, blocksize_mean_search>>>(conf.buf_rt2, conf.bufrt2_offset, conf.ddat_offs_search, conf.dsquare_mean_search, conf.nstream, conf.sclndim_search);
#endif
    }
  /* Get the scale of each chanel */
#ifdef FOLD_MODE
  scale_kernel<<<gridsize_scale_fold, blocksize_scale_fold>>>(conf.ddat_offs_fold, conf.dsquare_mean_fold, conf.ddat_scl_fold);
#else
  scale_kernel<<<gridsize_scale_search, blocksize_scale_search>>>(conf.ddat_offs_search, conf.dsquare_mean_search, conf.ddat_scl_search);
#endif
  CudaSynchronizeCall();
  
#ifdef DEBUG
  CudaSafeCall(cudaEventRecord(stop_event));
  CudaSafeCall(cudaEventSynchronize(stop_event));
  CudaSafeCall(cudaEventElapsedTime(&elapsed_event, start_event, stop_event));
#ifdef FOLD_MODE
  fprintf(stdout, "elapsed time to get scale with %.0f data is %f s\n", conf.sclndim_fold, elapsed_event/1.0E3);
#else
  fprintf(stdout, "elapsed time to get scale with %.0f data is %f s\n", conf.sclndim_search, elapsed_event/1.0E3);
#endif
#endif
#ifdef FOLD_MODE
  CudaSafeCall(cudaMemcpy(conf.hdat_offs_fold, conf.ddat_offs_fold, sizeof(float) * NCHAN_FOLD, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(conf.hdat_scl_fold, conf.ddat_scl_fold, sizeof(float) * NCHAN_FOLD, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(conf.hsquare_mean_fold, conf.dsquare_mean_fold, sizeof(float) * NCHAN_FOLD, cudaMemcpyDeviceToHost));
#else
  CudaSafeCall(cudaMemcpy(conf.hdat_offs_search, conf.ddat_offs_search, sizeof(float) * NCHAN_SEARCH, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(conf.hdat_scl_search, conf.ddat_scl_search, sizeof(float) * NCHAN_SEARCH, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(conf.hsquare_mean_search, conf.dsquare_mean_search, sizeof(float) * NCHAN_SEARCH, cudaMemcpyDeviceToHost));
#endif
 
#ifdef DEBUG
#ifdef FOLD_MODE
  for (i = 0; i< NCHAN_FOLD; i++)
    fprintf(stdout, "DAT_OFFS:\t%E\tDAT_SCL:\t%E\n", conf.hdat_offs_fold[i], conf.hdat_scl_fold[i]);
#else
  for (i = 0; i< NCHAN_SEARCH; i++)
    fprintf(stdout, "DAT_OFFS:\t%E\tDAT_SCL:\t%E\n", conf.hdat_offs_search[i], conf.hdat_scl_search[i]);
#endif
#endif
  /* Record scale into file */
  sprintf(fname, "%s/%s_scale.txt", conf.dir, conf.utc_start);
  fp = fopen(fname, "w");
  if(fp == NULL)
    {
      multilog (runtime_log, LOG_ERR, "Can not open scale file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not open scale file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

#ifdef FOLD_MODE
  for (i = 0; i< NCHAN_FOLD; i++)
    fprintf(fp, "%E\t%E\n", conf.hdat_offs_fold[i], conf.hdat_scl_fold[i]);
#else
  for (i = 0; i< NCHAN_SEARCH; i++)
    fprintf(fp, "%E\t%E\n", conf.hdat_offs_search[i], conf.hdat_scl_search[i]);
#endif
  
  fclose(fp);
  return EXIT_SUCCESS;
}

int destroy_process(conf_t conf)
{
  int i;
  CudaSafeCall(cudaSetDevice(conf.device_id));
  
  for (i = 0; i < conf.nstream; i++)
    {
      CudaSafeCall(cudaStreamDestroy(conf.streams[i]));
      CufftSafeCall(cufftDestroy(conf.fft_plans1[i]));
      CufftSafeCall(cufftDestroy(conf.fft_plans2[i]));
    }
  
  cudaFree(conf.dbuf_in);
#ifdef FOLD_MODE
  cudaFree(conf.dbuf_out_fold);
  cudaFreeHost(conf.hdat_offs_fold);
  cudaFreeHost(conf.hsquare_mean_fold);
  cudaFreeHost(conf.hdat_scl_fold);
  cudaFree(conf.ddat_offs_fold);
  cudaFree(conf.dsquare_mean_fold);
  cudaFree(conf.ddat_scl_fold);
  
  dada_hdu_unlock_write(conf.hdu_out_fold);
  dada_hdu_disconnect(conf.hdu_out_fold);
  dada_hdu_destroy(conf.hdu_out_fold);

#else
  cudaFree(conf.dbuf_out_search);
  cudaFreeHost(conf.hdat_offs_search);
  cudaFreeHost(conf.hsquare_mean_search);
  cudaFreeHost(conf.hdat_scl_search);
  cudaFree(conf.ddat_offs_search);
  cudaFree(conf.dsquare_mean_search);
  cudaFree(conf.ddat_scl_search);
  
  dada_hdu_unlock_write(conf.hdu_out_search);
  dada_hdu_disconnect(conf.hdu_out_search);
  dada_hdu_destroy(conf.hdu_out_search);
#endif

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
  size_t hdrsz;
  double freq;
  
  conf->hdrbuf_in  = ipcbuf_get_next_read(conf->hdu_in->header_block, &hdrsz);  
  if (!conf->hdrbuf_in)
    {
      multilog(runtime_log, LOG_ERR, "get next header block error.\n");
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if(hdrsz != DADA_HDR_SIZE)
    {
      multilog(runtime_log, LOG_ERR, "get next header block error.\n");
      fprintf(stderr, "Header size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

#ifdef FOLD_MODE
  conf->hdrbuf_out_fold = ipcbuf_get_next_write(conf->hdu_out_fold->header_block);
  if (!conf->hdrbuf_out_fold)
    {
      multilog(runtime_log, LOG_ERR, "get next header block error.\n");
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
    }
  
#else    
  conf->hdrbuf_out_search = ipcbuf_get_next_write(conf->hdu_out_search->header_block);
  if (!conf->hdrbuf_out_search)
    {
      multilog(runtime_log, LOG_ERR, "get next header block error.\n");
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
#endif
  
  if(conf->stream)
    {      
      /* Get utc_start from hdrin */
      if (ascii_header_get(conf->hdrbuf_in, "UTC_START", "%s", conf->utc_start) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_get UTC_START\n");
	  fprintf(stderr, "Error getting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
      fprintf(stdout, "\nGet UTC_START at process stage:\t\t%s\n", conf->utc_start);
      
      /* Get picoseconds from hdrin */
      if (ascii_header_get(conf->hdrbuf_in, "PICOSECONDS", "%"PRIu64, &(conf->picoseconds)) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_get PICOSECONDS\n");
	  fprintf(stderr, "Error getting PICOSECONDS, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
      fprintf(stdout, "Get PICOSECONDS at process stage:\t%"PRIu64"\n", conf->picoseconds);
      
      /* Get frequency from hdrin */
      if (ascii_header_get(conf->hdrbuf_in, "FREQ", "%lf", &freq) < 0)   // RA and DEC also need to pass from hdrin to hdrout
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_get FREQ\n");
	  fprintf(stderr, "Error getting FREQ, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
#ifdef FOLD_MODE
      if (fileread(conf->hfname, conf->hdrbuf_out_fold, DADA_HDR_SIZE) < 0)
	{
	  multilog(runtime_log, LOG_ERR, "cannot read header from %s\n", conf->hfname);
	  fprintf(stderr, "Error reading header file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
      
      /* Pass utc_start */
      if (ascii_header_set(conf->hdrbuf_out_fold, "UTC_START", "%s", conf->utc_start) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_set UTC_START\n");
	  fprintf(stderr, "Error setting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}	  
      fprintf(stdout, "Set UTC_START at process stage:\t\t%s\n", conf->utc_start);
      multilog(runtime_log, LOG_INFO, "UTC_START:\t%s\n", conf->utc_start);
      
      /* Pass picoseconds */
      if (ascii_header_set(conf->hdrbuf_out_fold, "PICOSECONDS", "%"PRIu64, conf->picoseconds) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_set PICOSECONDS\n");
	  fprintf(stderr, "Error setting PICOSECONDS, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}	  
      fprintf(stdout, "Set PICOSECONDS at process stage:\t%"PRIu64"\n\n", conf->picoseconds);
      multilog(runtime_log, LOG_INFO, "PICOSECONDS:\t%"PRIu64"\n", conf->picoseconds);
      
      /* Pass frequency */
      if (ascii_header_set(conf->hdrbuf_out_fold, "FREQ", "%.1lf", freq) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_set FREQ\n");
	  fprintf(stderr, "Error setting FREQ, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
#else	
      if (fileread(conf->hfname, conf->hdrbuf_out_search, DADA_HDR_SIZE) < 0)
	{
	  multilog(runtime_log, LOG_ERR, "cannot read header from %s\n", conf->hfname);
	  fprintf(stderr, "Error reading header file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
      
      /* Pass utc_start */
      if (ascii_header_set(conf->hdrbuf_out_search, "UTC_START", "%s", conf->utc_start) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_set UTC_START\n");
	  fprintf(stderr, "Error setting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
      fprintf(stdout, "Set UTC_START at process stage:\t\t%s\n", conf->utc_start);
      multilog(runtime_log, LOG_INFO, "UTC_START:\t%s\n", conf->utc_start);
      
      /* Pass picoseconds */
      if (ascii_header_set(conf->hdrbuf_out_search, "PICOSECONDS", "%"PRIu64, conf->picoseconds) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_set PICOSECONDS\n");
	  fprintf(stderr, "Error setting PICOSECONDS, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
      fprintf(stdout, "Set PICOSECONDS at process stage:\t%"PRIu64"\n\n", conf->picoseconds);
      multilog(runtime_log, LOG_INFO, "PICOSECONDS:\t%"PRIu64"\n", conf->picoseconds);
      
      /* Pass freq */
      if (ascii_header_set(conf->hdrbuf_out_search, "FREQ", "%.1lf", freq) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_get FREQ\n");
	  fprintf(stderr, "Error setting FREQ, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
#endif
    }
  else
    {      
      if (ascii_header_get(conf->hdrbuf_in, "UTC_START", "%s", conf->utc_start) < 0)  
	{
	  multilog(runtime_log, LOG_ERR, "failed ascii_header_get UTC_START\n");
	  fprintf(stderr, "Error getting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
#ifdef FOLD_MODE
      memcpy(conf->hdrbuf_out_fold, conf->hdrbuf_in, DADA_HDR_SIZE);
#else	
      memcpy(conf->hdrbuf_out_search, conf->hdrbuf_in, DADA_HDR_SIZE);
#endif
    }
  if(ipcbuf_mark_cleared (conf->hdu_in->header_block))  // We are the only one reader, so that we can clear it after read;
    {
      multilog(runtime_log, LOG_ERR, "Could not clear header block\n");
      fprintf(stderr, "Error header_clear, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  /* donot set header parameters anymore - acqn. doesn't start */
#ifdef FOLD_MODE
  if (ipcbuf_mark_filled (conf->hdu_out_fold->header_block, conf->hdrsz) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Could not mark filled header block\n");
      fprintf(stderr, "Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
#else
  if (ipcbuf_mark_filled (conf->hdu_out_search->header_block, conf->hdrsz) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Could not mark filled header block\n");
      fprintf(stderr, "Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
#endif
  return EXIT_SUCCESS;
}