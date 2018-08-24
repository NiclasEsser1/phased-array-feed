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
extern int quit;
extern pthread_mutex_t quit_mutex;

int dat_offs_scl(conf_t conf)
{
  /*
    The procedure for fold mode is:
    1. Get PTFT data as we did at process;
    2. Pad the data;
    3. Add the padded data in time;
    4. Get the mean of the added data;
    5. Get the scale with the mean;

    For baseband2baseband, we need to get a identical scale for all channels, which means we have to update the code here. 
  */
  uint64_t i, j;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap;
  dim3 gridsize_mean, blocksize_mean;
  dim3 gridsize_sum1, blocksize_sum1;
  dim3 gridsize_sum2, blocksize_sum2;
  dim3 gridsize_scale, blocksize_scale;
  dim3 gridsize_transpose_pad, blocksize_transpose_pad;
  uint64_t hbufin_offset, dbufin_offset, bufrt1_offset, bufrt2_offset;
  
  gridsize_unpack                      = conf.gridsize_unpack;
  blocksize_unpack                     = conf.blocksize_unpack;
  gridsize_swap_select_transpose_swap  = conf.gridsize_swap_select_transpose_swap;   
  blocksize_swap_select_transpose_swap = conf.blocksize_swap_select_transpose_swap; 
  gridsize_transpose_pad               = conf.gridsize_transpose_pad;
  blocksize_transpose_pad              = conf.blocksize_transpose_pad;
  	         	               						       
  gridsize_sum1              = conf.gridsize_sum1;	       
  blocksize_sum1             = conf.blocksize_sum1;
  gridsize_sum2              = conf.gridsize_sum2;	       
  blocksize_sum2             = conf.blocksize_sum2;
  gridsize_mean              = conf.gridsize_mean;	       
  blocksize_mean             = conf.blocksize_mean;
  gridsize_scale              = conf.gridsize_scale;	       
  blocksize_scale             = conf.blocksize_scale;
  
  if(conf.hdu_in->data_block->curbuf == NULL)
    {
      multilog (runtime_log, LOG_ERR, "Can not get buffer block from input ring buffer, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not get buffer block from input ring buffer, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
    
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

	  /* Prepare for inverse FFT */
	  swap_select_transpose_swap_kernel<<<gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap, 0, conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset], conf.nsamp1, conf.nsamp2); 
	  
	  /* Do inverse FFT */
	  CufftSafeCall(cufftExecC2C(conf.fft_plans2[j], &conf.buf_rt2[bufrt2_offset], &conf.buf_rt2[bufrt2_offset], CUFFT_INVERSE));
	  
	  /* Transpose the data from PTFT to FTP for later calculation */
	  transpose_pad_kernel<<<gridsize_transpose_pad, blocksize_transpose_pad, 0, conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], conf.nsamp2, &conf.buf_rt1[bufrt1_offset]);
	  
	  /* Get the sum of samples and square of samples */
	  sum_kernel<<<gridsize_sum1, blocksize_sum1, blocksize_sum1.x * sizeof(cufftComplex), conf.streams[j]>>>(&conf.buf_rt1[bufrt1_offset], &conf.buf_rt2[bufrt2_offset]);
	  sum_kernel<<<gridsize_sum2, blocksize_sum2, blocksize_sum2.x * sizeof(cufftComplex), conf.streams[j]>>>(&conf.buf_rt2[bufrt2_offset], &conf.buf_rt1[bufrt1_offset]);
	}
      CudaSynchronizeCall(); // Sync here is for multiple streams

      mean_kernel<<<gridsize_mean, blocksize_mean>>>(conf.buf_rt1, conf.bufrt1_offset, conf.ddat_offs, conf.dsquare_mean, conf.nstream, conf.sclndim);
    }
  /* Get the scale of each chanel */
  scale_kernel<<<gridsize_scale, blocksize_scale>>>(conf.ddat_offs, conf.dsquare_mean, conf.ddat_scl);
  CudaSynchronizeCall();
  
  CudaSafeCall(cudaMemcpy(conf.hdat_offs, conf.ddat_offs, sizeof(float) * NCHAN, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(conf.hdat_scl, conf.ddat_scl, sizeof(float) * NCHAN, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(conf.hsquare_mean, conf.dsquare_mean, sizeof(float) * NCHAN, cudaMemcpyDeviceToHost));
 
  return EXIT_SUCCESS;
}
