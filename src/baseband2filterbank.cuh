#ifndef _PROCESS_CUH
#define _PROCESS_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <stdio.h>

#include "dada_cuda.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"
#include "constants.h"

#define NBYTE_RT              8    // cudaComplex
#define NBYTE_OUT             1    // uint8_t

#define NBIT_OUT                 8
#define NDIM_OUT                 1
#define NPOL_OUT                 1
#define SCL_UINT8            64.0f          // uint8_t, detected samples should vary in 0.5 * range(uint8_t) = 127, to be safe, use 0.25
#define OFFS_UINT8           64.0f          // uint8_t, detected samples should center at 0.5 * range(uint8_t) = 127, to be safe, use 0.25
#define SCL_NSIG             3.0f

typedef struct conf_t
{
  FILE *logfile;

  int nrun_blk;
  int nchk_in, nchan_in;
  int cufft_nx, cufft_mod;
  int nchan_keep_band, nchan_out, nchan_keep_chan, nchan_edge;
  double nchan_ratei, bw, scl_dtsz;
    
  int stream_ndf_chk;
  int nstream;
  float sclndim;
  int sod;
  
  char dir[MSTR_LEN];
  char utc_start[MSTR_LEN];
  
  key_t key_in, key_out;
  dada_hdu_t *hdu_in, *hdu_out;

  ipcbuf_t *db_in, *db_out;
  char *curbuf_in, *curbuf_out;
  int64_t *dbuf_in;
  uint8_t *dbuf_out;
  
  uint64_t rbufin_ndf_chk;
  uint64_t bufin_size, bufout_size; // Device buffer size for all streams
  uint64_t sbufin_size, sbufout_size; // Buffer size for each stream
  uint64_t bufrt1_size, bufrt2_size;
  uint64_t sbufrt1_size, sbufrt2_size;
  cufftComplex *buf_rt1, *buf_rt2;
  uint64_t hbufin_offset, dbufin_offset;
  uint64_t bufrt1_offset, bufrt2_offset;
  uint64_t dbufout_offset, hbufout_offset;
  uint64_t nsamp1, npol1, ndata1;
  uint64_t nsamp2, npol2, ndata2;
  uint64_t nsamp3, npol3, ndata3; // For search part
  
  uint64_t hdrsz, rbufin_size, rbufout_size; // HDR size for both HDU and ring buffer size of input HDU;
  // Input ring buffer size is different from the size of bufin, which is the size for GPU input memory;
  // Out ring buffer size is the same with the size of bufout, which is the size for GPU output memory;
  
  float *ddat_offs, *dsquare_mean, *ddat_scl;
  float *hdat_offs, *hsquare_mean, *hdat_scl;
  cudaStream_t *streams;
  cufftHandle *fft_plans;
  
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose, blocksize_swap_select_transpose;
  dim3 gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale;
  dim3 gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose;
  dim3 gridsize_accumulate, blocksize_accumulate;
  dim3 gridsize_mean, blocksize_mean;
  dim3 gridsize_scale, blocksize_scale;
}conf_t; 

int init_baseband2filterbank(conf_t *conf);
int baseband2filterbank(conf_t conf);
int dat_offs_scl(conf_t conf);
int register_header(conf_t *conf);

int destroy_baseband2filterbank(conf_t conf);

#endif