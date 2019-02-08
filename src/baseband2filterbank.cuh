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

typedef struct conf_t
{
  FILE *log_file;

  int nrepeat_per_blk;
  int nchunk_in, nchan_in;
  int cufft_nx, cufft_mod;
  int nchan_keep_band, nchan_out, nchan_keep_chan, nchan_edge;
  double inverse_nchan_rate, bandwidth, scale_dtsz;
    
  int ndf_per_chunk_stream;
  int nstream;
  float ndim_scale;
  int sod;
  
  char dir[MSTR_LEN];
  char utc_start[MSTR_LEN];
  
  key_t key_in, key_out;
  dada_hdu_t *hdu_in, *hdu_out;

  ipcbuf_t *db_in, *db_out;
  char *cbuf_in, *cbuf_out;
  int64_t *dbuf_in;
  uint8_t *dbuf_out;
  double tsamp_in, tsamp_out;
  
  uint64_t ndf_per_chunk_rbufin;
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
  
  cufftComplex *offset_scale_d, *offset_scale_h;
  cudaStream_t *streams;
  cufftHandle *fft_plans;
  
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose, blocksize_swap_select_transpose;
  dim3 gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale;
  dim3 gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose;
  dim3 gridsize_taccumulate, blocksize_taccumulate;
  dim3 gridsize_scale, blocksize_scale;
  int naccumulate_pad, naccumulate_scale, naccumulate;
}conf_t; 

int initialize_baseband2filterbank(conf_t *conf);
int baseband2filterbank(conf_t conf);
int offset_scale(conf_t conf);
int register_header(conf_t *conf);

int destroy_baseband2filterbank(conf_t conf);
int default_arguments(conf_t *conf);
int examine_record_arguments(conf_t conf, char **argv, int argc);

#endif