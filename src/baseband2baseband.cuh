#ifndef _BASEBAND2BASEBAND_CUH
#define _BASEBAND2BASEBAND_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <stdio.h>

#include "dada_cuda.h"
#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ascii_header.h"
#include "ipcbuf.h"
#include "daemon.h"
#include "futils.h"
#include "constants.h"
#include "fits.h"

typedef struct conf_t
{
  char ip[MSTR_LEN];
  int port;
  int fits_flag;
  fits_t *fits;
  int dtsz_network, pktsz_network;
  FILE *log_file;
  int nchunk, nchan;
  int cufft_nx, cufft_mod, nchan_keep_chan;
  int ndf_per_chunk_stream;
  uint64_t naccumulate;
  uint64_t naccumulate_spectral;
  int nstream;
  float ndim_scale;
  double scale_dtsz;
  int nseg_per_blk, neth_per_blk;
  
  int nrepeat_per_blk;
  char dir[MSTR_LEN];
  char utc_start[MSTR_LEN];
  uint64_t picoseconds;
  
  key_t key_out, key_in;
  dada_hdu_t *hdu_out, *hdu_in;
  char *cbuf_in, *cbuf_out;
  int64_t *dbuf_in;
  int8_t *dbuf_out;
  float *dbuf_out_monitor;
  int beam_index;
  int pol_type;
  double center_freq;

  cufftComplex *buf_rt1, *buf_rt2;
  uint64_t ndf_per_chunk_rbufin;
  uint64_t bufin_size, bufout_size;
  uint64_t sbufin_size, sbufout_size;
  uint64_t bufrt1_size, bufrt2_size;
  uint64_t sbufrt1_size, sbufrt2_size;
  uint64_t hbufin_offset, dbufin_offset;
  uint64_t bufrt1_offset, bufrt2_offset;
  uint64_t dbufout_offset, hbufout_offset; 
  uint64_t nsamp_in, npol_in, ndata_in;
  uint64_t nsamp_keep, npol_keep, ndata_keep;
  uint64_t nsamp_out, npol_out, ndata_out;
  uint64_t sbufout_size_monitor, bufout_size_monitor, dbufout_offset_monitor;  
  uint64_t rbufin_size, rbufout_size;
  
  ipcbuf_t *db_in, *db_out;

  cufftComplex *offset_scale_d, *offset_scale_h;
  cudaStream_t *streams;
  cufftHandle *fft_plans_forward, *fft_plans_backward;
  
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap;
  dim3 gridsize_taccumulate, blocksize_taccumulate;
  dim3 gridsize_scale, blocksize_scale;

  dim3 gridsize_transpose_pad, blocksize_transpose_pad;
  dim3 gridsize_transpose_scale, blocksize_transpose_scale;
  dim3 gridsize_transpose_complex, blocksize_transpose_complex;
  dim3 gridsize_spectral_taccumulate, blocksize_spectral_taccumulate;
  double tsamp;
  char *hdrbuf_in;
  uint64_t file_size_in, bytes_per_second_in;
  int sod;
}conf_t; 

int default_arguments(conf_t *conf);
int initialize_baseband2baseband(conf_t *conf);
int baseband2baseband(conf_t conf);
int read_dada_header(conf_t *conf);
int register_dada_header(conf_t *conf);

int offset_scale(conf_t conf);
int destroy_baseband2baseband(conf_t conf);
int examine_record_arguments(conf_t conf, char **argv, int argc);

#endif