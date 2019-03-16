#ifndef _BASEBAND2SPECTRAL_CUH
#define _BASEBAND2SPECTRAL_CUH

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
#include "fits.h"

typedef struct conf_t
{
  uint64_t picoseconds;
  double center_freq;
  int beam_index;
  FILE *log_file;
  int naccumulate;
  int pol_type, ndim_out, npol_out;
  int output_network;
  char ip[MSTR_LEN];
  int port;
  int nrepeat_per_blk;
  int nchunk_in, nchan_in, nchan_out;
  int cufft_nx, cufft_mod;
  int nchan_keep_chan;
  double bandwidth, scale_dtsz;
  int nblk_accumulate;
  char *hdrbuf_in;
  
  int ndf_per_chunk_stream;
  int nstream;
  int sod;

  uint64_t file_size_in;
  uint64_t bytes_per_second_in;
  
  int nchunk_network;
  int nchan_per_chunk_network;
  int dtsz_network;
  int pktsz_network;
  
  char dir[MSTR_LEN];
  char utc_start[MSTR_LEN];
  
  key_t key_in, key_out;
  dada_hdu_t *hdu_in, *hdu_out;

  ipcbuf_t *db_in, *db_out;
  char *cbuf_in, *cbuf_out;
  int64_t *dbuf_in;
  float *dbuf_out;
  double tsamp_in, tsamp_out;
  
  uint64_t ndf_per_chunk_rbufin;
  uint64_t bufin_size, bufout_size; // Device buffer size for all streams
  uint64_t sbufin_size, sbufout_size; // Buffer size for each stream
  uint64_t bufrt1_size, bufrt2_size;
  uint64_t sbufrt1_size, sbufrt2_size;
  cufftComplex *buf_rt1, *buf_rt2;
  uint64_t hbufin_offset, dbufin_offset;
  uint64_t bufrt1_offset, bufrt2_offset;
  uint64_t dbufout_offset;
  uint64_t nsamp_in, npol_in, ndata_in;
  uint64_t nsamp_keep, npol_keep, ndata_keep;
  uint64_t nsamp_out, ndata_out; 
  
  uint64_t hdrsz, rbufin_size, rbufout_size; // HDR size for both HDU and ring buffer size of input HDU;
  // Input ring buffer size is different from the size of bufin, which is the size for GPU input memory;
  // Out ring buffer size is the same with the size of bufout, which is the size for GPU output memory;
  
  cudaStream_t *streams;
  cufftHandle *fft_plans;
  
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_pft1, blocksize_swap_select_transpose_pft1;
  dim3 gridsize_spectral_taccumulate, blocksize_spectral_taccumulate;
  dim3 gridsize_saccumulate, blocksize_saccumulate;
}conf_t; 

int initialize_baseband2spectral(conf_t *conf);
int baseband2spectral(conf_t conf);

int register_dada_header(conf_t *conf);
int read_dada_header(conf_t *conf);

int destroy_baseband2spectral(conf_t conf);
int default_arguments(conf_t *conf);
int examine_record_arguments(conf_t conf, char **argv, int argc);

#endif