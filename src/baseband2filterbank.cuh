#ifndef _BASEBAND2FILTERBANK_CUH
#define _BASEBAND2FILTERBANK_CUH

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

typedef struct fits_t
{
  int beam_index;
  char time_stamp[FITS_TIME_STAMP_LEN];
  float tsamp;
  int nchan;
  float center_freq;
  float chan_width;
  int pol_type;
  int pol_index;
  int nchunk;
  int chunk_index;
  float data[UDP_PAYLOAD_SIZE_MAX]; // Can not alloc dynamic
}fits_t;

typedef struct conf_t
{
  char *hdrbuf_in;
  int pol_type;
  char ip[MSTR_LEN];
  int port;
  FILE *log_file;
  uint64_t picoseconds;
  double center_freq;
  int beam_index;
  
  fits_t *fits;
  int neth_per_blk;
  int nseg_per_blk;
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
  int pktsz_network, dtsz_network;
  ipcbuf_t *db_in, *db_out;
  char *cbuf_in, *cbuf_out;
  int64_t *dbuf_in;
  uint8_t *dbuf_out1;
  float *dbuf_out2, *dbuf_out3, *dbuf_out4;
  double tsamp_in, tsamp_out1, tsamp_out2;
  
  uint64_t ndf_per_chunk_rbufin;
  uint64_t bufin_size, bufout1_size, bufout2_size; // Device buffer size for all streams
  uint64_t sbufin_size, sbufout1_size, sbufout2_size; // Buffer size for each stream
  uint64_t bufrt1_size, bufrt2_size;
  uint64_t sbufrt1_size, sbufrt2_size;
  cufftComplex *buf_rt1, *buf_rt2;
  uint64_t hbufin_offset, dbufin_offset;
  uint64_t bufrt1_offset, bufrt2_offset;
  uint64_t dbufout1_offset, dbufout2_offset, dbufout4_offset, hbufout1_offset, hbufout2_offset;
  uint64_t nsamp1, npol1, ndata1;
  uint64_t nsamp2, npol2, ndata2;
  uint64_t nsamp3, npol3, ndata3; // For search part
  
  uint64_t hdrsz, rbufin_size, rbufout1_size; // HDR size for both HDU and ring buffer size of input HDU;
  // Input ring buffer size is different from the size of bufin, which is the size for GPU input memory;
  // Out ring buffer size is the same with the size of bufout, which is the size for GPU output memory;
  
  cufftComplex *offset_scale_d, *offset_scale_h;
  cudaStream_t *streams;
  cufftHandle *fft_plans;
  int n_transpose, m_transpose;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_transpose, blocksize_transpose;
  dim3 gridsize_swap_select_transpose, blocksize_swap_select_transpose;
  dim3 gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale;
  dim3 gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose;
  dim3 gridsize_taccumulate, blocksize_taccumulate;
  dim3 gridsize_scale, blocksize_scale;
  int naccumulate_pad, naccumulate_scale, naccumulate;

  uint64_t file_size_in, bytes_per_second_in;
}conf_t; 

int initialize_baseband2filterbank(conf_t *conf);
int baseband2filterbank(conf_t conf);
int offset_scale(conf_t conf);
int read_dada_header(conf_t *conf);
int register_dada_header(conf_t *conf);

int destroy_baseband2filterbank(conf_t conf);
int default_arguments(conf_t *conf);
int examine_record_arguments(conf_t conf, char **argv, int argc);

#endif