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
#include "fits.h"

typedef struct conf_t
{
  char *hdrbuf_in;
  int ptype_monitor, ptype_spectral;
  char ip_monitor[MSTR_LEN], ip_spectral[MSTR_LEN];
  int port_monitor, port_spectral;
  FILE *log_file;
  uint64_t picoseconds;
  double cfreq_band;
  double cfreq_spectral;
  int beam_index;

  int monitor, spectral2disk, spectral2network;
  fits_t *fits_monitor;
  fits_t fits_spectral;
  int neth_per_blk, neth_per_blk_spectral;
  int nseg_per_blk, nseg_per_blk_spectral;
  
  int nrepeat_per_blk;
  int start_chunk;
  int nchunk_in, nchan_in, nchunk_in_spectral, nchan_in_spectral;
  int cufft_nx, cufft_mod, cufft_nx_spectral, cufft_mod_spectral;
  int nchan_keep_band, nchan_out, nchan_keep_chan, nchan_edge, nchan_out_spectral, nchan_keep_chan_spectral;
  double inverse_nchan_rate, bandwidth, scale_dtsz, bandwidth_spectral, scale_dtsz_spectral;
    
  int ndf_per_chunk_stream;
  int nstream;
  float ndim_scale;
  int sod, sod_spectral;

  int ndim_spectral, npol_spectral;
  char dir[MSTR_LEN];
  char utc_start[MSTR_LEN];
  
  key_t key_in, key_out, key_out_spectral;
  dada_hdu_t *hdu_in, *hdu_out, *hdu_out_spectral;
  int pktsz_network, dtsz_network, pktsz_network_spectral, dtsz_network_spectral;
  ipcbuf_t *db_in, *db_out, *db_out_spectral;
  ipcbuf_t *hdr_in, *hdr_out, *hdr_out_spectral;
  char *cbuf_in, *cbuf_out, *cbuf_out_spectral;
  int64_t *dbuf_in;
  uint8_t *dbuf_out_filterbank;
  float *dbuf_out_monitor1, *dbuf_out_monitor2, *dbuf_out_monitor3;
  double tsamp_in, tsamp_out_filterbank, tsamp_out_monitor, tsamp_out_spectral;
  
  uint64_t ndf_per_chunk_rbufin;
  uint64_t bufin_size, bufout_size_filterbank, bufout_size_monitor; // Device buffer size for all streams
  uint64_t sbufin_size, sbufout_size_filterbank, sbufout_size_monitor; // Buffer size for each stream
  uint64_t bufrt1_size, bufrt2_size;
  uint64_t sbufrt1_size, sbufrt2_size;
  cufftComplex *buf_rt1, *buf_rt2;
  uint64_t hbufin_offset, dbufin_offset;
  uint64_t bufrt1_offset, bufrt2_offset;
  uint64_t dbufout_offset_filterbank, dbufout_offset_monitor1, dbufout_offset_monitor2, dbufout_offset_monitor3, hbufout_offset_filterbank;
  uint64_t nsamp_in, npol_in, ndata_in;
  uint64_t nsamp_keep, npol_keep, ndata_keep;
  uint64_t nsamp_filterbank, npol_filterbank, ndata_filterbank; // For search part
  
  uint64_t rbufin_size, rbufout_size_filterbank, rbufout_size_spectral; // HDR size for both HDU and ring buffer size of input HDU;
  // Input ring buffer size is different from the size of bufin, which is the size for GPU input memory;
  // Out ring buffer size is the same with the size of bufout, which is the size for GPU output memory;
  
  cufftComplex *offset_scale_d, *offset_scale_h;
  cudaStream_t *streams;
  cufftHandle *fft_plans;
  cufftHandle *fft_plans_spectral;
  int n_transpose, m_transpose;
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_transpose, blocksize_transpose;
  dim3 gridsize_swap_select_transpose, blocksize_swap_select_transpose;
  dim3 gridsize_detect_faccumulate_scale, blocksize_detect_faccumulate_scale;
  dim3 gridsize_detect_faccumulate_pad_transpose, blocksize_detect_faccumulate_pad_transpose;
  dim3 gridsize_taccumulate, blocksize_taccumulate;
  dim3 gridsize_scale, blocksize_scale;
  uint64_t naccumulate_pad, naccumulate_scale, naccumulate, naccumulate_spectral;
  
  uint64_t bufin_size_spectral, bufout_size_spectral; // Device buffer size for all streams
  uint64_t sbufin_size_spectral, sbufout_size_spectral; // Buffer size for each stream
  uint64_t bufrt1_size_spectral, bufrt2_size_spectral;
  uint64_t sbufrt1_size_spectral, sbufrt2_size_spectral;
  cufftComplex *buf_rt1_spectral, *buf_rt2_spectral;
  uint64_t hbufin_offset_spectral, dbufin_offset_spectral;
  uint64_t bufrt1_offset_spectral, bufrt2_offset_spectral;
  uint64_t dbufout_offset_spectral;
  uint64_t nsamp_in_spectral, npol_in_spectral, ndata_in_spectral;
  uint64_t nsamp_keep_spectral, npol_keep_spectral, ndata_keep_spectral;
  uint64_t nsamp_out_spectral, npol_out_spectral, ndata_out_spectral;
  float *dbuf_out_spectral;
  int nblk_accumulate;

  int nchunk_network, nchan_per_chunk_network;
  
  dim3 gridsize_swap_select_transpose_pft1, blocksize_swap_select_transpose_pft1;
  dim3 gridsize_spectral_taccumulate, blocksize_spectral_taccumulate;
  dim3 gridsize_saccumulate, blocksize_saccumulate;
  uint64_t file_size_in, bytes_per_second_in;
}conf_t; 

int initialize_baseband2filterbank(conf_t *conf);
int baseband2filterbank(conf_t conf);
int offset_scale(conf_t conf);
int read_dada_header(conf_t *conf);
int register_dada_header(conf_t *conf);
int register_dada_header_spectral(conf_t *conf);

int destroy_baseband2filterbank(conf_t conf);
int default_arguments(conf_t *conf);
int examine_record_arguments(conf_t conf, char **argv, int argc);

#endif