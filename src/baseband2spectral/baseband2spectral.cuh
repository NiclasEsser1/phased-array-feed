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

#define MSTR_LEN    1024
#define DADA_HDRSZ  4096

#define NCHK_BEAM             48   // How many frequency chunks we will receive, we should read the number from metadata
#define NCHAN_CHK             7
#define NSAMP_DF              128
#define NPOL_IN               2
#define NDIM_IN               2
#define NCHAN_IN              (NCHK_BEAM * NCHAN_CHK)

#define NBYTE_IN              2    // int16_t
#define NBYTE_RT_C            8    // cudaComplex
#define NBYTE_RT_F            4    // float, different from filterbank part, I detect the data just after the FFT
#define NBYTE_OUT             4    // float

#define OSAMP_RATEI           0.84375
#define CUFFT_RANK            1
#define SUM1_BLKSZ            64

#define CUFFT_NX             64
#define CUFFT_MOD            27              // Set to remove oversampled data
#define NCHAN_KEEP_CHAN      (int)(CUFFT_NX * OSAMP_RATEI)
#define NCHAN_KEEP_BAND      (NCHAN_IN * NCHAN_KEEP_CHAN)
#define BW_OUT               NCHAN_IN
#define NBIT_OUT             (8 * NBYTE_OUT)
#define NDIM_OUT             1
#define NPOL_OUT             1

typedef struct conf_t
{
  int stream_ndf_chk;
  int nstream;

  int nrun_blk;
  char dir[MSTR_LEN];

  double scale;
  key_t key_in, key_out;
  dada_hdu_t *hdu_in, *hdu_out;

  ipcbuf_t *db_in, *db_out;
  char *curbuf_in, *curbuf_out;
  int64_t *dbuf_in;
  float *dbuf_out;
  
  double rbufin_ndf_chk;
  uint64_t bufin_size, bufout_size; // Device buffer size for all streams
  uint64_t sbufin_size, sbufout_size; // Buffer size for each stream
  
  uint64_t bufrtc_size, bufrtf1_size, bufrtf2_size;
  uint64_t sbufrtc_size, sbufrtf1_size, sbufrtf2_size;

  cufftComplex *buf_rtc;       // For cufftComplex before and after FFT
  float *buf_rtf1, *buf_rtf2; // For float after detection
  
  uint64_t hbufin_offset, dbufin_offset;
  uint64_t bufrtc_offset, bufrtf1_offset, bufrtf2_offset;
  
  uint64_t dbufout_offset, hbufout_offset;
  uint64_t nsamp_in, npol_in, ndata_in;
  uint64_t nsamp_out, npol_out, ndata_out;
  
  uint64_t nsamp_rtc, npol_rtc, ndata_rtc;
  uint64_t nsamp_rtf1, npol_rtf1, ndata_rtf1;
  uint64_t nsamp_rtf2, npol_rtf2, ndata_rtf2;
  
  uint64_t hdrsz, rbufin_size, rbufout_size; // HDR size for both HDU and ring buffer size of input HDU;
  // Input ring buffer size is different from the size of bufin, which is the size for GPU input memory;
  // Out ring buffer size is the same with the size of bufout, which is the size for GPU output memory;
  
  cudaStream_t *streams;
  cufftHandle *fft_plans;
  
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_detect, blocksize_swap_select_transpose_detect;
  dim3 gridsize_sum1, blocksize_sum1;
  dim3 gridsize_sum2, blocksize_sum2;
}conf_t; 

int init_baseband2spectral(conf_t *conf);
int baseband2spectral(conf_t conf);
int register_header(conf_t *conf);
int destroy_baseband2spectral(conf_t conf);

#endif