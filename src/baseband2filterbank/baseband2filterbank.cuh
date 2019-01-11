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
#define NPOL_IN             2
#define NDIM_IN              2
#define NCHAN_IN              (NCHK_BEAM * NCHAN_CHK)

#define NBYTE_RT              8    // cudaComplex
#define NBYTE_IN              2    // int16_t
#define NBYTE_OUT             1    // uint8_t

#define OSAMP_RATEI           0.84375
#define CUFFT_RANK            1

#define CUFFT_NX             64
#define CUFFT_MOD            27              // Set to remove oversampled data
#define NCHAN_KEEP_CHAN      (int)(CUFFT_NX * OSAMP_RATEI)
//#define NCHAN_OUT            512            // Final number of channels for search mode
#define NCHAN_OUT            1024            // Final number of channels for search mode
#define NCHAN_KEEP_BAND      16384           // a good number which is divisible by NCHAN_OUT
#define NCHAN_EDGE           (int)((NCHAN_IN * NCHAN_KEEP_CHAN - NCHAN_KEEP_BAND)/2)
//#define NSAMP_AVE            (int)(NCHAN_KEEP_BAND / NCHAN_OUT)

#define NCHAN_RATEI          (NCHAN_IN * NCHAN_KEEP_CHAN / (double)NCHAN_KEEP_BAND) 

#define SCL_DTSZ             (OSAMP_RATEI * (double)(NBYTE_OUT * NCHAN_OUT)/ (NCHAN_RATEI * NCHAN_KEEP_BAND * NPOL_IN * NDIM_IN * (double)NBYTE_IN))
//#define SCL_DTSZ             (OSAMP_RATEI * (double)NBYTE_OUT/ (NCHAN_RATEI * NSAMP_AVE * NPOL_IN * NDIM_IN * (double)NBYTE_IN))
#define BW                   (NCHAN_KEEP_BAND/(double)NCHAN_KEEP_CHAN)
#define NBIT_OUT                 8
#define NDIM_OUT                 1
#define NPOL_OUT                 1
#define SCL_UINT8            64.0f          // uint8_t, detected samples should vary in 0.5 * range(uint8_t) = 127, to be safe, use 0.25
#define OFFS_UINT8           64.0f          // uint8_t, detected samples should center at 0.5 * range(uint8_t) = 127, to be safe, use 0.25
#define SCL_NSIG             3.0f
  
typedef struct conf_t
{
  int stream_ndf_chk;
  int nstream;
  float sclndim;
  int sod;
  
  int nrun_blk;
  char dir[MSTR_LEN];
  char utc_start[MSTR_LEN];
  
  key_t key_in, key_out;
  dada_hdu_t *hdu_in, *hdu_out;

  ipcbuf_t *db_in, *db_out;
  char *curbuf_in, *curbuf_out;
  int64_t *dbuf_in;
  uint8_t *dbuf_out;
  
  double rbufin_ndf_chk;
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
  dim3 gridsize_detect_add_scale, blocksize_detect_add_scale;
  dim3 gridsize_detect_add_pad_transpose, blocksize_detect_add_pad_transpose;
  dim3 gridsize_sum, blocksize_sum;
  dim3 gridsize_mean, blocksize_mean;
  dim3 gridsize_scale, blocksize_scale;
}conf_t; 

int init_baseband2filterbank(conf_t *conf);
int baseband2filterbank(conf_t conf);
int dat_offs_scl(conf_t conf);
int register_header(conf_t *conf);

int destroy_baseband2filterbank(conf_t conf);

#endif