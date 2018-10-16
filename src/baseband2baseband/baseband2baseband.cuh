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

#define MSTR_LEN      1024
#define DADA_HDRSZ    4096

#define NCHK_BEAM             48   // How many frequency chunks we will receive, we should read the number from metadata
#define NCHAN_CHK             7
#define NSAMP_DF              128
#define NPOL_SAMP             2
#define NDIM_POL              2
#define NCHAN_IN              (NCHK_BEAM * NCHAN_CHK)

#define NBYTE_RT              8
#define NBYTE_IN              2   // 16 bits
#define NBYTE_OUT             1   // 8 bits

#define OSAMP_RATEI           0.84375  // 27.0/32.0
#define CUFFT_RANK1           1
#define CUFFT_RANK2           1         

#define CUFFT_NX1             64
#define CUFFT_MOD1            27              // Set to remove oversampled data
#define NCHAN_KEEP_CHAN       (int)(CUFFT_NX1 * OSAMP_RATEI)
#define CUFFT_NX2             (int)(CUFFT_NX1 * OSAMP_RATEI)              // We work in seperate raw channels
#define CUFFT_MOD2            (int)(CUFFT_NX2/2)         

#define NCHAN_OUT             324             // Final number of channels, multiple times of CUFFT2_NX2
#define NCHAN_KEEP_BAND       (int)(CUFFT_NX2 * NCHAN_OUT)
#define NCHAN_RATEI           (NCHAN_IN * NCHAN_KEEP_CHAN / (double)NCHAN_KEEP_BAND)

#define NCHAN_EDGE            (int)((NCHAN_IN * NCHAN_KEEP_CHAN - NCHAN_KEEP_BAND)/2)
#define TILE_DIM              CUFFT_NX2
#define NROWBLOCK_TRANS       18               // Multiple times of TILE_DIM (CUFFT_NX2)

#define SCL_DTSZ              (OSAMP_RATEI * (double)NBYTE_OUT/ (NCHAN_RATEI * (double)NBYTE_IN))
#define SCL_SIG               ((NBYTE_IN - NBYTE_OUT) * 8 - (int)__log2f(CUFFT_NX1)) // Not exact
#define TSAMP                 (NCHAN_KEEP_CHAN/(double)CUFFT_NX2)
#define NBIT                  8

#define SCL_INT8              127.0f          // int8_t
#define OFFS_INT8             0.0f
#define SCL_NSIG              6.0f            // 4 sigma, 99.993666%

typedef struct conf_t
{
  int stream_ndf_chk;
  int nstream;
  float sclndim;

  int nrun_blk;
  char dir[MSTR_LEN];
  char utc_start[MSTR_LEN];
  
  key_t key_out, key_in;
  dada_hdu_t *hdu_out, *hdu_in;
  char *curbuf_in, *curbuf_out;
  int64_t *dbuf_in;
  int8_t *dbuf_out;
  
  uint64_t rbufin_ndf_chk;
  uint64_t bufin_size, bufout_size;
  uint64_t sbufin_size, sbufout_size;
  uint64_t bufrt1_size, bufrt2_size;
  uint64_t sbufrt1_size, sbufrt2_size;
  cufftComplex *buf_rt1, *buf_rt2;
  uint64_t hbufin_offset, dbufin_offset;
  uint64_t bufrt1_offset, bufrt2_offset;
  uint64_t dbufout_offset, hbufout_offset; 
  uint64_t nsamp1, npol1, ndata1;
  uint64_t nsamp2, npol2, ndata2;
  //uint64_t nbufin_rbuf;   // How many input GPU memory buffer can be fitted into the input ring buffer;
  
  uint64_t rbufin_size, rbufout_size;
  // Input ring buffer size is different from the size of bufin, which is the size for GPU input memory;
  // Out ring buffer size is the same with the size of bufout, which is the size for GPU output memory;

  ipcbuf_t *db_in, *db_out;
  
  float *ddat_offs, *dsquare_mean, *ddat_scl;
  float *hdat_offs, *hsquare_mean, *hdat_scl;
  cudaStream_t *streams;
  cufftHandle *fft_plans1, *fft_plans2;
  
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_mean, blocksize_mean;
  dim3 gridsize_transpose_pad, blocksize_transpose_pad;
  dim3 gridsize_scale, blocksize_scale;
  dim3 gridsize_sum1, blocksize_sum1;
  dim3 gridsize_sum2, blocksize_sum2;
  dim3 gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap;
  
  dim3 gridsize_transpose_scale, blocksize_transpose_scale;
  dim3 gridsize_transpose_float, blocksize_transpose_float;
}conf_t; 

int init_baseband2baseband(conf_t *conf);
int baseband2baseband(conf_t conf);
int dat_offs_scl(conf_t conf);
int register_header(conf_t *conf);
int dat_offs_scl(conf_t conf);

int destroy_baseband2baseband(conf_t conf);

#endif