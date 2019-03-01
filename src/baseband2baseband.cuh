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

#define NCHAN_KEEP_CHAN       (int)(CUFFT_NX1 * OSAMP_RATEI)
#define CUFFT_NX2             (int)(CUFFT_NX1 * OSAMP_RATEI)              // We work in seperate raw channels
#define CUFFT_MOD2            (int)(CUFFT_NX2/2)         

#define NCHAN_OUT             324             // Final number of channels, multiple times of CUFFT2_NX2
#define NCHAN_KEEP_BAND       (int)(CUFFT_NX2 * NCHAN_OUT)
#define NCHAN_RATEI           (NCHAN_IN * NCHAN_KEEP_CHAN / (double)NCHAN_KEEP_BAND)

#define NCHAN_EDGE            (int)((NCHAN_IN * NCHAN_KEEP_CHAN - NCHAN_KEEP_BAND)/2)

#define SCL_DTSZ              (OSAMP_RATEI * (double)NBYTE_OUT/ (NCHAN_RATEI * (double)NBYTE_IN))
#define TSAMP                 (NCHAN_KEEP_CHAN/(double)CUFFT_NX2)

typedef struct conf_t
{
  int nchunk, nchan;
  int cufft_nx, cufft_mod, nchan_keep_chan;
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

  cufftComplex *buf_rt1, *buf_rt2;
  uint64_t rbufin_ndf_chk;
  uint64_t bufin_size, bufout_size;
  uint64_t sbufin_size, sbufout_size;
  uint64_t bufrt1_size, bufrt2_size;
  uint64_t sbufrt1_size, sbufrt2_size;
  uint64_t hbufin_offset, dbufin_offset;
  uint64_t bufrt1_offset, bufrt2_offset;
  uint64_t dbufout_offset, hbufout_offset; 
  uint64_t nsamp1, npol1, ndata1;
  uint64_t nsamp2, npol2, ndata2;
    
  uint64_t rbufin_size, rbufout_size;
  
  ipcbuf_t *db_in, *db_out;

  cufftComplex *offset_scale_d, *offset_scale_h;
  cudaStream_t *streams;
  cufftHandle *fft_plans1, *fft_plans2;
  
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap;
  dim3 gridsize_taccumulate, blocksize_taccumulate;
  dim3 gridsize_scale, blocksize_scale;

  dim3 gridsize_transpose_pad, blocksize_transpose_pad;
  dim3 gridsize_transpose_scale, blocksize_transpose_scale;
}conf_t; 

int init_baseband2baseband(conf_t *conf);
int baseband2baseband(conf_t conf);
int dat_offs_scl(conf_t conf);
int register_header(conf_t *conf);
int dat_offs_scl(conf_t conf);

int destroy_baseband2baseband(conf_t conf);

#endif