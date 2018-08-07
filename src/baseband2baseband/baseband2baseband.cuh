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
#include "daemon.h"
#include "futils.h"

#define MSTR_LEN    512
#define HN_LEN      8        // The length of host name

#define CUFFT_NX64
#define FOLD_MODE    
#define FOLD_NPOW2

#define DADA_HDR_SIZE         4096
#define NCHK_NIC              48   // How many frequency chunks we will receive, we should read the number from metadata
#define NCHAN_CHK             7
#define NSAMP_DF              128
#define NPOL_SAMP             2
#define NDIM_POL              2

#define NBYTE_RT              8    // cudaComplex
#define NBYTE_IN              2    // int16_t
#define NBYTE_OUT_FOLD        1    // int8_t
#define NBYTE_OUT_SEARCH      1    // uint8_t
//#define NBYTE_OUT_FOLD      4    // float

#define OSAMP_RATEI           0.84375
#define CUFFT_RANK1           1
#define CUFFT_RANK2           1               // Only for fold mode

#ifdef CUFFT_NX32
#define CUFFT_NX1             32
#define CUFFT_MOD1            13              // Set to remove oversampled data
#define NCHAN_KEEP1           27              // (OSAMP_RATEI * CUFFT_NX1)

#if defined (FOLD_MODE) && defined (FOLD_NPOW2)
#define CUFFT_NX2             27
#define CUFFT_MOD2            14              
#define NCHAN_KEEP2           8748            // (CUFFT_NX2 * NCHAN_FOLD) for fold mode, a good number which is divisible by NCHAN_SEARCH for search mode
#define NCHAN_EDGE            162             // (NCHAN_KEEP1 * NCHK_NIC * NCHAN_CHK - NCHAN_KEEP2)/2
#define TILE_DIM              27              // CUFFT_NX2, only for fold mode
#define NROWBLOCK_TRANS       9               // a good number which can be devided by CUFFT_NX2 (TILE_DIM), only for fold mode
#define NCHAN_FOLD            324             // Final number of channels for fold mode
#define NCHAN_RATEI           (28.0/27.0)     // (NCHAN_KEEP1 * NCHK_NIC * NCHAN_CHK)/NCHAN_KEEP2
#define NCHAN_SEARCH          1024            // Final number of channels for search mode
#else
#define CUFFT_NX2             32
#define CUFFT_MOD2            16              // CUFFT_NX2 / 2
#define NCHAN_KEEP2           8192            // (CUFFT_NX2 * NCHAN_FOLD) for fold mode, a good number which is divisible by NCHAN_SEARCH for search mode
#define NCHAN_EDGE            440             // (NCHAN_KEEP1 * NCHK_NIC * NCHAN_CHK - NCHAN_KEEP2)/2
#define TILE_DIM              32              // CUFFT_NX2, only for fold mode
#define NROWBLOCK_TRANS       8               // a good number which can be devided by CUFFT_NX2 (TILE_DIM), only for fold mode
#define NCHAN_FOLD            256             // Final number of channels for fold mode
#define NCHAN_SEARCH          1024            // Final number of channels for search mode
#define NCHAN_RATEI           1.107421875     // (NCHAN_KEEP1 * NCHK_NIC * NCHAN_CHK)/NCHAN_KEEP2
#endif
#endif

#ifdef CUFFT_NX64
#define CUFFT_NX1             64
#define CUFFT_MOD1            27              // Set to remove oversampled data
#define NCHAN_KEEP1           54              // (OSAMP_RATEI * CUFFT_NX1)
#if defined (FOLD_MODE) && defined (FOLD_NPOW2)
#define CUFFT_NX2             54
#define CUFFT_MOD2            27              // CUFFT_NX2 / 2
#define NCHAN_KEEP2           17496            // (CUFFT_NX2 * NCHAN_FOLD) for fold mode, a good number which is divisible by NCHAN_SEARCH for search mode
#define NCHAN_EDGE            324             // (NCHAN_KEEP1 * NCHK_NIC * NCHAN_CHK - NCHAN_KEEP2)/2
#define TILE_DIM              54              // CUFFT_NX2, only for fold mode
#define NROWBLOCK_TRANS       18               // a good number which can be devided by CUFFT_NX2 (TILE_DIM), only for fold mode
#define NCHAN_FOLD            324             // Final number of channels for fold mode
#define NCHAN_RATEI           (28.0/27.0)     // (NCHAN_KEEP1 * NCHK_NIC * NCHAN_CHK)/NCHAN_KEEP2
#define NCHAN_SEARCH          1024            // Final number of channels for search mode
#else
#define CUFFT_NX2             64
#define CUFFT_MOD2            32              // CUFFT_NX2 / 2
#define NCHAN_KEEP2           16384           // (CUFFT_NX2 * NCHAN_FOLD) for fold mode, a good number which is divisible by NCHAN_SEARCH for search mode
#define NCHAN_EDGE            880             // (NCHAN_KEEP1 * NCHK_NIC * NCHAN_CHK - NCHAN_KEEP2)/2
#define TILE_DIM              64              // CUFFT_NX2, only for fold mode
#define NROWBLOCK_TRANS       8              // a good number which can be devided by CUFFT_NX2 (TILE_DIM), only for fold mode
#define NCHAN_FOLD            256             // Final number of channels for fold mode
#define NCHAN_SEARCH          1024            // Final number of channels for search mode
#define NCHAN_RATEI           1.107421875     // (NCHAN_KEEP1 * NCHK_NIC * NCHAN_CHK)/NCHAN_KEEP2
#endif
#endif

#define FOLD_SCALE            ((NBYTE_IN - NBYTE_OUT_FOLD) * 8 + (int)__log2f(CUFFT_NX1))

#define SCL_INT8              127.0f          // For int8_t, for fold mode
#define SCL_UINT8             255.0f          // For uint8_t, for search mode
#define SCL_NSIG              4.0f            // 4 sigma, 99.993666%

/* 
   The following parameters are for the speedup of transpose_scale_kernel3 and transpose_scale_kernel4;
   Be careful here as these parameters are sort of fixed. 
   For example, if we change the NCHAN_FOLD or CUFFT_NX2, we need to change the parameters here;
   We may not be able to find a suitable parameter if we do the change;
   In short, the current setup here only works with selected configuration.
   The second look into it turns out that the parameters here are more general.
*/
//#define TILE_DIM              32              // CUFFT_NX2
//#define NROWBLOCK_TRANS       8               // a good number which can be devided by CUFFT_NX2 (TILE_DIM)

typedef struct conf_t
{
  int device_id;
  int stream;
  
  char hfname[MSTR_LEN];
  int sod;
  int stream_ndf;
  int nstream;
  float sclndim_fold, sclndim_search;

  int nrun_blk;
  char dir[MSTR_LEN];
  char utc_start[MSTR_LEN];
  uint64_t picoseconds;
  
  key_t key_out_fold, key_in, key_out_search;
  dada_hdu_t *hdu_out_fold, *hdu_in, *hdu_out_search;
  
  char *hdrbuf_in, *hdrbuf_out_fold, *hdrbuf_out_search;
  int64_t *dbuf_in;
  int8_t *dbuf_out_fold;
  uint8_t *dbuf_out_search;
  //float *dbuf_out;

  double freq;
  
  double rbufin_ndf;
  size_t bufin_size, bufout_size_fold, bufout_size_search; // Device buffer size for all streams
  size_t sbufin_size, sbufout_size_fold, sbufout_size_search; // Buffer size for each stream
  size_t bufrt1_size, bufrt2_size;
  size_t sbufrt1_size, sbufrt2_size;
  cufftComplex *buf_rt1, *buf_rt2;
  size_t hbufin_offset, dbufin_offset;
  size_t bufrt1_offset, bufrt2_offset;
  size_t dbufout_offset_fold, hbufout_offset_fold; 
  size_t dbufout_offset_search, hbufout_offset_search;
  size_t nsamp1, npol1, ndata1;
  size_t nsamp2, npol2, ndata2;
  size_t nsamp3, npol3, ndata3; // For search part
  //size_t nbufin_rbuf;   // How many input GPU memory buffer can be fitted into the input ring buffer;
  
  size_t hdrsz, rbufin_size, rbufout_size_fold, rbufout_size_search; // HDR size for both HDU and ring buffer size of input HDU;
  // Input ring buffer size is different from the size of bufin, which is the size for GPU input memory;
  // Out ring buffer size is the same with the size of bufout, which is the size for GPU output memory;
  
  float *ddat_offs_fold, *dsquare_mean_fold, *ddat_scl_fold;
  float *hdat_offs_fold, *hsquare_mean_fold, *hdat_scl_fold;
  float *ddat_offs_search, *dsquare_mean_search, *ddat_scl_search;
  float *hdat_offs_search, *hsquare_mean_search, *hdat_scl_search;
  cudaStream_t *streams;
  cufftHandle *fft_plans1, *fft_plans2;
  
  dim3 gridsize_unpack, blocksize_unpack;
  dim3 gridsize_transpose_pad, blocksize_transpose_pad;
  dim3 gridsize_sum1_fold, blocksize_sum1_fold;
  dim3 gridsize_sum2_fold, blocksize_sum2_fold;
  dim3 gridsize_swap_select_transpose, blocksize_swap_select_transpose;
  dim3 gridsize_swap_select_transpose_swap, blocksize_swap_select_transpose_swap;
  dim3 gridsize_swap, blocksize_swap;
  dim3 gridsize_scale_fold, blocksize_scale_fold;
  
  dim3 gridsize_mean_fold, blocksize_mean_fold;
  dim3 gridsize_transpose_scale, blocksize_transpose_scale;
  dim3 gridsize_transpose_scale2, blocksize_transpose_scale2;
  dim3 gridsize_transpose_scale3, blocksize_transpose_scale3;
  dim3 gridsize_transpose_scale4, blocksize_transpose_scale4;
  dim3 gridsize_transpose_float, blocksize_transpose_float;

  /* For search mode */
  dim3 gridsize_add_detect_scale, blocksize_add_detect_scale;
  dim3 gridsize_add_detect_pad, blocksize_add_detect_pad;
  dim3 gridsize_sum_search, blocksize_sum_search;
  dim3 gridsize_mean_search, blocksize_mean_search;
  dim3 gridsize_scale_search, blocksize_scale_search;
}conf_t; 

int init_process(conf_t *conf);
int do_process(conf_t conf);
int dat_offs_scl(conf_t conf);
int register_header(conf_t *conf);

int destroy_process(conf_t conf);

#endif