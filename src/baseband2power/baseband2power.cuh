#ifndef _BASEBAND2POWER_CUH
#define _BASEBAND2POWER_CUH

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

#define MSTR_LEN      1024
#define DADA_HDRSZ    4096
#define NSAMP_DF      128
#define NPOL_SAMP     2
#define NDIM_POL      2
#define NCHK_BEAM     48
#define NCHAN_CHK     7
#define NCHAN_IN      (NCHK_BEAM * NCHAN_CHK)

#define SUM1_BLKSZ    1024
#define NBYTE_RT      4 // float
#define NBYTE_IN      2 // int16_t
#define NBYTE_OUT     4 // float
#define NBIT          (8 * NBYTE_OUT)

typedef struct conf_t
{
  char dir[MSTR_LEN];
  cudaStream_t *streams;
  
  uint64_t hbufin_offset, dbufin_offset;
  uint64_t bufrt1_offset, bufrt2_offset;
  uint64_t dbufout_offset, hbufout_offset;
  
  key_t key_in, key_out;
  dada_hdu_t *hdu_in, *hdu_out;
  char *hdrbuf_in, *hdrbuf_out;
  
  uint64_t bufin_size, bufrt_size, bufout_size;
  uint64_t sbufin_size, sbufrt_size, sbufout_size;
  uint64_t nsamp_in, nsamp_rt, nsamp_out;
  uint64_t ndata_in, ndata_rt, ndata_out;
  uint64_t hdrsz;
  uint64_t picoseconds;
  double mjd_start;

  char *curbuf_in, *curbuf_out;
  double bufin_ndf;

  uint64_t rbufin_ndf_chk;
  int nstream, nrun_blk, stream_ndf_chk;
  int64_t *dbuf_in;
  float *dbuf_out;
  float *buf_rt1, *buf_rt2;

  double tsamp_out, tsamp_in;
  double fsz_out, fsz_in;
  double bps_out, bps_in;
  
  dim3 gridsize_unpack_detect, blocksize_unpack_detect;
  dim3 gridsize_sum1, blocksize_sum1;
  dim3 gridsize_sum2, blocksize_sum2;

  ipcbuf_t *db_in, *db_out;
}conf_t;

int init_baseband2power(conf_t *conf);
int baseband2power(conf_t conf);
int destroy_baseband2power(conf_t conf);
int register_header(conf_t *conf);
#endif