#ifdef __cplusplus
extern "C" {
#endif
#ifndef _CONSTANTS_CUH
#define _CONSTANTS_CUH

#define TILE_DIM  32
#define NROWBLOCK_TRANS 8
  
#define TRANS_BUFSZ    1048576 // 1MB

#define MSTR_LEN       1024
#define DADA_HDRSZ     4096

#define NBYTE_RT             8    // cudaComplex, for one pol
  
#define NPOL_BASEBAND        2
#define NDIM_BASEBAND        2
#define NBYTE_BASEBAND       2    // int16_t
  
#define NBIT_FILTERBANK      8
#define NDIM_FILTERBANK      1
#define NPOL_FILTERBANK      1
#define NBYTE_FILTERBANK     1    // uint8_t
  
#define NDATA_PER_SAMP_FULL  6
#define NDATA_PER_SAMP_RBUF  4
#define NBIT_SPECTRAL        32
#define NBYTE_SPECTRAL       4    // float
  
#define NPORT_MAX       10
#define NSAMP_DF        128
#define NCHAN_PER_CHUNK 7
#define OVER_SAMP_RATE  (32.0l/27.0)
#define DFSZ            7232
#define DF_HDRSZ        64
#define PERIOD          27
#define NDF_PER_CHUNK_PER_PERIOD 250000
#define NBEAM_MAX      36
#define NCHUNK_MAX     48
#define BAND_LIMIT_UP   1920.0
#define BAND_LIMIT_DOWN 640.0
  
#define SECDAY         86400.0
#define MJD1970        40587.0
#define NPROCESS_PER_NODE_MAX    2
#define CUFFT_RANK     1
#define MAX_RAND       1024

#define SCL_UINT8      64.0f          // uint8_t, detected samples should vary in 0.5 * range(uint8_t) = 127, to be safe, use 0.25
#define OFFS_UINT8     64.0f          // uint8_t, detected samples should center at 0.5 * range(uint8_t) = 127, to be safe, use 0.25
#define SCL_NSIG       3.0f
  
#endif
#ifdef __cplusplus
} 
#endif
