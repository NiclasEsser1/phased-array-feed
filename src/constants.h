#ifdef __cplusplus
extern "C" {
#endif
#ifndef _CONSTANTS_CUH
#define _CONSTANTS_CUH

#define MSTR_LEN       1024
#define DADA_HDRSZ     4096

#define NPOL_IN        2
#define NDIM_IN        2
#define NBYTE_IN       2    // int16_t
#define NSAMP_DF       128
#define NCHAN_PER_CHUNK 7
#define OVER_SAMP_RATE  (32.0l/27.0)

#define SECDAY         86400.0
#define MJD1970        40587.0
#define DFSZ           7232
#define DF_HDRSZ       64
#define PERIOD         27
#define NDF_PER_CHUNK_PER_PERIOD 250000
#define NCHUNK_FULL_BAND         48
#define NPROCESS_PER_NODE_MAX    2
#define CUFFT_RANK     1
#define NCPU_PER_NUMA_NODE 10
#define MAX_RAND       1024
#define BAND_LIMIT_UP   640.0
#define BAND_LIMIT_DOWN 1920.0

#endif

#ifdef __cplusplus
} 
#endif
