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
#define NCHAN_CHK      7
#define OSAMP_RATEI    0.84375

#define SECDAY         86400.0
#define MJD1970        40587.0
#define DFSZ           7232
#define PRD            27
#define NDF_CHK_PRD    250000
#define NCHK_FULL_BAND 48
  
#define CUFFT_RANK     1

#define MAX_RAND       1000
  
#endif

#ifdef __cplusplus
} 
#endif
