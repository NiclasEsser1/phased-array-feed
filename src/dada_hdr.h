#ifdef __cplusplus
extern "C" {
#endif
#ifndef _DADA_HDR_H
#define _DADA_HDR_H

#include "constants.h"
  
  typedef struct dada_hdr_t
  {
    char obs_id[MSTR_LEN];
    char primary[MSTR_LEN];
    char secondary[MSTR_LEN];
    char file_name[MSTR_LEN];

    uint64_t file_size;
    int file_number;
    char utc_start[MSTR_LEN];
    double mjd_start;
    uint64_t picoseconds;

    uint64_t obs_offset;
    uint64_t obs_overlap;

    char source[MSTR_LEN];
    char ra[MSTR_LEN];
    char dec[MSTR_LEN];
    char telescope[MSTR_LEN];
    char instrument[MSTR_LEN];
    char receiver[MSTR_LEN];

    double freq;
    double bw;
    double tsamp;

    uint64_t bytes_per_second;
    int nbit;
    int ndim;
    int npol;
    int nchan;
    
    int resolution;
    int dsb;	
  }dada_hdr_t;
  
#endif
  
#ifdef __cplusplus
} 
#endif
