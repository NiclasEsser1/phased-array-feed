#ifdef __cplusplus
extern "C" {
#endif
#ifndef _PSRDADA_H
#define _PSRDADA_H

#include "constants.h"
#include "dada_hdu.h"
#include "ipcbuf.h"
#include "dada_cuda.h"
  
  typedef struct dada_header_t
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
  }dada_header_t;

  int read_dada_header_from_buffer(ipcbuf_t *hdr, dada_header_t *dada_header);
  int read_dada_header_from_file(char *fname, dada_header_t *dada_header);
  int read_dada_header_work(char *hdrbuf, dada_header_t *dada_header);
  int write_dada_header(ipcbuf_t *hdr, dada_header_t dada_header);
  dada_hdu_t* dada_hdu_create_wrap(key_t key, int write, int dbregister);
  int dada_hdu_destroy_wrap(dada_hdu_t *hdu, key_t key, int write, int dbregister);
  
#endif
  
#ifdef __cplusplus
} 
#endif
