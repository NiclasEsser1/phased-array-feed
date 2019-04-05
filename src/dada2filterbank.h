#ifdef __cplusplus
extern "C" {
#endif

#ifndef __DADA2FILTERBANK_H
#define __DADA2FILTERBANK_H

#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"
#include "constants.h"
  
typedef struct conf_t
{
  int file;
  FILE *log_file;
  
  key_t key;
  dada_hdu_t *hdu;
  char f_fname[MSTR_LEN], d_fname[MSTR_LEN], source_name[MSTR_LEN], dir[MSTR_LEN], ra[MSTR_LEN], de[MSTR_LEN];
  ipcbuf_t *db;
  double mjd_start, freq, bw, tsamp, raj, dej, fch1, foff;
  uint64_t picoseconds;
  int nchans, nbits, npol, ndim, nifs, data_type, machine_id, telescope_id, ibeam, nbeams;
  FILE *d_fp, *f_fp;
}conf_t;

  int filterbank_header(conf_t conf);
  int dada_header(conf_t *conf);
  int filterbank_data(conf_t conf);
  int destroy(conf_t conf);
  int initialization(conf_t *conf);
  int write_string(FILE *fp, char *string);
  int write_int(FILE *fp, char *key, int value);
  int write_double(FILE *fp, char *key, double value);
#endif
#ifdef __cplusplus
} 
#endif
