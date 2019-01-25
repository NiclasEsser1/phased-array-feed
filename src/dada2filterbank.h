#ifndef __DADA2FILTERBANK_H
#define __DADA2FILTERBANK_H

#define MSTR_LEN      1024
#define DADA_HDRSZ    4096
#define PKTSZ         1048576 // 1GB

#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"

typedef struct conf_t
{
  key_t key;
  dada_hdu_t *hdu;
  char f_fname[MSTR_LEN], source_name[MSTR_LEN], dir[MSTR_LEN], ra[MSTR_LEN], dec[MSTR_LEN];
  ipcbuf_t *db;
  double mjd_start, picoseconds, freq, bw, tsamp, raj, decj, fch1, foff;
  int nchans, nbits, npol, ndim, nifs, data_type, machine_id, telescope_id, nbeams, ibeam;
  FILE *d_fp, *f_fp;
}conf_t;

int filterbank_header(conf_t conf);
int dada_header(conf_t *conf);
int filterbank_data(conf_t conf);
int destroy(conf_t conf);
int initialization(conf_t *conf);
#endif
