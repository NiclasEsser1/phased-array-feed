#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include "dada2filterbank.h"

extern multilog_t *runtime_log;

int filterbank_header(conf_t conf)
{
  char field[MSTR_LEN];
  int length;

  conf.telescope_id = 8;
  conf.data_type = 1;
  conf.machine_id = 0;
  
  /* Write filterbank header */
  length = 12;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "HEADER_START");
  fwrite(field, sizeof(char), length, conf.f_fp);

  length = 12;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "telescope_id");
  fwrite(field, sizeof(char), length, conf.f_fp);
  fwrite((char*)&conf.telescope_id, sizeof(int), 1, conf.f_fp);

  length = 9;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "data_type");
  fwrite(field, sizeof(char), length, conf.f_fp);
  fwrite((char*)&conf.data_type, sizeof(int), 1, conf.f_fp);

  length = 5;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "tsamp");
  fwrite(field, sizeof(char), length, conf.f_fp);
  fwrite((char*)&conf.tsamp, sizeof(double), 1, conf.f_fp);
  
  length = 6;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "tstart");
  fwrite(field, sizeof(char), length, conf.f_fp);
  fwrite((char*)&conf.mjd_start, sizeof(double), 1, conf.f_fp);

  length = 5;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "nbits");
  fwrite(field, sizeof(char), length, conf.f_fp);
  fwrite((char*)&conf.nbits, sizeof(int), 1, conf.f_fp);

  length = 4;
  conf.nifs = conf.npol * conf.ndim;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "nifs");
  fwrite(field, sizeof(char), length, conf.f_fp);
  fwrite((char*)&conf.nifs, sizeof(int), 1, conf.f_fp);
  
  length = 4;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "fch1");
  fwrite(field, sizeof(char), length, conf.f_fp);
  fwrite((char*)&conf.fch1, sizeof(double), 1, conf.f_fp);
  
  length = 4;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "foff");
  fwrite(field, sizeof(char), length, conf.f_fp);
  fwrite((char*)&conf.foff, sizeof(double), 1, conf.f_fp);

  length = 6;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "nchans");
  fwrite(field, sizeof(char), length, conf.f_fp);
  fwrite((char*)&conf.nchans, sizeof(int), 1, conf.f_fp);

  length = 11;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "source_name");
  fwrite(field, sizeof(char), length, conf.f_fp);
  length = strlen(conf.source_name);
  strncpy(field, conf.source_name, length);
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  fwrite(field, sizeof(char), length, conf.f_fp);

  length = 7;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "src_raj");
  fwrite(field, sizeof(char), length, conf.f_fp);
  length = strlen(conf.ra);
  strncpy(field, conf.ra, length);
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  fwrite(field, sizeof(char), length, conf.f_fp);

  length = 7;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "src_dej");
  fwrite(field, sizeof(char), length, conf.f_fp);
  length = strlen(conf.dec);
  strncpy(field, conf.dec, length);
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  fwrite(field, sizeof(char), length, conf.f_fp);

  length = 10;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "machine_id");
  fwrite(field, sizeof(char), length, conf.f_fp);
  fwrite((char*)&conf.machine_id, sizeof(int), 1, conf.f_fp);
  
  length = 6;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "nbeams");
  fwrite(field, sizeof(char), length, conf.f_fp);
  fwrite((char*)&conf.nbeams, sizeof(int), 1, conf.f_fp);
  
  length = 5;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "ibeam");
  fwrite(field, sizeof(char), length, conf.f_fp);
  fwrite((char*)&conf.ibeam, sizeof(int), 1, conf.f_fp);
  
  length = 10;
  fwrite((char*)&length, sizeof(int), 1, conf.f_fp);
  strcpy(field, "HEADER_END");
  fwrite(field, sizeof(char), length, conf.f_fp);

  return EXIT_SUCCESS;
}

int dada_header(conf_t *conf)
{
  uint64_t hdrsz;
  char *hdrbuf = NULL;
  
  hdrbuf = ipcbuf_get_next_read(conf->hdu->header_block, &hdrsz);
  ascii_header_get(hdrbuf, "NBIT", "%d", &conf->nbits);
  ascii_header_get(hdrbuf, "SOURCE", "%s", conf->source_name);
  ascii_header_get(hdrbuf, "RA", "%s", conf->ra);
  ascii_header_get(hdrbuf, "DEC", "%s", conf->dec);
  ascii_header_get(hdrbuf, "MJD_START", "%lf", &conf->mjd_start);
  ascii_header_get(hdrbuf, "PICOSECONDS", "%lf", &conf->picoseconds);
  ascii_header_get(hdrbuf, "FREQ", "%lf", &conf->freq);
  ascii_header_get(hdrbuf, "NCHAN", "%d", &conf->nchans);
  ascii_header_get(hdrbuf, "NPOL", "%d", &conf->npol);
  ascii_header_get(hdrbuf, "NDIM", "%d", &conf->ndim);
  ascii_header_get(hdrbuf, "TSAMP", "%lf", &conf->tsamp);
  ascii_header_get(hdrbuf, "BW", "%lf", &conf->bw);
  ascii_header_get(hdrbuf, "NBEAM", "%d", &conf->nbeams);
  ascii_header_get(hdrbuf, "IBEAM", "%d", &conf->ibeam);
  
  ipcbuf_mark_cleared(conf->hdu->header_block);
  
  conf->mjd_start = conf->mjd_start + conf->picoseconds / 86400.0E12;
  conf->bw        = -256.0 * 32.0 / 27.0;
  conf->fch1      = conf->freq - 0.5 * conf->bw / conf->nchans * (conf->nchans - 1);
  conf->foff      = conf->bw / conf->nchans;
  conf->tsamp     = conf->tsamp / 1.0E6;
  fprintf(stdout, "%.10f\t%.10f\t%.10f\t%.10f\n", conf->mjd_start, conf->bw, conf->fch1, conf->foff);

  return EXIT_SUCCESS;
}

int filterbank_data(conf_t conf)
{
  char buf[PKTSZ];
  size_t rsz, wsz;
  char *curbuf = NULL;
  
  /* Copy data from DADA file to filterbank file */
  while(!ipcbuf_eod(conf.db))
    {
      curbuf  = ipcbuf_get_next_read(conf.db, &rsz);
      wsz = fwrite(curbuf, sizeof(char), rsz, conf.f_fp);
      ipcbuf_mark_cleared(conf.db);      
    }
      
  return EXIT_SUCCESS;
}

int destroy(conf_t conf)
{
  dada_hdu_unlock_read(conf.hdu);
  dada_hdu_disconnect(conf.hdu);
  dada_hdu_destroy(conf.hdu);

  fclose(conf.f_fp);
  
  return EXIT_SUCCESS;
}

int initialization(conf_t *conf)
{
  uint64_t hdrsz;
  conf->f_fp = fopen(conf->f_fname, "w");
  
  conf->hdu = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu, conf->key);
  
  if(dada_hdu_connect(conf->hdu) < 0)
    {
      multilog(runtime_log, LOG_ERR, "could not connect to hdu\n");
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      exit(EXIT_FAILURE);    
    }  
  conf->db = (ipcbuf_t *) conf->hdu->data_block;      
  hdrsz = ipcbuf_get_bufsz(conf->hdu->header_block);  
  if(hdrsz != DADA_HDRSZ)    // This number should match
    {
      multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      exit(EXIT_FAILURE);    
    }
  if(dada_hdu_lock_read(conf->hdu) < 0) // make ourselves the read client 
    {
      multilog(runtime_log, LOG_ERR, "open_hdu: could not lock write\n");
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }      
      
  return EXIT_SUCCESS;
}
