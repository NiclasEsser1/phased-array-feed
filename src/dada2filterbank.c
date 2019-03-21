#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include "log.h"
#include "constants.h"
#include "dada2filterbank.h"
pthread_mutex_t log_mutex;

int filterbank_header(conf_t conf)
{
  /* If the data is generated in a different machine, we may have litter and big proglem */
  char field[MSTR_LEN];
  int length;

  conf.telescope_id = 8;
  conf.data_type = 1;
  conf.machine_id = 2;
    
  /* Write filterbank header */
  length = 12;
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  strcpy(field, "HEADER_START");
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
    
  length = 11;
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  strcpy(field, "rawdatafile");
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  fwrite(conf.f_fname, NBYTE_INT, 1, conf.f_fp);
  fprintf(stdout, "%s\n", conf.f_fname);
  fflush(stdout);
  
  length = 12;
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  strcpy(field, "telescope_id");
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  fwrite((char*)&conf.telescope_id, NBYTE_INT, 1, conf.f_fp);
  fprintf(stdout, "%d\n", conf.telescope_id);
  fflush(stdout);
    
  length = 9;
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  strcpy(field, "data_type");
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  fwrite((char*)&conf.data_type, NBYTE_INT, 1, conf.f_fp);
  fprintf(stdout, "%d\n", conf.data_type);
  fflush(stdout);
    
  //length = 5;
  //double refdm = 196.0;
  //fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  //strcpy(field, "refdm");
  //fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  //fwrite((char*)&refdm, NBYTE_INT, 1, conf.f_fp);
  //fprintf(stdout, "%f\n", refdm);
  //fflush(stdout);
	
  length = 5;
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  strcpy(field, "tsamp");
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  fwrite((char*)&conf.tsamp, sizeof(double), 1, conf.f_fp);
  fprintf(stdout, "%f\n", conf.tsamp);
  fflush(stdout);
  
  length = 6;
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  strcpy(field, "tstart");
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  fwrite((char*)&conf.mjd_start, sizeof(double), 1, conf.f_fp);
  fprintf(stdout, "%f\n", conf.mjd_start);
  fflush(stdout);
  
  length = 5;
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  strcpy(field, "nbits");
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  fwrite((char*)&conf.nbits, NBYTE_INT, 1, conf.f_fp);
  fprintf(stdout, "%d\n", conf.nbits);
  fflush(stdout);
  
  length = 4;
  conf.nifs = conf.npol * conf.ndim;
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  strcpy(field, "nifs");
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  fwrite((char*)&conf.nifs, NBYTE_INT, 1, conf.f_fp);
  fprintf(stdout, "%d\n", conf.nifs);
  fflush(stdout);
  
  length = 4;
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  strcpy(field, "fch1");
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  fwrite((char*)&conf.fch1, sizeof(double), 1, conf.f_fp);
  fprintf(stdout, "%f\n", conf.fch1);
  fflush(stdout);
  
  length = 4;
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  strcpy(field, "foff");
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  fwrite((char*)&conf.foff, sizeof(double), 1, conf.f_fp);  
  fprintf(stdout, "%f\n", conf.foff);
  fflush(stdout);
  
  length = 6;
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  strcpy(field, "nchans");
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  fwrite((char*)&conf.nchans, NBYTE_INT, 1, conf.f_fp);
  fprintf(stdout, "%d\n", conf.nchans);
  fflush(stdout);
  
  //length = 6;
  //int nbeams = 1;
  //fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  //strcpy(field, "nbeams");
  //fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  //fwrite((char*)&nbeams, NBYTE_INT, 1, conf.f_fp);
  //fprintf(stdout, "%d\n", nbeams);
  //fflush(stdout);
  //
  //length = 5;
  //int ibeam = 1;
  //fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  //strcpy(field, "ibeam");
  //fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  //fwrite((char*)&ibeam, NBYTE_INT, 1, conf.f_fp);
  //fprintf(stdout, "%d\n", ibeam);
  //fflush(stdout);
  
  length = 11;
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  strcpy(field, "source_name");
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  length = strlen(conf.source_name);
  strncpy(field, conf.source_name, length);
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  fprintf(stdout, "%s\n", conf.source_name);
  fflush(stdout);
  
  //length = 7;
  //conf.raj = 30;
  //fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  //strcpy(field, "src_raj");
  //fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  //fwrite((char*)&conf.raj, sizeof(double), 1, conf.f_fp);
  //fprintf(stdout, "%f\n", conf.raj);
  //fflush(stdout);
  //
  //length = 7;
  //conf.decj = 10;
  //fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  //strcpy(field, "src_dej");
  //fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  //fwrite((char*)&conf.decj, sizeof(double), 1, conf.f_fp); 
  //fprintf(stdout, "%f\n", conf.decj);
  //fflush(stdout);
  
  length = 10;
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  strcpy(field, "machine_id");
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
  fwrite((char*)&conf.machine_id, NBYTE_INT, 1, conf.f_fp);
  fprintf(stdout, "%d\n", conf.machine_id);
  fflush(stdout);
    
  length = 10;
  fwrite((char*)&length, NBYTE_INT, 1, conf.f_fp);
  strcpy(field, "HEADER_END");
  fwrite(field, NBYTE_CHAR, length, conf.f_fp);
    
  return EXIT_SUCCESS;
}

int dada_header(conf_t *conf)
{
  uint64_t hdrsz;
  char *hdrbuf = NULL;
  char buf[DADA_HDRSZ];
  
  if(conf->file == 0)
    hdrbuf = ipcbuf_get_next_read(conf->hdu->header_block, &hdrsz);
  else
    {
      fread(buf, 1, DADA_HDRSZ, conf->d_fp);
      //fileread(conf->d_fname, hdrbuf, DADA_HDRSZ);
      hdrbuf = buf;
    }
  ascii_header_get(hdrbuf, "NBIT", "%d", &conf->nbits);
  ascii_header_get(hdrbuf, "SOURCE", "%s", conf->source_name);
  ascii_header_get(hdrbuf, "RA", "%lf", conf->ra);
  ascii_header_get(hdrbuf, "DEC", "%lf", conf->dec);
  ascii_header_get(hdrbuf, "MJD_START", "%lf", &conf->mjd_start);
  ascii_header_get(hdrbuf, "PICOSECONDS", "%lf", &conf->picoseconds);
  ascii_header_get(hdrbuf, "FREQ", "%lf", &conf->freq);
  ascii_header_get(hdrbuf, "NCHAN", "%d", &conf->nchans);
  ascii_header_get(hdrbuf, "NPOL", "%d", &conf->npol);
  ascii_header_get(hdrbuf, "NDIM", "%d", &conf->ndim);
  ascii_header_get(hdrbuf, "TSAMP", "%lf", &conf->tsamp);
  ascii_header_get(hdrbuf, "BW", "%lf", &conf->bw);
  ascii_header_get(hdrbuf, "RECEIVER", "%d", &conf->beam_id);

  if(conf->file == 0)
    ipcbuf_mark_cleared(conf->hdu->header_block);
  else
    fseek(conf->d_fp, DADA_HDRSZ, SEEK_SET);
  
  conf->mjd_start = conf->mjd_start + conf->picoseconds / 86400.0E12;
  conf->fch1      = conf->freq - 0.5 * conf->bw / conf->nchans * (conf->nchans - 1);
  conf->foff      = conf->bw / conf->nchans;
  conf->tsamp     = conf->tsamp / 1.0E6;
  fprintf(stdout, "%.10f\t%.10f\t%.10f\t%.10f\t%f\n", conf->mjd_start, conf->bw, conf->fch1, conf->foff, conf->tsamp);

  return EXIT_SUCCESS;
}

int filterbank_data(conf_t conf)
{
  char buf[TRANS_BUFSZ];
  size_t rsz, wsz;
  char *curbuf = NULL;
  
  /* Copy data from DADA file to filterbank file */
  if(conf.file == 0)
    {
      while(!ipcbuf_eod(conf.db))
	{
	  curbuf  = ipcbuf_get_next_read(conf.db, &rsz);
	  wsz = fwrite(curbuf, NBYTE_CHAR, rsz, conf.f_fp);
	  ipcbuf_mark_cleared(conf.db);      
	}
    }
  else
    {
      while(!feof(conf.d_fp))
	{
	  rsz = fread(buf, 1, TRANS_BUFSZ, conf.d_fp);
	  fwrite(buf, 1, rsz, conf.f_fp);
	}
    }
  
  return EXIT_SUCCESS;
}

int destroy(conf_t conf)
{
  if(conf.file==0)
    {
      dada_hdu_unlock_read(conf.hdu);
      dada_hdu_destroy(conf.hdu);
    }
  else
    fclose(conf.d_fp);
  
  fclose(conf.f_fp);
  
  return EXIT_SUCCESS;
}

int initialization(conf_t *conf)
{
  uint64_t hdrsz;
  conf->f_fp = fopen(conf->f_fname, "w");

  if(conf->file == 0)
    {
      conf->hdu = dada_hdu_create(NULL);
      dada_hdu_set_key(conf->hdu, conf->key);
      
      if(dada_hdu_connect(conf->hdu) < 0)
	{
	  log_add(conf->log_file, "INFO", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	  fprintf(stderr, "DADA2FILTERBANK_ERROR: Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  exit(EXIT_FAILURE);    
	}  
      conf->db = (ipcbuf_t *) conf->hdu->data_block;      
      hdrsz = ipcbuf_get_bufsz(conf->hdu->header_block);  
      if(hdrsz != DADA_HDRSZ)    // This number should match
	{
	  log_add(conf->log_file, "INFO", 1, log_mutex, "data buffer size mismatch, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	  fprintf(stderr, "DADA2FILTERBANK_ERROR: Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  exit(EXIT_FAILURE);    
	}
      if(dada_hdu_lock_read(conf->hdu) < 0) // make ourselves the read client 
	{
	  log_add(conf->log_file, "INFO", 1, log_mutex, "open_hdu: could not lock write, which happens at \"%s\", line [%d].", __FILE__, __LINE__);
	  fprintf(stderr, "DADA2FILTERBANK_ERROR: Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  exit(EXIT_FAILURE);
	}      
    }
  else
    conf->d_fp = fopen(conf->d_fname, "rb");
    
  return EXIT_SUCCESS;
}
