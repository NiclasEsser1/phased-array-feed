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
  char hdrline[DADA_HDRSZ];
  int nread = 0;
  uint64_t hdrsz;
  char *hdrbuf = NULL;
  
  if (conf->file == 1)
    {
      /* Setup filterbank header parameters */
      while(strstr(hdrline, "end of header") == NULL)
	{
	  fgets(hdrline, sizeof(hdrline), conf->d_fp);
	  if(strstr(hdrline, "SOURCE"))
	    {
	      sscanf(hdrline, "%*s %s", conf->source_name);
	      fprintf(stdout, "%s\n", conf->source_name);
	      nread++;
	    }
	  if(strstr(hdrline, "MJD_START"))
	    {
	      sscanf(hdrline, "%*s %lf", &conf->mjd_start);
	      fprintf(stdout, "%f\n", conf->mjd_start);
	      nread++;
	    }      
	  if(strstr(hdrline, "PICOSECONDS"))
	    {
	      sscanf(hdrline, "%*s %lf", &conf->picoseconds);
	      fprintf(stdout, "%f\n", conf->picoseconds);
	      nread++;
	    }
	  if(strstr(hdrline, "FREQ"))
	    {
	      sscanf(hdrline, "%*s %lf", &conf->freq);
	      fprintf(stdout, "%f\n", conf->freq);
	      nread++;
	    }  
	  if(strstr(hdrline, "NCHAN"))
	    {
	      sscanf(hdrline, "%*s %d", &conf->nchans);
	      fprintf(stdout, "%d\n", conf->nchans);
	      nread++;
	    }   
	  if(strstr(hdrline, "NBIT"))
	    {
	      sscanf(hdrline, "%*s %d", &conf->nbits);
	      fprintf(stdout, "%d\n", conf->nbits);
	      nread++;
	    }   
	  if(strstr(hdrline, "NPOL"))
	    {
	      sscanf(hdrline, "%*s %d", &conf->npol);
	      fprintf(stdout, "%d\n", conf->npol);
	      nread++;
	    }     
	  if(strstr(hdrline, "NDIM"))
	    {
	      sscanf(hdrline, "%*s %d", &conf->ndim);
	      fprintf(stdout, "%d\n", conf->ndim);
	      nread++;
	    }   
	  if(strstr(hdrline, "TSAMP"))
	    {
	      sscanf(hdrline, "%*s %lf", &conf->tsamp);
	      fprintf(stdout, "%f\n", conf->tsamp);
	      nread++;
	    }   
	  if(strstr(hdrline, "BW"))
	    {
	      sscanf(hdrline, "%*s %lf", &conf->bw);
	      fprintf(stdout, "%f\n", conf->bw);
	      nread++;
	    }  
	  if(strstr(hdrline, "NBEAM"))
	    {
	      sscanf(hdrline, "%*s %d", &conf->nbeams);
	      fprintf(stdout, "%d\n", conf->nbeams);
	      nread++;
	    }   
	  if(strstr(hdrline, "IBEAM"))
	    {
	      sscanf(hdrline, "%*s %d", &conf->ibeam);
	      fprintf(stdout, "%d\n", conf->ibeam);
	      nread++;
	    }      
	}
    }
  else
    {
      hdrbuf = ipcbuf_get_next_read(conf->hdu->header_block, &hdrsz);
      fprintf(stdout, "%s\n\n\n\n", hdrbuf);
      sscanf(hdrbuf, "%s", hdrline);
      sscanf(hdrbuf, "%s", hdrline);
      fprintf(stdout, "%s\n", hdrline);
    }
    
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
  if(conf.file == 1)
    {
      fseek(conf.d_fp, DADA_HDRSZ, SEEK_SET);
      while(!feof(conf.d_fp))
	{
	  rsz = fread(buf, sizeof(char), PKTSZ, conf.d_fp);
	  wsz = fwrite(buf, sizeof(char), rsz, conf.f_fp);
	}
    }
  else
    {
      while(!ipcbuf_eod(conf.db))
	{
	  curbuf  = ipcbuf_get_next_read(conf.db, &rsz);
	  wsz = fwrite(curbuf, sizeof(char), rsz, conf.f_fp);
	  ipcbuf_mark_cleared(conf.db);      
	}
    }
  
  return EXIT_SUCCESS;
}

int destroy(conf_t conf)
{
  if(conf.file == 1)
    fclose(conf.d_fp);
  else
    {
      dada_hdu_unlock_write(conf.hdu);
      dada_hdu_disconnect(conf.hdu);
      dada_hdu_destroy(conf.hdu);
    }

  fclose(conf.f_fp);
  
  return EXIT_SUCCESS;
}

int initialization(conf_t *conf)
{
  uint64_t hdrsz;
  conf->f_fp = fopen(conf->f_fname, "w");
  
  if(conf->file == 1)
    conf->d_fp = fopen(conf->d_fname, "r");
  else
    {
      sscanf (conf->d_fname, "%x", &conf->key);
      conf->hdu = dada_hdu_create(runtime_log);
      dada_hdu_set_key(conf->hdu, conf->key);
      
      if(dada_hdu_connect(conf->hdu) < 0)
	{
	  multilog(runtime_log, LOG_ERR, "could not connect to hdu\n");
	  fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;    
	}  
      conf->db = (ipcbuf_t *) conf->hdu->data_block;      
      hdrsz = ipcbuf_get_bufsz(conf->hdu->header_block);  
      if(hdrsz != DADA_HDRSZ)    // This number should match
	{
	  multilog(runtime_log, LOG_ERR, "data buffer size mismatch\n");
	  fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;    
	}
      if(dada_hdu_lock_read(conf->hdu) < 0) // make ourselves the read client 
	{
	  multilog(runtime_log, LOG_ERR, "open_hdu: could not lock write\n");
	  fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}      
    }
  
  return EXIT_SUCCESS;
}
