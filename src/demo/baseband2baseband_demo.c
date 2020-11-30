#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <sys/socket.h>
#include <linux/un.h>
#include <pthread.h>

#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"

#define MSTR_LEN     1024
#define DADA_HDRSZ   4096
#define SECDAY       86400.0
#define MJD1970      40587.0

// It is a demo to check the sod, eod and start time of baseband2baseband
// It also helps to understand the control of baseband2baseband
// gcc -o baseband2baseband_demo baseband2baseband_demo.c -L/usr/local/cuda/lib64 -I/usr/local/include -lpsrdada -lcudart -lm

multilog_t *runtime_log;

typedef struct conf_t
{
  dada_hdu_t *hdu_in, *hdu_out;
  key_t key_in, key_out;
  char *curbuf_in, *curbuf_out;
  int pktsz;
}conf_t;

int baseband2baseband(conf_t conf);

void usage ()
{
  fprintf (stdout,
	   "baseband2baseband_demo - A demo to pass baseband data from a ring buffer to another ring buffer \n"
	   "\n"
	   "Usage: baseband2baseband_main [options]\n"
	   " -a  Hexacdecimal shared memory key for incoming ring buffer\n"
	   " -b  Hexacdecimal shared memory key for outcoming ring buffer\n"
	   " -c  The packet size\n"
	   " -h  show help\n");
}

int main(int argc, char **argv)
{
  conf_t conf;
  int i, arg, pktsz;
  
  /* Init */
  while((arg=getopt(argc,argv,"a:b:hc:")) != -1)
    {
      
      switch(arg)
	{
	case 'h':
	  usage();
	  return EXIT_FAILURE;
	  
	case 'a':	  	  
	  if(sscanf(optarg, "%x", &conf.key_in) != 1)
	    {
	      fprintf(stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;

	case 'b':
	  if (sscanf (optarg, "%x", &conf.key_out) != 1)
	    {
	      fprintf (stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;

	case 'c':
	  sscanf(optarg, "%d", &conf.pktsz);
	  break;	  	  
	}
    }
  char fname_log[MSTR_LEN];
  FILE *fp_log = NULL;
  sprintf(fname_log, "/beegfs/DENG/docker/baseband2baseband_demo.log");
  fp_log = fopen(fname_log, "ab+"); 
  if(fp_log == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", fname_log);
      return EXIT_FAILURE;
    }
  runtime_log = multilog_open("baseband2baseband_demo", 1);
  multilog_add(runtime_log, fp_log);
  multilog(runtime_log, LOG_INFO, "BASEBAND2BASEBAND_DEMO START\n");
  
  /* attach to input ring buffer */
  conf.hdu_in = dada_hdu_create(runtime_log);
  if(conf.hdu_in == NULL)
    {
      fprintf(stdout, "HERE DADA_HDU_CREATE\n");
      exit(1);
    }
  dada_hdu_set_key(conf.hdu_in, conf.key_in);
  if(dada_hdu_connect(conf.hdu_in))    
    {
      fprintf(stdout, "HERE DADA_HDU_CONNECT\n");
      exit(1);
    }
  if(dada_hdu_lock_read(conf.hdu_in))
    {      
      fprintf(stdout, "HERE DADA_HDU_LOCK_READ\n");
      exit(1);
    }
  
  /* Prepare output ring buffer */
  conf.hdu_out = dada_hdu_create(runtime_log);
  if(conf.hdu_out == NULL)    
    {
      fprintf(stdout, "HERE DADA_HDU_CREATE\n");
      exit(1);
    }
  dada_hdu_set_key(conf.hdu_out, conf.key_out);
  if(dada_hdu_connect(conf.hdu_out))      
    {
      fprintf(stdout, "HERE DADA_HDU_CONNECT\n");
      exit(1);
    }
  if(dada_hdu_lock_write(conf.hdu_out))
    {      
      fprintf(stdout, "HERE DADA_HDU_LOCK_READ\n");
      exit(1);
    }
  
  /* Do the real job */
  baseband2baseband(conf);
  
  /* Destroy */
  dada_hdu_unlock_write(conf.hdu_out);    
  dada_hdu_disconnect(conf.hdu_out);
  dada_hdu_destroy(conf.hdu_out);
  
  dada_hdu_unlock_read(conf.hdu_in);
  dada_hdu_disconnect(conf.hdu_in);
  dada_hdu_destroy(conf.hdu_in);

  multilog(runtime_log, LOG_INFO, "BASEBAND2BASEBAND_DEMO END\n");
  fclose(fp_log);
  
  return EXIT_SUCCESS;
}

int baseband2baseband(conf_t conf)
{
  ipcbuf_t *db_in = NULL, *db_out = NULL;
  ipcbuf_t *hdr_in = NULL, *hdr_out = NULL;
  uint64_t hdrsz;
  double mjdstart_ref;
  char *hdrbuf_in = NULL, *hdrbuf_out = NULL;
  uint64_t curbufsz;
  
  /* To see if these two buffers are the same size */
  db_in  = (ipcbuf_t *)conf.hdu_in->data_block;
  db_out = (ipcbuf_t *)conf.hdu_out->data_block;
  hdr_in  = (ipcbuf_t *)conf.hdu_in->header_block;
  hdr_out  = (ipcbuf_t *)conf.hdu_out->header_block;
  
  hdrbuf_in = ipcbuf_get_next_read(hdr_in, &hdrsz);
  if(!hdrbuf_in)
    {
      multilog(runtime_log, LOG_ERR, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  hdrbuf_out = ipcbuf_get_next_write(hdr_out);
  if(!hdrbuf_out)
    {
      multilog(runtime_log, LOG_ERR, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ);  // Get a copy of the header
  ipcbuf_mark_filled(hdr_out, DADA_HDRSZ);
  ipcbuf_mark_cleared(hdr_in);
  
  conf.curbuf_in  = ipcbuf_get_next_read(db_in, &curbufsz);
  conf.curbuf_out = ipcbuf_get_next_write(db_out);

  while(!ipcbuf_eod(db_in))
    {
      memcpy(conf.curbuf_out, conf.curbuf_in, conf.pktsz);
      
      ipcbuf_mark_filled(db_out, conf.pktsz);
      ipcbuf_mark_cleared(db_in);
      conf.curbuf_in  = ipcbuf_get_next_read(db_in, &curbufsz);
      conf.curbuf_out = ipcbuf_get_next_write(db_out);
      
      fprintf(stdout, "HERE EOD\t%d\t", ipcbuf_eod(db_in));
      fprintf(stdout, "HERE SOD\t%d\n", ipcbuf_sod(db_in));
      fprintf(stdout, "HERE EOD\t%d\t", ipcbuf_eod(db_in));
      fprintf(stdout, "HERE SOD\t%d\n\n", ipcbuf_sod(db_in));

    }
  return EXIT_SUCCESS;
}
