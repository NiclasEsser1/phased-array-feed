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

#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"

#define DADA_HDRSZ   4096
// It is a demo to check the sod, eod and start time of baseband2baseband
// It also helps to understand the control of baseband2baseband
// gcc -o baseband2baseband_demo baseband2baseband_demo.c -L/usr/local/cuda/lib64 -I/usr/local/include -lpsrdada -lcudart

#define MSTR_LEN   1024
void usage ()
{
  fprintf (stdout,
	   "baseband2baseband_demo - A demo to pass baseband data from a ring buffer to another ring buffer \n"
	   "\n"
	   "Usage: baseband2baseband_main [options]\n"
	   " -a  Hexacdecimal shared memory key for incoming ring buffer\n"
	   " -b  Hexacdecimal shared memory key for outcoming ring buffer\n"
	   " -c  The number of data frame (per frequency chunk) of each incoming ring buffer block\n"
	   " -d  The number of chunks\n"
	   " -e  The packet size\n"
	   " -h  show help\n");
}

int main(int argc, char **argv)
{
  int i, arg, pktsz, nchunk;
  key_t key_in, key_out;
  uint64_t ndf_chk, hdrsz, blksz, read_blksz, write_blksz;
  dada_hdu_t *hdu_in, *hdu_out;
  ipcbuf_t *db_in, *db_out;
  uint64_t write_blkid, read_blkid, curbufsz;
  char *hdrbuf_in, *hdrbuf_out;
  char *buf_in, *buf_out;
  struct timespec start, stop;
  double elapsed_time;
  multilog_t *runtime_log;
  
  /* Init */
  while((arg=getopt(argc,argv,"a:b:c:hd:e:")) != -1)
    {
      
      switch(arg)
	{
	case 'h':
	  usage();
	  return EXIT_FAILURE;
	  
	case 'a':	  	  
	  if(sscanf(optarg, "%x", &key_in) != 1)
	    {
	      fprintf(stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;

	case 'b':
	  if (sscanf (optarg, "%x", &key_out) != 1)
	    {
	      fprintf (stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;

	case 'c':
	  sscanf(optarg, "%"SCNu64"", &ndf_chk);
	  break;
	  
	case 'd':
	  sscanf(optarg, "%d", &nchunk);
	  break;
	  
	case 'e':
	  sscanf(optarg, "%d", &pktsz);
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
  hdu_in = dada_hdu_create(runtime_log);
  if(hdu_in == NULL)
    {
      fprintf(stdout, "HERE DADA_HDU_CREATE\n");
      exit(1);
    }
  dada_hdu_set_key(hdu_in, key_in);
  if(dada_hdu_connect(hdu_in))    
    {
      fprintf(stdout, "HERE DADA_HDU_CONNECT\n");
      exit(1);
    }
  if(dada_hdu_lock_read(hdu_in))
    {      
      fprintf(stdout, "HERE DADA_HDU_LOCK_READ\n");
      exit(1);
    }
  
  /* Prepare output ring buffer */
  hdu_out = dada_hdu_create(runtime_log);
  if(hdu_out == NULL)    
    {
      fprintf(stdout, "HERE DADA_HDU_CREATE\n");
      exit(1);
    }
  dada_hdu_set_key(hdu_out, key_out);
  if(dada_hdu_connect(hdu_out))      
    {
      fprintf(stdout, "HERE DADA_HDU_CONNECT\n");
      exit(1);
    }
  if(dada_hdu_lock_write(hdu_out))
    {      
      fprintf(stdout, "HERE DADA_HDU_LOCK_READ\n");
      exit(1);
    }

  /* To see if these two buffers are the same size */
  db_in  = (ipcbuf_t *)hdu_in->data_block;
  db_out = (ipcbuf_t *)hdu_out->data_block;
  fprintf(stdout, "IPCBUF_SOD:\t%d\n", ipcbuf_sod(db_in));
  fprintf(stdout, "IPCBUF_EOD:\t%d\n", ipcbuf_eod(db_in));
  
  ipcbuf_disable_sod(db_out);
  
  /* Pass the header to next ring buffer */
  hdrbuf_in  = ipcbuf_get_next_read(hdu_in->header_block, &hdrsz);  
  hdrbuf_out = ipcbuf_get_next_write(hdu_out->header_block);
  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header 
  ipcbuf_mark_filled(hdu_out->header_block, DADA_HDRSZ);
  ipcbuf_mark_cleared(hdu_in->header_block);

  buf_out = ipcbuf_get_next_write(db_out);
  read_blksz = 0;
  buf_in  = ipcbuf_get_next_read(db_in, &read_blksz);
  while(true)
    {
      memcpy(buf_out, buf_in, pktsz);
      ipcbuf_mark_cleared(db_in);
      ipcbuf_mark_filled(db_out, pktsz);

      fprintf(stdout, "IPCBUF_SOD:\t%d\n", ipcbuf_sod(db_in));
      fprintf(stdout, "IPCBUF_EOD:\t%d\n", ipcbuf_eod(db_in));
      fprintf(stdout, "IPCBUF_STATE:\t%d\n", db_in->state);
	
      if(!ipcbuf_sod(db_in))
	{
	  fprintf(stdout, "HERE EOD\t%d\n", (db_in->state));
	  fprintf(stdout, "IPCBUF_SOD:\t%d\n", ipcbuf_sod(db_in));
	  fprintf(stdout, "IPCBUF_EOD:\t%d\n", ipcbuf_eod(db_in));
	  //if(ipcbuf_enable_eod(db_out))
	  //break;
	  //ipcbuf_disable_sod(db_out);

	  hdrbuf_in  = ipcbuf_get_next_read(hdu_in->header_block, &hdrsz);
	  hdrbuf_out = ipcbuf_get_next_write(hdu_out->header_block);
	  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header
	  
	  if((hdrbuf_out == NULL) || (hdrbuf_in == NULL))
	    break;
	  ipcbuf_mark_filled(hdu_out->header_block, DADA_HDRSZ);
	  ipcbuf_mark_cleared(hdu_in->header_block);

	  fprintf(stdout, "IPCBUF_SOD:\t%d\t%"PRIu64"\n", ipcbuf_sod(db_in), ipcbuf_get_read_count(db_in));
	  fprintf(stdout, "IPCBUF_EOD:\t%d\t%"PRIu64"\n", ipcbuf_eod(db_in), ipcbuf_get_read_count(db_in));
	  
	  if(ipcbuf_enable_sod(db_out, ipcbuf_get_write_count(db_out), 0))
	    {
	      fprintf(stdout, "HERE SOD INSIDE\n");
	      break;
	    }
	  fprintf(stdout, "%d\n", ipcbuf_sod(db_in));
	  
	  read_blksz = 0;
	  buf_in  = ipcbuf_get_next_readable(db_in, &read_blksz);
	  buf_in  = ipcbuf_get_next_read(db_in, &read_blksz); 
	  buf_out = ipcbuf_get_next_write(db_out);
	  if(buf_out == NULL)
	    {
	      fprintf(stdout, "HERE AFTER OUT OPEN OUT\n");
	      break;
	    }
	  if(buf_in == NULL)
	    {
	      fprintf(stdout, "HERE AFTER OUT OPEN IN\n");
	      break;
	    }
	}
      else
	{
	  fprintf(stdout, "HERE BEFORE OPEN\t%d\n", (db_out->state));
	  buf_out = ipcbuf_get_next_write(db_out);

	  read_blksz = 0;
	  buf_in  = ipcbuf_get_next_read(db_in, &read_blksz);
	  
	  if(buf_out == NULL || buf_in == NULL)
	    break;
	}      
    }
  
  dada_hdu_unlock_write(hdu_out);    
  dada_hdu_disconnect(hdu_out);
  dada_hdu_destroy(hdu_out);
  
  dada_hdu_unlock_read(hdu_in);
  dada_hdu_disconnect(hdu_in);
  dada_hdu_destroy(hdu_in);

  fclose(fp_log);
  multilog(runtime_log, LOG_INFO, "BASEBAND2BASEBAND_DEMO END\n");
  
  return EXIT_SUCCESS;
}
