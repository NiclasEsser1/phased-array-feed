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
// gcc -o baseband2baseband baseband2baseband.c -L/usr/local/cuda/lib64 -I/usr/local/include -lpsrdada -lcudart

void usage ()
{
  fprintf (stdout,
	   "baseband2baseband - A demo to pass baseband data from a ring buffer to another ring buffer \n"
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
  int i, arg, pktsz;
  key_t key_in, key_out;
  uint64_t ndf_chk, hdrsz;
  dada_hdu_t *hdu_in, *hdu_out;
  ipcbuf_t *db;
  uint64_t write_blkid, read_blkid, curbufsz;
  char *hdrbuf_in, *hdrbuf_out;
  
  /* Init */
  while((arg=getopt(argc,argv,"a:b:c:hd:")) != -1)
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
	  sscanf(optarg, "%d", &pktsz);
	  break;
	}
    }
  
  /* attach to input ring buffer */
  hdu_in = dada_hdu_create(NULL);
  dada_hdu_set_key(hdu_in, key_in);
  dada_hdu_connect(hdu_in);
  dada_hdu_lock_read(hdu_in);    
  
  /* Prepare output ring buffer */
  hdu_out = dada_hdu_create(NULL);
  dada_hdu_set_key(hdu_out, key_out);
  dada_hdu_connect(hdu_out);
  dada_hdu_lock_write(hdu_out);

  /* To see if these two buffers are the same size */
  if(!(ipcbuf_get_bufsz((ipcbuf_t *)hdu_in->data_block) == ipcbuf_get_bufsz((ipcbuf_t *)hdu_out->data_block)))
    {
      fprintf(stderr, "Input and output buffer size is not match, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  /* Pass the header to next ring buffer */
  hdrbuf_in  = ipcbuf_get_next_read(hdu_in->header_block, &hdrsz);  
  hdrbuf_out = ipcbuf_get_next_write(hdu_out->header_block);
  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header 
  ipcbuf_mark_filled(hdu_in->header_block, DADA_HDRSZ);
  ipcbuf_mark_cleared(hdu_in->header_block);

  /* Loop */ 
  ipcio_open_block_write(hdu_out->data_block, &write_blkid);   /* Open buffer to write */
  ipcio_open_block_read(hdu_in->data_block, &curbufsz, &read_blkid);
  memcpy(hdu_out->data_block->curbuf, hdu_in->data_block->curbuf, curbufsz);
  while(true)
    {
      ipcio_close_block_write(hdu_out->data_block, curbufsz);  
      ipcio_close_block_read(hdu_in->data_block, hdu_in->data_block->curbufsz);
      
      if(ipcbuf_eod((ipcbuf_t *)hdu_in->data_block) > 0)
	{
	  ipcbuf_enable_eod((ipcbuf_t *)hdu_out->data_block);
	  
	  hdrbuf_in  = ipcbuf_get_next_read(hdu_in->header_block, &hdrsz);  
	  hdrbuf_out = ipcbuf_get_next_write(hdu_out->header_block);
	  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header 
	  ipcbuf_mark_filled(hdu_in->header_block, DADA_HDRSZ);
	  ipcbuf_mark_cleared(hdu_in->header_block);
    	  
	  ipcio_open_block_write(hdu_out->data_block, &write_blkid);   /* Open buffer to write */
	  ipcio_open_block_read(hdu_in->data_block, &curbufsz, &read_blkid);
	  memcpy(hdu_out->data_block->curbuf, hdu_in->data_block->curbuf, curbufsz);
	  
	  ipcbuf_enable_sod((ipcbuf_t *)hdu_out->data_block, write_blkid, 0);
	}
      else
	{
	  ipcio_open_block_write(hdu_out->data_block, &write_blkid);   /* Open buffer to write */
	  ipcio_open_block_read(hdu_in->data_block, &curbufsz, &read_blkid);
	  memcpy(hdu_out->data_block->curbuf, hdu_in->data_block->curbuf, curbufsz);
	}
    }

  return EXIT_SUCCESS;
}
