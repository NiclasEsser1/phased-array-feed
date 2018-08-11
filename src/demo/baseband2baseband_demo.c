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
  ipcbuf_t *db;
  uint64_t write_blkid, read_blkid, curbufsz;
  char *hdrbuf_in, *hdrbuf_out;
  char *buf_in, *buf_out;
  struct timespec start, stop;
  double elapsed_time;
  
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
  blksz = pktsz * nchunk * ndf_chk;
  
  /* attach to input ring buffer */
  hdu_in = dada_hdu_create(NULL);
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
  hdu_out = dada_hdu_create(NULL);
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
  read_blksz = ipcbuf_get_bufsz((ipcbuf_t *)hdu_in->data_block);
  write_blksz = ipcbuf_get_bufsz((ipcbuf_t *)hdu_out->data_block);
  //if(!(read_blksz == write_blksz) || !(blksz == write_blksz) || !(read_blksz == blksz) )
  //  {
  //    fprintf(stderr, "Input and output buffer size is not match, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    return EXIT_FAILURE;
  //  }

  /* Pass the header to next ring buffer */
  hdrbuf_in  = ipcbuf_get_next_read(hdu_in->header_block, &hdrsz);  
  hdrbuf_out = ipcbuf_get_next_write(hdu_out->header_block);
  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header 
  ipcbuf_mark_filled(hdu_out->header_block, DADA_HDRSZ);
  ipcbuf_mark_cleared(hdu_in->header_block);

  //ipcbuf_enable_sod((ipcbuf_t *)hdu_out->data_block, 0, 0);
  //fprintf(stdout, "HERE ENABLE_EOD\t%d\n", ((ipcbuf_t *)(hdu_out->data_block))->state);
  buf_out = ipcbuf_get_next_write((ipcbuf_t*)hdu_out->data_block);
  buf_in  = ipcbuf_get_next_read((ipcbuf_t*)hdu_in->data_block, &read_blksz);
  //fprintf(stdout, "HERE %"PRIu64"\t%"PRIu64"\n", blksz, ipcbuf_get_bufsz((ipcbuf_t *)hdu_in->data_block));
  while(true)
    {
      memcpy(buf_out, buf_in, pktsz);
      //fprintf(stdout, "HERE BEFORE CLEAR\t%d\n", ((ipcbuf_t *)(hdu_out->data_block))->state);
      ipcbuf_mark_cleared((ipcbuf_t *)hdu_in->data_block);
      //fprintf(stdout, "HERE AFTER CLEAR1\t%d\n", ((ipcbuf_t *)(hdu_out->data_block))->state);
      //ipcbuf_mark_filled((ipcbuf_t *)hdu_out->data_block, blksz);
      ipcbuf_mark_filled((ipcbuf_t *)hdu_out->data_block, pktsz);
      //fprintf(stdout, "HERE AFTER CLEAR2\t%d\n", ((ipcbuf_t *)(hdu_out->data_block))->state);
      //if(ipcbuf_eod((ipcbuf_t *)hdu_in->data_block) > 0)
      //fprintf(stdout, "SOD IPCBUF\t%d\n", (ipcbuf_sod((ipcbuf_t *)hdu_in->data_block)));
      if(!(ipcbuf_sod((ipcbuf_t *)hdu_in->data_block)))
	{
	  fprintf(stdout, "HERE EOD\t%d\n", ((ipcbuf_t *)(hdu_out->data_block))->state);
	  if(ipcbuf_enable_eod((ipcbuf_t *)hdu_out->data_block))
	    return EXIT_FAILURE;
	  ipcbuf_disable_sod((ipcbuf_t *)hdu_out->data_block);
	  
	  hdrbuf_in  = ipcbuf_get_next_read(hdu_in->header_block, &hdrsz);  
	  hdrbuf_out = ipcbuf_get_next_write(hdu_out->header_block);
	  memcpy(hdrbuf_out, hdrbuf_in, DADA_HDRSZ); // Pass the header
	  
	  if(hdrbuf_out == NULL)
	    return EXIT_FAILURE;
	  if (hdrbuf_in == NULL)
	    return EXIT_FAILURE;
	  ipcbuf_mark_filled(hdu_out->header_block, DADA_HDRSZ);
	  ipcbuf_mark_cleared(hdu_in->header_block);
	  fprintf(stdout, "HERE AFTER HDR\n");
	  
	  //ipcbuf_enable_sod((ipcbuf_t *)hdu_out->data_block, ipcbuf_get_write_count((ipcbuf_t *)hdu_out->data_block), 0);
	  fprintf(stdout, "HERE AFTER ENABLE_SOD\n");

	  buf_in  = ipcbuf_get_next_read((ipcbuf_t*)hdu_in->data_block, &blksz);
	  if(buf_out == NULL)
	    {
	      fprintf(stdout, "HERE AFTER IN OPEN\n");
	      return EXIT_FAILURE;
	    }
	  //fprintf(stdout, "HERE AFTER IN OPEN\t%"PRIu64"\n", ipcbuf_get_nfull((ipcbuf_t*)hdu_out->data_block));
	  fprintf(stdout, "HERE AFTER IN OPEN\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%d\n", ipcbuf_get_nfull((ipcbuf_t*)hdu_out->data_block), ((ipcbuf_t*)hdu_out->data_block)->sync->w_buf, ((ipcbuf_t*)hdu_out->data_block)->sync->nbufs, ((ipcbuf_t*)hdu_out->data_block)->count[(((ipcbuf_t*)hdu_out->data_block)->sync->w_buf)%(((ipcbuf_t*)hdu_out->data_block)->sync->nbufs)]);
	  //return EXIT_FAILURE;
	  
	  buf_out = ipcbuf_get_next_write((ipcbuf_t*)hdu_out->data_block);
	  if(ipcbuf_enable_sod((ipcbuf_t *)hdu_out->data_block, ipcbuf_get_write_count((ipcbuf_t *)hdu_out->data_block), 0))
	    {
	      fprintf(stdout, "HERE SOD INSIDE\n");
	      return EXIT_FAILURE;
	    }
	  
	  fprintf(stdout, "HERE after get_next_write\n");
	  //return EXIT_FAILURE;
	  
	  if(buf_out == NULL)
	    {
	      fprintf(stdout, "HERE AFTER OUT OPEN\n");
	      return EXIT_FAILURE;
	    }
	  fprintf(stdout, "HERE AFTER OUT OPEN\n");
	  
	  //if((buf_out == NULL) || (buf_in == NULL))
	  //  {
	  //    //fprintf(stdout, "HERE AFTER OUT OPEN\n");
	  //    return EXIT_FAILURE;
	  //  }
	}
      else
	{
	  fprintf(stdout, "HERE BEFORE OPEN\t%d\n", ((ipcbuf_t *)(hdu_out->data_block))->state);
	  buf_out = ipcbuf_get_next_write((ipcbuf_t*)hdu_out->data_block);
	  buf_in  = ipcbuf_get_next_read((ipcbuf_t*)hdu_in->data_block, &blksz);
	  if(buf_out == NULL) 
	    return EXIT_FAILURE;
	  if(buf_in == NULL)
	    return EXIT_FAILURE;
	  
	  //fprintf(stdout, "HERE AFTER OPEN\t%d\n", ((ipcbuf_t *)(hdu_out->data_block))->state);
	}      
      //fprintf(stdout, "HERE\n");      
    }

  return EXIT_SUCCESS;
}
