#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"


void usage()
{
  fprintf(stdout,
	  "ipcio_read - To test the buffer read with ipcio\n"
	  "\n"
	  "Usage: ipcio_read [options]\n"
	  " -a Hexadecimal shared memory key for capture \n"
	  );
}

// gcc -o ipcbuf_read ipcbuf_read.c -lpsrdada -lcudart -lcuda -L/usr/local/cuda/lib64
// ./ipcbuf_read -a dada

int main(int argc, char **argv)
{
  int i, arg;
  dada_hdu_t *hdu = NULL;
  key_t key;
  uint64_t byte, block_id;
  multilog_t *runtime_log;
  char *cbuf = NULL;
  
  /* Initial */
  while((arg=getopt(argc,argv,"a:hb:")) != -1)
    {
      switch(arg)
	{	  
	case 'h':
	  usage();
	  return EXIT_FAILURE;
	  
	case 'a':	  	  
	  if(sscanf(optarg, "%x", &key) != 1)
	    {
	      fprintf(stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;
	}
    }
  
  /* Create HDU and open data block to read */
  hdu = dada_hdu_create(runtime_log);
  dada_hdu_set_key(hdu, key);
  dada_hdu_connect(hdu);
  dada_hdu_lock_read(hdu);

  while(1)
    {
      cbuf = ipcio_open_block_read(hdu->data_block, &byte, &block_id);
      ipcio_close_block_read(hdu->data_block, byte);
    }
  dada_hdu_unlock_read(hdu);
  dada_hdu_disconnect(hdu);
  dada_hdu_destroy(hdu);

  return EXIT_SUCCESS;
}
