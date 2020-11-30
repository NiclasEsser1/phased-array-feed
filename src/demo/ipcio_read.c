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
	  " -b Bytes to read\n"
	  );
}

// gcc -o ipcio_read ipcio_read.c -lpsrdada -lcudart -lcuda -L/usr/local/cuda/lib64
// ./ipcio_read -a dada -b 10000

int main(int argc, char **argv)
{
  int i, arg;
  dada_hdu_t *hdu = NULL;
  key_t key;
  uint64_t byte;
  ipcio_t *ringbuf = NULL;
  char *buf = NULL;
  multilog_t *runtime_log;
  
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
	  
	case 'b':
	  sscanf(optarg, "%"SCNu64"", &byte);
	  break;	  
	}
    }
  
  /* Create buffer */
  buf = (char *)malloc(byte * sizeof(char));
  
  /* Create HDU and open data block to read */
  hdu = dada_hdu_create(runtime_log);
  dada_hdu_set_key(hdu, key);
  dada_hdu_connect(hdu);
  ringbuf = hdu->data_block;
  ipcio_open(ringbuf, 'R');

  while(1)
    {
      ipcio_read(ringbuf, buf, byte);
      fprintf(stdout, "HERE\n");
    }
  /* Close data block and destroy HDU */
  ipcio_close(ringbuf);
  dada_hdu_disconnect(hdu);
  dada_hdu_destroy(hdu);

  /* Release buffer */
  free(buf);
  
  return EXIT_SUCCESS;
}
