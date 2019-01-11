#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>

#include "dada_cuda.h"

void usage()
{  
  fprintf (stdout,
	   "dbregister_main - register the data_block in the hdu via cudaHostRegister  \n"
	   "\n"
	   "Usage: dbregister_main [options]\n"
	   " -a  Hexacdecimal shared memory key for the hdu\n"
	   );
}

int main(int argc, char *argv[])
{
  int arg;
  int dbregister;
  key_t key;
  dada_hdu_t *hdu = NULL;
  
  while((arg=getopt(argc,argv,"a:hb:")) != -1)
    switch(arg)
      {
      case 'a':
	sscanf (optarg, "%x", &key);
	break;
	
      case 'b':
	sscanf (optarg, "%d", &dbregister);
	break;
	
      case 'h':
	usage();
	return EXIT_FAILURE;
      }

  hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(hdu, key);
  dada_hdu_connect(hdu);
  if (dbregister)
    dada_cuda_dbregister(hdu);
  else
    dada_cuda_dbunregister(hdu);
  dada_hdu_disconnect(hdu);
  dada_hdu_destroy(hdu);
  
  return EXIT_SUCCESS;
}
