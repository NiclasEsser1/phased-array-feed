#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include "log.h"
#include "dada2filterbank.h"

void usage()
{
  fprintf(stdout,
	  "dada2filterbank_main - Convert dada format data into filterbank format\n"
	  "\n"
	  "Usage: dada2filterbank_main [options]\n"
	  " -a input dada file name or buffer key, k_key or f_fname \n"
	  " -b output filterbank file name \n"
	  " -c Directory to put runtime file \n"
	   );
}

int main(int argc, char **argv)
{
  int i, arg;
  conf_t conf;
  char log_fname[MSTR_LEN] = {'\0'};
  
  /* Get input parameters */
  while((arg=getopt(argc,argv,"a:hb:c:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  return EXIT_FAILURE;
	  
	case 'a':
	  if(optarg[0] == 'k')
	    {
	      conf.file = 0;
	      sscanf(&optarg[2], "%x", &conf.key);
	    }
	  if(optarg[0] == 'f')
	    {
	      conf.file = 1;
	      sscanf(&optarg[2], "%s", conf.d_fname);
	    }
	  break;

	case 'b':
	  sscanf(optarg, "%s", conf.f_fname);
	  break;
	  	  
	case 'c':
	  sscanf(optarg, "%s", conf.dir);
	  break;	  
	}
    }
  fprintf(stdout, "%s\n", conf.d_fname);
  
  /* Setup log interface */
  sprintf(log_fname, "%s/dada2filterbank.log", conf.dir);
  conf.log_file = log_open(log_fname, "ab+");
  if(conf.log_file == NULL)
    {
      fprintf(stderr, "DADA2FILTERBANK_ERROR: Can not open log file %s\n", log_fname);
      return EXIT_FAILURE;
    }
  log_add(conf.log_file, "INFO", 1,  "DADA2FILTERBANK START");
  
  /* init the thing */
  initialization(&conf);
  
  /* Read dada header */
  dada_header(&conf);
  
  /* Write filterbank header */
  filterbank_header(conf);

  /* Write filterbank data */
  filterbank_data(conf);

  /* Destory the thing */
  destroy(conf);
  
  /* Destory log interface */
  log_add(conf.log_file, "INFO", 1,  "FINISH DADA2FILTERBANK");
  fclose(conf.log_file);

  return EXIT_SUCCESS;
}
