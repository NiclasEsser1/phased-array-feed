#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include "dada2filterbank.h"
multilog_t *runtime_log;

void usage()
{
  fprintf(stdout,
	  "dada2filterbank_main - Convert dada format data into filterbank format\n"
	  "\n"
	  "Usage: dada2filterbank_main [options]\n"
	  " -a input dada file name or buffer key \n"
	  " -b output filterbank file name \n"
	  " -c Directory to put runtime file \n"
	   );
}

int main(int argc, char **argv)
{
  FILE *fp_log = NULL;
  int i, arg;
  conf_t conf;
  char log_fname[MSTR_LEN];
  
  /* Get input parameters */
  while((arg=getopt(argc,argv,"a:hb:c:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  return EXIT_FAILURE;
	  
	case 'a':
	  sscanf(optarg, "%x", &conf.key);
	  break;

	case 'b':
	  sscanf(optarg, "%s", conf.f_fname);
	  break;
	  	  
	case 'c':
	  sscanf(optarg, "%s", conf.dir);
	  break;	  
	}
    }

  /* Setup log interface */
  sprintf(log_fname, "%s/dada2filterbank.log", conf.dir);
  fp_log = fopen(log_fname, "ab+");
  if(fp_log == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", log_fname);
      return EXIT_FAILURE;
    }
  runtime_log = multilog_open("dada2filterbank", 1);
  multilog_add(runtime_log, fp_log);
  multilog(runtime_log, LOG_INFO, "START DADA2FILTERBANK\n");

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
  multilog(runtime_log, LOG_INFO, "FINISH DADA2FILTERBANK\n\n");
  multilog_close(runtime_log);
  fclose(fp_log);

  return EXIT_SUCCESS;
}
