#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>

#include "multilog.h"
#include "baseband2baseband.cuh"
#include "cudautil.cuh"


void usage ()
{
  fprintf (stdout,
	   "paf_baseband2baseband - Pre-process PAF BMF raw data or DADA format file for dspsr \n"
	   "\n"
	   "Usage: paf_baseband2baseband [options]\n"
	   " -a  Hexacdecimal shared memory key for incoming ring buffer\n"
	   " -b  Hexacdecimal shared memory key for outcoming ring buffer\n"
	   " -c  The number of data frame steps of each incoming ring buffer block\n"
	   " -d  How many times we need to repeat the process and finish one incoming block\n"
	   " -e  The number of streams \n"
	   " -f  The number of data stream steps of each stream\n"
	   " -g  Enable start-of-data or not\n"
	   " -h  show help\n"
	   " -i  The index of GPU\n"
	   " -j  The name of DADA header template\n"
	   " -k  The directory for data recording\n"	   );
}

multilog_t *runtime_log;

int main(int argc, char *argv[])
{
  int arg;
  conf_t conf;
  FILE *fp_log = NULL;
  char log_fname[MSTR_LEN];
  
  /* Initial part */  
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:g:hi:j:k:l:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  return EXIT_FAILURE;
	  
	case 'a':	  
	  if (sscanf (optarg, "%x", &conf.key_in) != 1)
	    {
	      fprintf (stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
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
	  sscanf(optarg, "%lf", &conf.rbufin_ndf);
	  break;
	  
	case 'd':
	  sscanf(optarg, "%d", &conf.nrun_blk);
	  break;
	  
	case 'e':
	  sscanf(optarg, "%d", &conf.nstream);
	  break;
	  
	case 'f':
	  sscanf(optarg, "%d", &conf.stream_ndf);
	  break;
	  	  
	case 'g':
	  sscanf(optarg, "%d", &conf.sod);
	  break;
	  
	case 'i':
	  sscanf(optarg, "%d", &conf.device_id);
	  break;

	case 'j':	  	  
	  sscanf(optarg, "%s", conf.hfname);
	  break;

	case 'k':
	  sscanf(optarg, "%s", conf.dir);
	  break;
	}
    }

  /* Setup log interface */
  sprintf(log_fname, "%s/paf_baseband2baseband.log", conf.dir);
  fp_log = fopen(log_fname, "ab+");
  if(fp_log == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", log_fname);
      return EXIT_FAILURE;
    }
  runtime_log = multilog_open("paf_baseband2baseband", 1);
  multilog_add(runtime_log, fp_log);
  multilog(runtime_log, LOG_INFO, "START PAF_BASEBAND2BASEBAND\n");

  /* Here to make sure that if we only expose one GPU into docker container, we can get the right index of it */ 
  int deviceCount;
  CudaSafeCall(cudaGetDeviceCount(&deviceCount));
  if(deviceCount == 1)
    conf.device_id = 0;

  init_baseband2baseband(&conf);

  if(do_baseband2baseband(conf))
    {
      multilog (runtime_log, LOG_ERR, "Can not finish the process, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not finish the process, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  destroy_baseband2baseband(conf);

  /* Destory log interface */
  multilog(runtime_log, LOG_INFO, "FINISH PAF_BASEBAND2BASEBAND\n\n");
  multilog_close(runtime_log);
  fclose(fp_log);
  
  return EXIT_SUCCESS;
}