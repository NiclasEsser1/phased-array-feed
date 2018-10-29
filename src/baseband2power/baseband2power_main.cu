#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>

#include "multilog.h"
#include "baseband2power.cuh"
#include "cudautil.cuh"
#include "kernel.cuh"

void usage ()
{
  fprintf (stdout,
	   "baseband2power_main - To detect baseband data with original channels and average the detected data in given time\n"
	   "\n"
	   "Usage: paf_baseband2power [options]\n"
	   " -a  Hexacdecimal shared memory key for incoming ring buffer\n"
	   " -b  Hexacdecimal shared memory key for outcoming ring buffer\n"
	   " -c  The number of data frames (per frequency chunk) of input ring buffer\n"	   
	   " -d  How many times we need to repeat the process to finish one incoming block\n"
	   " -e  The number of streams \n"
	   " -f  The number of data frames (per frequency chunk) of each stream\n"
	   " -g  Do we need to run sum_kernel twice\n"
	   " -h  show help\n"	   
	   " -i  The block size of the first sum_kernel\n"
	   " -j  The directory to put log file\n");
}

multilog_t *runtime_log;

int main(int argc, char *argv[])
{
  int arg;
  FILE *fp_log = NULL;
  char log_fname[MSTR_LEN];
  conf_t conf;
  
  /* configuration from command line */
  while((arg=getopt(argc,argv,"a:b:c:d:he:f:g:i:j:")) != -1)
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
	  sscanf(optarg, "%"SCNu64"", &conf.rbufin_ndf_chk);
	  break;
	  
	case 'd':
	  sscanf(optarg, "%d", &conf.nrun_blk);
	  break;
	  
	case 'e':
	  sscanf(optarg, "%d", &conf.nstream);
	  break;
	  
	case 'f':
	  sscanf(optarg, "%d", &conf.stream_ndf_chk);
	  break;
	  	  
	case 'g':
	  sscanf(optarg, "%d", &conf.twice_sum);
	  break;
	  
	case 'i':
	  sscanf(optarg, "%d", &conf.sum1_blksz);
	  break;
	  
	case 'j':
	  sscanf(optarg, "%s", conf.dir);
	  break;
	}
    }
  
  /* Setup log interface */
  sprintf(log_fname, "%s/paf_baseband2power.log", conf.dir);
  fp_log = fopen(log_fname, "ab+");
  if(fp_log == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", log_fname);
      return EXIT_FAILURE;
    }
  runtime_log = multilog_open("baseband2power_main", 1);
  multilog_add(runtime_log, fp_log);
  multilog(runtime_log, LOG_INFO, "START BASEBAND2POWER_MAIN\n");

  /* Init process */
  init_baseband2power(&conf);
  
  /* Do process */
  baseband2power(conf);

  destroy_baseband2power(conf);
  
  multilog(runtime_log, LOG_INFO, "FINISH BASEBAND2POWER\n\n");
  multilog_close(runtime_log);
  fclose(fp_log);
  
  return EXIT_SUCCESS;
}
