#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>

#include "multilog.h"
#include "baseband2spectral.cuh"
#include "cudautil.cuh"


void usage ()
{
  fprintf (stdout,
	   "baseband2spectral_main - Convert BMF 16bits baseband data into 8bits spectral data \n"
	   "\n"
	   "Usage: baseband2spectral_main [options]\n"
	   " -a  Hexacdecimal shared memory key for incoming ring buffer\n"
	   " -b  Hexacdecimal shared memory key for outcoming ring buffer\n"
	   " -c  The number of data frame (per frequency chunk) of each incoming ring buffer block\n"
	   " -d  How many times we need to repeat the process and finish one incoming block\n"
	   " -e  The number of streams \n"
	   " -f  The number of data frame (per frequency chunk) of each stream\n"
	   " -g  Do we need to run sum_kernel twice\n"
	   " -h  show help\n"	   
	   " -i  The block size of the first sum_kernel\n"
	   " -j  The directory to put log file\n");
}

multilog_t *runtime_log;

int main(int argc, char *argv[])
{
  int arg;
  conf_t conf;
  FILE *fp_log = NULL;
  char log_fname[MSTR_LEN];
  
  /* Initial part */  
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:hg:i:j:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  return EXIT_FAILURE;
	  
	case 'a':	  
	  if (sscanf (optarg, "%x", &conf.key_in) != 1)
	    {
	      //multilog (runtime_log, LOG_ERR, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      fprintf (stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;
	  
	case 'b':
	  if (sscanf (optarg, "%x", &conf.key_out) != 1)
	    {
	      //multilog (runtime_log, LOG_ERR, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      fprintf (stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;
	  	  
	case 'c':
	  sscanf(optarg, "%lf", &conf.rbufin_ndf_chk);
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
  sprintf(log_fname, "%s/baseband2spectral.log", conf.dir);
  fp_log = fopen(log_fname, "ab+");
  if(fp_log == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", log_fname);
      return EXIT_FAILURE;
    }
  runtime_log = multilog_open("baseband2spectral", 1);
  multilog_add(runtime_log, fp_log);
  multilog(runtime_log, LOG_INFO, "START BASEBAND2SPECTRAL\n");

#ifdef DEBUG
  struct timespec start, stop;
  double elapsed_time;
  clock_gettime(CLOCK_REALTIME, &start);
#endif
  if(init_baseband2spectral(&conf))
    return EXIT_FAILURE;
#ifdef DEBUG
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000000.0L;
  fprintf(stdout, "elapsed time for processing prepare is %f s\n\n\n\n\n", elapsed_time);
#endif
  
  /* Play with data */
#ifdef DEBUG
  clock_gettime(CLOCK_REALTIME, &start);
#endif
  if(baseband2spectral(conf))
    {
      multilog (runtime_log, LOG_ERR, "Can not finish the process, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not finish the process, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  #ifdef DEBUG
      clock_gettime(CLOCK_REALTIME, &stop);
      elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000000.0L;
      fprintf(stdout, "elapsed time for data processing is %f s\n", elapsed_time);
#endif

  destroy_baseband2spectral(conf);

  /* Destory log interface */
  multilog(runtime_log, LOG_INFO, "FINISH BASEBAND2SPECTRAL\n\n");
  multilog_close(runtime_log);
  fclose(fp_log);
  
  return EXIT_SUCCESS;
}