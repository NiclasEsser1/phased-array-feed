#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>
#include <dirent.h>
#include <errno.h>

#include "baseband2filterbank.cuh"
#include "cudautil.cuh"
#include "log.h"

// Clean up unused kernels and parameters
// Clean up testers also

pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

void usage ()
{
  fprintf (stdout,
	   "baseband2filterbank_main - Convert BMF 16bits baseband data into 8bits filterbank data \n"
	   "\n"
	   "Usage: baseband2filterbank_main [options]\n"
	   " -a  Hexacdecimal shared memory key for incoming ring buffer\n"
	   " -b  Hexacdecimal shared memory key for outcoming ring buffer\n"
	   " -c  The number of data frame (per frequency chunk) of each incoming ring buffer block\n"
	   " -d  The number of streams \n"
	   " -e  The number of data frame (per frequency chunk) of each stream\n"
	   " -f  The directory to put runtime files\n"
	   " -g  Start of the data or not\n"
	   " -h  show help\n"
	   " -i  Number of chunks of input\n"
	   " -j  FFT length\n"
	   " -k  The number of output channels\n"
	   " -l  Number of channels keep for the band\n");
}

int main(int argc, char *argv[])
{
  int arg;
  conf_t conf;
  char log_fname[MSTR_LEN] = {'\0'};

  struct timespec start, stop;
  double elapsed_time;
  
  /* Default argument */
  default_arguments(&conf);
  
  /* Initializeial part */  
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:hg:i:j:k:l:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);
	  
	case 'a':	  
	  if (sscanf (optarg, "%x", &conf.key_in) != 1)
	    {
	      fprintf (stderr, "BASEBAND2FILTERBANK_ERROR:\tCould not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'b':
	  if (sscanf (optarg, "%x", &conf.key_out) != 1)
	    {
	      fprintf (stderr, "BASEBAND2FILTERBANK_ERROR:\tCould not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  	  
	case 'c':
	  sscanf(optarg, "%"SCNu64"", &conf.ndf_per_chunk_rbufin);
	  break;
	  
	case 'd':
	  sscanf(optarg, "%d", &conf.nstream);
	  break;
	  
	case 'e':
	  sscanf(optarg, "%d", &conf.ndf_per_chunk_stream);
	  break;
	  	  
	case 'f':
	  sscanf(optarg, "%s", conf.dir);
	  break;
	  
	case 'g':
	  sscanf(optarg, "%d", &conf.sod);
	  break;
	  
	case 'i':
	  sscanf(optarg, "%d", &conf.nchunk_in);
	  break;

	case 'j':
	  sscanf(optarg, "%d", &conf.cufft_nx);
	  break;

	case 'k':
	  sscanf(optarg, "%d", &conf.nchan_out);
	  break;
	  
	case 'l':
	  sscanf(optarg, "%d", &conf.nchan_keep_band);
	  break;
	}
    }

  /* Setup log interface */
  DIR* dir = opendir(conf.dir); // First to check if the directory exists
  if(dir)
    closedir(dir);
  else
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Failed to open %s with opendir or it does not exist, which happens at which happens at \"%s\", line [%d], has to abort\n", conf.dir, __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
  sprintf(log_fname, "%s/baseband2filterbank.log", conf.dir);
  conf.log_file = log_open(log_fname, "ab+");
  if(conf.log_file == NULL)
    {
      fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: Can not open log file %s\n", log_fname);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "BASEBAND2FILTERBANK START");

  /* check the command line and record it */
  examine_record_arguments(conf, argv, argc);
  
  /* initialize */
  clock_gettime(CLOCK_REALTIME, &start);
  initialize_baseband2filterbank(&conf);
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "elapse_time for filterbank for initialize is %f\n", elapsed_time);
  fflush(stdout);

  //fprintf(stderr, "FORCE TO QUIT\n");
  //exit(EXIT_FAILURE);
  
  /* Play with data */
  baseband2filterbank(conf);

  /* Destroy */
  log_add(conf.log_file, "INFO", 1, log_mutex, "BEFORE destroy");  
  destroy_baseband2filterbank(conf);
  log_add(conf.log_file, "INFO", 1, log_mutex, "END destroy");
  
  /* Destory log interface */  
  log_add(conf.log_file, "INFO", 1, log_mutex, "BASEBAND2FILTERBANK END");  
  log_close(conf.log_file);
  
  return EXIT_SUCCESS;
}