#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>

#include "baseband2filterbank.cuh"
#include "cudautil.cuh"
#include "log.h"

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
  int i, arg;
  conf_t conf;
  char log_fname[MSTR_LEN] = {'\0'};
  char command_line[MSTR_LEN] = {'\0'};

  conf.sod = 0; // We do not enable sod by default
  /* Initializeial part */  
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:hg:i:j:k:l:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  fprintf (stderr, "BASEBAND2FILTERBANK_ERROR:\tno input, which happens at \"%s\", line [%d].\n",  __FILE__, __LINE__);
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
	  sscanf(optarg, "%"SCNu64"", &conf.ndf_chunk_rbufin);
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
  sprintf(log_fname, "%s/baseband2filterbank.log", conf.dir);
  conf.log_file = log_open(log_fname, "ab+");
  if(conf.log_file == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", log_fname);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "BASEBAND2FILTERBANK START");

  /* Log the input */
  strcpy(command_line, argv[0]);
  for(i = 1; i < argc; i++)
    {
      strcat(command_line, " ");
      strcat(command_line, argv[i]);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "The command line is \"%s\"", command_line);
  log_add(conf.log_file, "INFO", 1, log_mutex, "The input ring buffer key is %x", conf.key_in); 
  log_add(conf.log_file, "INFO", 1, log_mutex, "The output ring buffer key is %x", conf.key_out);
  log_add(conf.log_file, "INFO", 1, log_mutex, "Each input ring buffer block has %"PRIu64" packets per frequency chunk", conf.ndf_chunk_rbufin);
  log_add(conf.log_file, "INFO", 1, log_mutex, "%d streams run on GPU", conf.nstream);
  log_add(conf.log_file, "INFO", 1, log_mutex, "Each stream process %d packets per frequency chunk", conf.ndf_per_chunk_stream);
  log_add(conf.log_file, "INFO", 1, log_mutex, "The runtime information is %s", conf.dir);
  if(conf.sod)
    log_add(conf.log_file, "INFO", 1, log_mutex, "The filterbank data is enabled at the beginning");
  else
    log_add(conf.log_file, "INFO", 1, log_mutex, "The filterbank data is NOT enabled at the beginning");
  log_add(conf.log_file, "INFO", 1, log_mutex, "%d chunks of input data", conf.nchunk_in);
  log_add(conf.log_file, "INFO", 1, log_mutex, "We use %d points FFT", conf.cufft_nx);
  log_add(conf.log_file, "INFO", 1, log_mutex, "We output %d channels", conf.nchan_out);
  log_add(conf.log_file, "INFO", 1, log_mutex, "We keep %d fine channels for the whole band after FFT", conf.nchan_keep_band);
  
  /* initialize */
  initialize_baseband2filterbank(&conf);

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