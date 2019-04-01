#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>
#include <dirent.h>
#include <cuda_profiler_api.h>

#include "baseband2baseband.cuh"
#include "cudautil.cuh"
#include "log.h"

void usage ()
{
  fprintf (stdout,
	   "baseband2baseband_main - Convert BMF baseband data from 16bits to 8bits and remove the oversampling \n"
	   "\n"
	   "Usage: baseband2baseband_main [options]\n"
	   " -a  Hexacdecimal shared memory key for incoming ring buffer\n"
	   " -b  Hexacdecimal shared memory key for outcoming ring buffer\n"
	   " -c  The number of data frames (per frequency chunk) of each incoming ring buffer block\n"
	   " -d  The number of streams \n"
	   " -e  The number of data frames (per frequency chunk) of each stream\n"
	   " -f  The directory to put runtime file\n"	   
	   " -g  The number of input frequency chunks\n"
	   " -h  show help\n"
	   " -i  FFT length\n"
	   " -j  Start-of-data or not\n"
	   " -k  Network interface Y_ip_port_ptype or N\n"
	   );
}

int main(int argc, char *argv[])
{
  int arg;
  conf_t conf;
  char log_fname[MSTR_LEN] = {'\0'};
  struct timespec start, stop;
  double elapsed_time;
  
  /* Default arguments */
  default_arguments(&conf);
    
  /* Initial part */  
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:hg:i:j:k:")) != -1)
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
	  sscanf(optarg, "%d", &conf.nchunk);
	  break;
	  
	case 'i':
	  sscanf(optarg, "%d", &conf.cufft_nx);
	  break;
	  
	case 'j':
	  sscanf(optarg, "%d", &conf.sod);
	  break;
	  
	case 'k':
	  if(optarg[0] == 'Y')
	    {
	      conf.fits_flag = 1;
	      sscanf(optarg, "%*[^_]_%[^_]_%d_%d", conf.ip, &conf.port, &conf.pol_type);
	    }
	  break;
	}
    }

  /* Setup log interface */
  
  /* Setup log interface */
  DIR* dir = opendir(conf.dir); // First to check if the directory exists
  if(dir)
    closedir(dir);
  else
    {
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Failed to open %s with opendir or it does not exist, which happens at which happens at \"%s\", line [%d], has to abort\n", conf.dir, __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
  sprintf(log_fname, "%s/baseband2baseband.log", conf.dir);
  conf.log_file = log_open(log_fname, "ab+");
  if(conf.log_file == NULL)
    {
      fprintf(stderr, "BASEBAND2BASEBAND_ERROR: Can not open log file %s\n", log_fname);
      exit(EXIT_FAILURE);
    }

  /* Check  input */
  examine_record_arguments(conf, argv, argc);
  
  /* Initialize */
  clock_gettime(CLOCK_REALTIME, &start);
  initialize_baseband2baseband(&conf);
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "elapsed_time for baseband2baseband initialization is %f\n", elapsed_time);
  fflush(stdout);
  
  /* Play with the data */  
  fprintf(stdout, "BASEBAND2BASEBAND_READY\n");  // Ready to take data from ring buffer, just before the header thing
  fflush(stdout);
  log_add(conf.log_file, "INFO", 1,  "BASEBAND2BASEBAND_READY");
  baseband2baseband(conf);
  
  /* Destory */
  log_add(conf.log_file, "INFO", 1,  "BEFORE destroy");  
  destroy_baseband2baseband(conf);
  log_add(conf.log_file, "INFO", 1,  "END destroy");
  
  /* Destory log interface */  
  log_add(conf.log_file, "INFO", 1,  "BASEBAND2BASEBAND END");  
  log_close(conf.log_file);
  fprintf(stdout, "HERE AFTER LOG CLOSE\n");
  fflush(stdout);

  return EXIT_SUCCESS;
}