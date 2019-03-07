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
#include <cuda_profiler_api.h>

#include "baseband2spectral.cuh"
#include "cudautil.cuh"
#include "log.h"

// Clean up unused kernels and parameters
// Clean up testers also

pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

void usage ()
{
  fprintf (stdout,
	   "baseband2spectral_main - Convert BMF 16bits baseband data into 32bits float spectral data \n"
	   "\n"
	   "Usage: baseband2spectral_main [options]\n"
	   " -a  Hexacdecimal shared memory key for incoming ring buffer\n"
	   " -b  Output information, \"k_key_sod, or n_ip_port\""
	   " -c  The number of data frame (per frequency chunk) of each incoming ring buffer block\n"
	   " -d  The number of streams \n"
	   " -e  The number of data frame (per frequency chunk) of each stream\n"
	   " -f  The directory to put runtime files\n"
	   " -g  The number of input frequency chunks\n"
	   " -h  show help\n"
	   " -i  FFT length\n"
	   " -j  Pol type, 1 for Stokes I, 2 for AABB and 4 for IQUV\n"
	   " -k  The number of buffer blocks to accumulate\n"
	   );
}

int main(int argc, char *argv[])
{
  int arg;
  conf_t conf;
  char log_fname[MSTR_LEN] = {'\0'};
  //char temp[MSTR_LEN] = {'\0'};
  struct timespec start, stop;
  double elapsed_time;
      
  /* Default argument */
  default_arguments(&conf);
  
  /* Initializeial part */  
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:hg:i:j:k:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);
	  
	case 'a':	  
	  if (sscanf (optarg, "%x", &conf.key_in) != 1)
	    {
	      fprintf (stderr, "BASEBAND2SPECTRAL_ERROR:\tCould not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'b':
	  if(optarg[0] == 'k')
	    {
	      conf.output_network = 0;
	      //if(sscanf(optarg, "%*[^_]_%[^_]_%d", temp, &conf.sod) != 2)
	      if(sscanf(optarg, "%*[^_]_%x_%d", &conf.key_out, &conf.sod) != 2)
		{
	     	  fprintf (stderr, "BASEBAND2SPECTRAL_ERROR:Can not get output ring buffer configuration, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	     	  exit(EXIT_FAILURE);
	     	}
	     //if(sscanf(temp, "%x", &conf.key_out)!=1)		
	     //	{
	     //	  fprintf (stderr, "BASEBAND2SPECTRAL_ERROR:Can not get output ring buffer key, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	     //	  exit(EXIT_FAILURE);
	     //	}
	    }
	  if(optarg[0] == 'n')
	    {
	      conf.output_network = 1;
	      if(sscanf(optarg, "%*[^_]_%[^_]_%d", conf.ip, &conf.port) != 2)
		{		  
	  	  fprintf (stderr, "BASEBAND2SPECTRAL_ERROR:Can not get output network configuration\t, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  	  exit(EXIT_FAILURE);
	  	}
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
	  sscanf(optarg, "%d", &conf.nchunk_in);
	  break;

	case 'i':
	  sscanf(optarg, "%d", &conf.cufft_nx);
	  break;
	  
	case 'j':
	  sscanf(optarg, "%d", &conf.pol_type);
	  if(conf.pol_type == 1)
	    {
	      conf.ndim_out = 1;
	      conf.npol_out = 1;
	    }
	  else if(conf.pol_type == 2)
	    {	      
	      conf.ndim_out = 1;
	      conf.npol_out = 2;
	    }
	  else if(conf.pol_type == 4)
	    {	      
	      conf.ndim_out = 2;
	      conf.npol_out = 2;
	    }
	  else
	    {
	      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: pol_type should be 1, 2 or 4, but it is %d, which happens at \"%s\", line [%d], has to abort\n",conf.pol_type,  __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  	  
	case 'k':
	  sscanf(optarg, "%d", &conf.nblk_accumulate);
	  break;
	}      
    }
  
  /* Setup log interface */
  DIR* dir = opendir(conf.dir); // First to check if the directory exists
  if(dir)
    closedir(dir);
  else
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Failed to open %s with opendir or it does not exist, which happens at which happens at \"%s\", line [%d], has to abort\n", conf.dir, __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
  sprintf(log_fname, "%s/baseband2spectral.log", conf.dir);
  conf.log_file = log_open(log_fname, "ab+");
  if(conf.log_file == NULL)
    {
      fprintf(stderr, "BASEBAND2SPECTRAL_ERROR: Can not open log file %s\n", log_fname);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "BASEBAND2SPECTRAL START");

  /* check the command line and record it */
  examine_record_arguments(conf, argv, argc);

  /* initialize */
  clock_gettime(CLOCK_REALTIME, &start);
  initialize_baseband2spectral(&conf);
  clock_gettime(CLOCK_REALTIME, &stop);
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "elapse_time for spectral for initialize is %f\n", elapsed_time);
  fflush(stdout);
  
  /* Play with data */  
  fprintf(stdout, "BASEBAND2SPECTRAL_READY\n");  // Ready to take data from ring buffer, just before the header thing
  fflush(stdout);
  log_add(conf.log_file, "INFO", 1, log_mutex, "BASEBAND2SPECTRAL_READY");
  baseband2spectral(conf);

  /* Destroy */
  log_add(conf.log_file, "INFO", 1, log_mutex, "BEFORE destroy");  
  destroy_baseband2spectral(conf);
  log_add(conf.log_file, "INFO", 1, log_mutex, "END destroy");
  
  /* Destory log interface */  
  log_add(conf.log_file, "INFO", 1, log_mutex, "BASEBAND2SPECTRAL END");  
  log_close(conf.log_file);

  CudaSafeCall(cudaProfilerStop());
  return EXIT_SUCCESS;
}