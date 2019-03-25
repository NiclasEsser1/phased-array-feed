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

#include "baseband2filterbank.cuh"
#include "cudautil.cuh"
#include "log.h"

// Clean up unused kernels and parameters
// Clean up testers also

pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
extern int quit;

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
	   " -l  Monitor, Y_ip_port_ptype or N\n"
	   " -m  Commensal spectral, n_ip_port_ptype_chunk0_nchunk_naccumulate_cufft-nx, k_key_sod_ptype_chunk0_nchunk_naccumulate_cufft-nx or N\n"
	   );
}

int main(int argc, char *argv[])
{
  int arg;
  conf_t conf;
  char log_fname[MSTR_LEN] = {'\0'};
  char temp[MSTR_LEN] = {'\0'};
  
  struct timespec start, stop;
  double elapsed_time;
  
  /* Default argument */
  default_arguments(&conf);
  
  /* Initializeial part */  
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:hg:i:j:k:l:m:")) != -1)
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
	  if(optarg[0] == 'Y')
	    {
	      conf.monitor = 1;
	      sscanf(optarg, "%*[^_]_%[^_]_%d_%d", conf.ip_monitor, &conf.port_monitor, &conf.ptype_monitor);
	    }
	  break;
	  
	case 'm':
	  if(optarg[0] == 'n')
	    {
	      conf.spectral2network = 1;
	      sscanf(optarg, "%*[^_]_%[^_]_%d_%d_%d_%d_%d_%d", conf.ip_spectral, &conf.port_spectral, &conf.ptype_spectral, &conf.start_chunk, &conf.nchunk_in_spectral, &conf.nblk_accumulate, &conf.cufft_nx_spectral);
	    }
	  if(optarg[0] == 'k')
	    {
	      conf.spectral2disk = 1;
	      sscanf(optarg, "%*[^_]_%[^_]_%d_%d_%d_%d_%d_%d", temp, &conf.sod_spectral, &conf.ptype_spectral, &conf.start_chunk, &conf.nchunk_in_spectral, &conf.nblk_accumulate, &conf.cufft_nx_spectral);	      
	      if(sscanf(temp, "%x", &conf.key_out_spectral)!=1)		
	     	{
	     	  fprintf (stderr, "BASEBAND2FILTERBANK_ERROR: Can not get spectral ring buffer key, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	     	  exit(EXIT_FAILURE);
	     	}
	      
	      if(conf.ptype_spectral == 1)
		{
		  conf.ndim_spectral = 1;
		  conf.npol_spectral = 1;
		}
	      else if(conf.ptype_spectral == 2)
		{	      
		  conf.ndim_spectral = 1;
		  conf.npol_spectral = 2;
		}
	      else if(conf.ptype_spectral == 4)
		{	      
		  conf.ndim_spectral = 2;
		  conf.npol_spectral = 2;
		}
	      else
		{
		  fprintf(stderr, "BASEBAND2FILTERBANK_ERROR: ptype_spectral should be 1, 2 or 4, but it is %d, which happens at \"%s\", line [%d], has to abort\n",conf.ptype_spectral,  __FILE__, __LINE__);
		  exit(EXIT_FAILURE);
		}
	    }
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
  fprintf(stdout, "elapsed_time for baseband2filterbank initialization is %f\n", elapsed_time);
  fflush(stdout);

  //fprintf(stderr, "FORCE TO QUIT\n");
  //exit(EXIT_FAILURE);
  
  /* Play with data */  
  //baseband2filterbank(conf);
  threads(conf);
  
  /* Destroy */
  log_add(conf.log_file, "INFO", 1, log_mutex, "BEFORE destroy");  
  destroy_baseband2filterbank(conf);
  log_add(conf.log_file, "INFO", 1, log_mutex, "END destroy");
  
  /* Destory log interface */  
  log_add(conf.log_file, "INFO", 1, log_mutex, "BASEBAND2FILTERBANK END");  
  log_close(conf.log_file);
  fprintf(stdout, "HERE AFTER LOG CLOSE\n");
  fflush(stdout);
  
  if(quit == 2)
    exit(EXIT_FAILURE);
  else
    return EXIT_SUCCESS;
}