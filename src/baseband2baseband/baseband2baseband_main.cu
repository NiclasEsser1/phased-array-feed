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
	   "paf_process - Pre-process PAF BMF raw data or DADA format file for dspsr \n"
	   "\n"
	   "Usage: paf_process [options]\n"
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
	   " -k  The directory for data recording\n"
	   " -l  The source of data, stream or files\n");
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
	      //multilog (runtime_log, LOG_ERR, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      fprintf (stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;
	  
	case 'b':
#ifdef FOLD_MODE
	  if (sscanf (optarg, "%x", &conf.key_out_fold) != 1)
	    {
	      //multilog (runtime_log, LOG_ERR, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      fprintf (stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
#else
	  if (sscanf (optarg, "%x", &conf.key_out_search) != 1)
	    {
	      //multilog (runtime_log, LOG_ERR, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      fprintf (stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
#endif
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
	  
	case 'l':
	  sscanf(optarg, "%d", &conf.stream);
	  break;	  
	}
    }

  /* Setup log interface */
  sprintf(log_fname, "%s/paf_process.log", conf.dir);
  fp_log = fopen(log_fname, "ab+");
  if(fp_log == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", log_fname);
      return EXIT_FAILURE;
    }
  runtime_log = multilog_open("paf_process", 1);
  multilog_add(runtime_log, fp_log);
  multilog(runtime_log, LOG_INFO, "START PAF_PROCESS\n");

  /* Here to make sure that if we only expose one GPU into docker container, we can get the right index of it */ 
  int deviceCount;
  CudaSafeCall(cudaGetDeviceCount(&deviceCount));
  if(deviceCount == 1)
    conf.device_id = 0;

#ifdef DEBUG
  struct timespec start, stop;
  double elapsed_time;
  clock_gettime(CLOCK_REALTIME, &start);
#endif
  init_process(&conf);
#ifdef DEBUG
      clock_gettime(CLOCK_REALTIME, &stop);
      elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1000000000.0L;
      fprintf(stdout, "elapsed time for processing prepare is %f s\n\n\n\n\n", elapsed_time);
#endif
  
  /* Check on-board gpus */
//#ifdef DEBUG
//  int deviceCount, device;
//  struct cudaDeviceProp properties;
//  CudaSafeCall(cudaGetDeviceCount(&deviceCount));
//  fprintf(stdout, "Number of devices %d\n", deviceCount);
//  for(device = 0; device < deviceCount; ++device)
//    {
//      cudaGetDeviceProperties(&properties, device);
//      if (properties.major != 9999) /* 9999 means emulation only */
//  	{
//  	  printf("multiProcessorCount %d\n",properties.multiProcessorCount);
//  	  printf("maxThreadsPerMultiProcessor %d\n",properties.maxThreadsPerMultiProcessor);
//  	  printf("pciDeviceID %d\n",properties.pciDeviceID);
//  	  printf("pciBusID %d\n",properties.pciBusID);
//  	}
//    }
//#endif
  
  /* Play with data */
#ifdef DEBUG
  clock_gettime(CLOCK_REALTIME, &start);
#endif
  if(do_process(conf))
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

  destroy_process(conf);

  /* Destory log interface */
  multilog(runtime_log, LOG_INFO, "FINISH PAF_PROCESS\n\n");
  multilog_close(runtime_log);
  fclose(fp_log);
  
  return EXIT_SUCCESS;
}