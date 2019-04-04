#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>

#include "dada_def.h"
#include "capture.h"
#include "log.h"

void usage()
{
  fprintf(stdout,
	  "capture_main - capture PAF BMF raw data from NiC\n"
	  "\n"
	  "Usage: paf_capture [options]\n"
	  " -a Hexadecimal shared memory key for capture \n"
	  " -b Start point of beamformer packet to record, to decide record packet header or not\n"
	  " -c IP adress and port, the format of it is \"ip_port_nchunkexpected_nchunkactual\" \n"
	  " -d The center frequency of captured data\n"
	  " -e Reference information for the current capture, get from BMF packet header, epoch_sec_idf\n"
	  " -f Which directory to put runtime file\n"
	  " -g The number of data frames in each buffer block of each frequency chunk\n"
	  " -h Show help\n"
	  " -i The number of data frames in each temp buffer of each frequency chunk\n"
	  " -j The name of header template for PSRDADA\n"
	  " -k The index of beam, use RECEIVER in DADA header to record it \n"
	  );
}

int main(int argc, char **argv)
{
  /* Initial part */
  int i, arg, source = 0;
  conf_t conf;
  char fname_log[MSTR_LEN] = {'\0'};

  /* default arguments*/
  default_arguments(&conf);

  /* read in argument from command line */
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:g:hi:j:k:l:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);
	  
	case 'a':	  	  
	  if(sscanf(optarg, "%x", &conf.key) != 1)
	    {
	      fprintf(stderr, "CAPTURE_ERROR: Could not parse key from %s, which happens at \"%s\", line [%d], has to abort.\n", optarg, __FILE__, __LINE__);
	      usage();
	      
	      exit(EXIT_FAILURE);
	    }
	  break;

	case 'b':	  	  
	  sscanf(optarg, "%d", &conf.dfsz_seek);
	  break;

	case 'c':
	  sscanf(optarg, "%[^_]_%d_%d_%d", conf.ip, &conf.port, &conf.nchunk_expect, &conf.nchunk_actual);
	  break;
	  
	case 'd':
	  sscanf(optarg, "%lf", &conf.dada_header.freq);
	  break;

	case 'e':
	  sscanf(optarg, "%d_%"SCNd64"_%"SCNd64"", &conf.days_from_1970, &conf.seconds_from_epoch, &conf.df_in_period);
	  break;
	  
	case 'f':
	  sscanf(optarg, "%s", conf.dir);  // It should be different for different beams and the directory name should be setup by pipeline;
	  break;
	  	  
	case 'g':
	  sscanf(optarg, "%"SCNu64"", &conf.ndf_per_chunk_rbuf);
	  break;
	  
	case 'i':
	  sscanf(optarg, "%"SCNu64"", &conf.ndf_per_chunk_tbuf);
	  break;
	  
	case 'j':
	  sscanf(optarg, "%s", conf.dada_header_template);
	  break;
	  
	case 'k':
	  sscanf(optarg, "%s", conf.dada_header.receiver);
	  break;
	}
    }
  
  /* Setup log interface */
  DIR* dir = opendir(conf.dir); // First to check if the directory exists
  if(dir)
    closedir(dir);
  else
    {
      fprintf(stderr, "CAPTURE_ERROR: Failed to open %s with opendir or it does not exist, which happens at which happens at \"%s\", line [%d], has to abort\n", conf.dir, __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
  sprintf(fname_log, "%s/capture.log", conf.dir);  // Open the log file
  conf.log_file = log_open(fname_log, "ab+");
  if(conf.log_file == NULL)
    {
      fprintf(stderr, "CAPTURE_ERROR: Can not open log file %s, which happends at \"%s\", line [%d], has to abort\n", fname_log, __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "CAPTURE START");
  
  /* check the command line and record it */
  examine_record_arguments(conf, argv, argc);

  /* Destory log interface */
  log_add(conf.log_file, "INFO", 1,  "CAPTURE END");
  fclose(conf.log_file);
  
  return EXIT_SUCCESS;
}
