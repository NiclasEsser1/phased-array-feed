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

extern int quit;

void usage()
{
  fprintf(stdout,
	  "capture_main - capture PAF BMF raw data from NiC\n"
	  "\n"
	  "Usage: paf_capture [options]\n"
	  " -a Hexadecimal shared memory key for capture \n"
	  " -b Start point of packet\n"
	  " -c Alive IP adress and port, accept multiple values with -e value1 -e value2 ... the format of it is \"ip_port_nchunkexpected_nchunkactual_cpu\" \n"
	  " -d Dead IP adress and port, accept multiple values with -e value1 -e value2 ... the format of it is \"ip_port_nchunkexpected\" \n"
	  " -e The center frequency of captured data\n"
	  " -f Reference information for the current capture, get from BMF packet header, epoch_sec_idf\n"
	  " -g Which directory to put runtime file\n"
	  " -h Show help\n"
	  " -i The CPU for buf control thread\n"
	  " -j The setup for the capture control, the format of it is \"capturectrl_capturectrlcpu\"\n"
	  " -k Bind thread to CPU or not\n"
	  " -l The number of data frames in each buffer block of each frequency chunk\n"
	  " -m The number of data frames in each temp buffer of each frequency chunk\n"
	  " -n The name of header template for PSRDADA\n"
	  " -o The source information, which is required for the case without capture control, in the format \"name_ra_dec\" \n"
	  " -p Force to pad band edge \n"
	  " -q The index of  beam \n"
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
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:g:hi:j:k:l:m:n:o:p:q:")) != -1)
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
	  sscanf(optarg, "%[^_]_%d_%d_%d_%d", conf.ip_alive[conf.nport_alive], &conf.port_alive[conf.nport_alive], &conf.nchunk_alive_expect_on_port[conf.nport_alive], &conf.nchunk_alive_actual_on_port[conf.nport_alive], &conf.capture_cpu[conf.nport_alive]);
	  conf.nport_alive++;
	  break;
	  
	case 'd':
	  sscanf(optarg, "%[^_]_%d_%d", conf.ip_dead[conf.nport_dead], &conf.port_dead[conf.nport_dead], &conf.nchunk_dead_on_port[conf.nport_dead]);
	  conf.nport_dead++;
	  break;
	  
	case 'e':
	  sscanf(optarg, "%lf", &conf.center_freq);
	  break;

	case 'f':
	  sscanf(optarg, "%d_%"SCNd64"_%"SCNd64"", &conf.days_from_1970, &conf.seconds_from_epoch, &conf.df_in_period);
	  break;
	  
	case 'g':
	  sscanf(optarg, "%s", conf.dir);  // It should be different for different beams and the directory name should be setup by pipeline;
	  break;
	  
	case 'i':
	  sscanf(optarg, "%d", &conf.rbuf_ctrl_cpu);
	  break;
	  
	case 'j':
	  sscanf(optarg, "%d_%d", &conf.capture_ctrl, &conf.capture_ctrl_cpu);
	  break;
	  
	case 'k':
	  sscanf(optarg, "%d", &conf.cpu_bind);
	  break;
	  	  
	case 'l':
	  sscanf(optarg, "%"SCNu64"", &conf.ndf_per_chunk_rbuf);
	  break;
	  
	case 'm':
	  sscanf(optarg, "%"SCNu64"", &conf.ndf_per_chunk_tbuf);
	  break;
	  
	case 'n':
	  sscanf(optarg, "%s", conf.dada_header_template);
	  break;
	  
	case 'o':
	  {
	    sscanf(optarg, "%[^_]_%[^_]_%s", conf.source, conf.ra, conf.dec);
	    break;
	  }
	  
	case 'p':
	  sscanf(optarg, "%d", &conf.pad);
	  break;
	  	  
	case 'q':
	  sscanf(optarg, "%d", &conf.beam_index);
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
  
  /* Initialize capture */
  initialize_capture(&conf);
  
  /* Do the job */
  threads(&conf);
  
  /* Destory capture */
  destroy_capture(conf);
  
  /* Destory log interface */
  log_add(conf.log_file, "INFO", 1,  "The last quit is %d", quit);
  log_add(conf.log_file, "INFO", 1,  "CAPTURE END");
  fclose(conf.log_file);

  /* termine it */
  if(quit > 1)
    exit(EXIT_FAILURE);

  return EXIT_SUCCESS;
}
