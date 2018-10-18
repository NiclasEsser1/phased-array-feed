#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "multilog.h"
#include "power2udp.h"

void usage ()
{
  fprintf (stdout,
	   "paf_power2udp - To packet the power data with other parameters to UDP and send it via given ip_udp:port_udp, it also forward TOS metadata from ip_meta:port_meta\n"
	   "\n"
	   "Usage: paf_power2udp [options]\n"
	   " -h  show help\n"
	   " -a  Hexacdecimal shared memory key for incoming ring buffer\n"
	   " -b  The name of the directory in which we will record the log\n"
	   " -c  IP address to send data \n"
	   " -d  Port number to send data \n"
	   " -e  IP address to receive metadata \n"
	   " -f  Port number to receive metadata \n"
	   " -g  Number of runs to finish one buffer \n"
	   );
}

multilog_t *runtime_log;

int main(int argc, char *argv[])
{
  int arg;
  FILE *fp_log = NULL;
  char log_fname[MSTR_LEN];
  conf_t conf;
  
  while((arg=getopt(argc,argv,"a:b:hc:d:e:f:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  return EXIT_FAILURE;

	case 'a':	  	  	  
	  if (sscanf (optarg, "%x", &conf.key) != 1)
	    {
	      fprintf (stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;
	  
	case 'b':	  	  	  
	  sscanf (optarg, "%s", conf.dir);
	  break;
	  
	case 'c':	  	  	  
	  sscanf (optarg, "%s", conf.ip_udp);
	  break;

	case 'd':	  	  	  
	  sscanf (optarg, "%d", &conf.port_udp);
	  break;
	  
	case 'e':	  	  	  
	  sscanf (optarg, "%s", conf.ip_meta);
	  break;

	case 'f':	  	  	  
	  sscanf (optarg, "%d", &conf.port_meta);
	  break;
	  
	case 'g':	  	  	  
	  sscanf (optarg, "%d", &conf.nrun);
	  break;
	}
    }

  /* Setup log interface */
  sprintf(log_fname, "%s/power2udp_main.log", conf.dir);
  fp_log = fopen(log_fname, "ab+"); // File to record log information
  if(fp_log == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", log_fname);
      return EXIT_FAILURE;
    }
  runtime_log = multilog_open("power2udp_main", 1);
  multilog_add(runtime_log, fp_log);
  multilog(runtime_log, LOG_INFO, "START POWER2UDP_MAIN\n");
  
  /* init */
  init_power2udp(&conf);

  /* do it */
  power2udp(conf);

  /* delete it */
  destroy_power2udp(conf);
  
  /* Destory log interface */
  multilog(runtime_log, LOG_INFO, "FINISH POWER2UDP_MAIN\n\n");
  multilog_close(runtime_log);
  fclose(fp_log);

  return EXIT_SUCCESS;
}
