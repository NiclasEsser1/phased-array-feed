#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include "multilog.h"
#include "dada_def.h"
#include "capture.h"

// ./capture_main -a dada -b /beegfs/DENG/docker/ -c 10.17.0.1:17100:8:8 -c 10.17.0.1:17101:8:7 -c 10.17.0.1:17102:8:7 -c 10.17.0.1:17103:8:7 -c 10.17.0.1:17104:8:7 -c 10.17.0.1:17105:8:7
// ./capture_main -a dada -b /beegfs/DENG/docker/ -c 10.17.0.1:17100:8:8 -c 10.17.0.1:17101:8:7 -c 10.17.0.1:17102:8:7 -c 10.17.0.1:17103:8:7 -d 10.17.0.1:17104:8 -d 10.17.0.1:17105:8 -e ../../config/header_16bit.txt

void usage()
{
  fprintf (stdout,
	   "paf_capture - capture PAF BMF raw data from NiC\n"
	   "\n"
	   "Usage: paf_capture [options]\n"
	   " -a Hexadecimal shared memory key for capture \n"
	   " -b Record header of data packets or not\n"
	   " -c Active IP adress and port, accept multiple values with -e value1 -e value2 ... the format of it is \"ip:port:nchunk:chunk\" \n"
	   " -d Dead IP adress and port, accept multiple values with -e value1 -e value2 ... the format of it is \"ip:port:nchunk:chunk\" \n"
	   " -e The name of DADA header file\n"
	   " -f The center frequency of captured data\n"
	   " -g Number of channels of current capture\n"
	   " -h Show help\n"
	   " -i Reference for the current capture, the format of it is \"df_sec:df_idf:utc_start:mjd_start:picoseconds\"\n"
	   " -j Which directory will be used to record data\n"
	   );
}

int main(int argc, char **argv)
{
  /* Initial part */
  int i, arg;
  conf_t conf;
  
  conf.nport_active = 0;
  conf.nport_dead   = 0;
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:g:hi:j:")) != -1)
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
	  sscanf(optarg, "%d", &conf.hdr);
	  break;

	case 'c':
	  sscanf(optarg, "%[^:]:%d:%d:%d", conf.ip_active[conf.nport_active], &conf.port_active[conf.nport_active], &conf.nchunk_active_expect[conf.nport_active], &conf.nchunk_active_actual[conf.nport_active]);
	  fprintf(stdout, "%d\t%d\n", conf.nchunk_active_expect[conf.nport_active], conf.nchunk_active_actual[conf.nport_active]);
	  conf.nport_active++;
	  break;
	  
	case 'd':
	  sscanf(optarg, "%[^:]:%d:%d", conf.ip_dead[conf.nport_dead], &conf.port_dead[conf.nport_dead], &conf.nchunk_dead[conf.nport_dead]);
	  fprintf(stdout, "%d\n", conf.nchunk_dead[conf.nport_dead]);
	  conf.nport_dead++;
	  break;
	  
	case 'e':	  	  
	  sscanf(optarg, "%s", conf.hdr_fname);
	  fprintf(stdout, "%s\n", conf.hdr_fname);
	  break;
	}
    }
  
  return EXIT_SUCCESS;
}
