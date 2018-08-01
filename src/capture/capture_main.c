#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include "multilog.h"
#include "dada_def.h"
#include "capture.h"

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
	   " -e The name of DADA header template file\n"
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
	  sscanf(optarg, "%[^:]:%d:%d", conf.ip_active[conf.nport_active], &conf.port_active[conf.nport_active], &conf.nchunk_active_expect[conf.nport_active]);
	  fprintf(stdout, "%d\n", conf.nchunk_active_expect[conf.nport_active]);
	  conf.nport_active++;
	  break;

	  
	}
    }
  
  return EXIT_SUCCESS;
}
