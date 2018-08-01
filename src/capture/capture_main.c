#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include "multilog.h"
#include "dada_def.h"
#include "capture.h"

// ./capture_main -a dada -b /beegfs/DENG/docker/ -c 10.17.0.1_17100_8_8 -c 10.17.0.1_17101_8_7 -c 10.17.0.1_17102_8_7 -c 10.17.0.1_17103_8_7 -c 10.17.0.1_17104_8_7 -c 10.17.0.1_17105_8_7  -e ../../config/header_16bit.txt -f 1340.5 -g 336 -i 2749527_16035_2018-08-01-19:15:46_58331.802615740744_731780000000 -j /beegfs/DENG/docker
// ./capture_main -a dada -b /beegfs/DENG/docker/ -c 10.17.0.1_17100_8_8 -c 10.17.0.1_17101_8_7 -c 10.17.0.1_17102_8_7 -c 10.17.0.1_17103_8_7 -d 10.17.0.1_17104_8 -d 10.17.0.1_17105_8 -e ../../config/header_16bit.txt -f 1340.5 -g 336 -i 2749527_16035_2018-08-01-19:15:46_58331.802615740744_731780000000 -j /beegfs/DENG/docker
  
void usage()
{
  fprintf (stdout,
	   "paf_capture - capture PAF BMF raw data from NiC\n"
	   "\n"
	   "Usage: paf_capture [options]\n"
	   " -a Hexadecimal shared memory key for capture \n"
	   " -b Record header of data packets or not\n"
	   " -c Active IP adress and port, accept multiple values with -e value1 -e value2 ... the format of it is \"ip_port_nchunk_nchunk\" \n"
	   " -d Dead IP adress and port, accept multiple values with -e value1 -e value2 ... the format of it is \"ip_port_nchunk\" \n"
	   " -e The name of DADA header file\n"
	   " -f The center frequency of captured data\n"
	   " -g Number of channels of current capture\n"
	   " -h Show help\n"
	   " -i Reference for the current capture, the format of it is \"dfsec_dfidf_utcstart_mjdstart_picoseconds\"\n"
	   " -j Which directory to put log file\n"
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
	  sscanf(optarg, "%[^_]_%d_%d_%d", conf.ip_active[conf.nport_active], &conf.port_active[conf.nport_active], &conf.nchunk_active_expect[conf.nport_active], &conf.nchunk_active_actual[conf.nport_active]);
	  //fprintf(stdout, "%d\t%d\n", conf.nchunk_active_expect[conf.nport_active], conf.nchunk_active_actual[conf.nport_active]);
	  conf.nport_active++;
	  break;
	  
	case 'd':
	  sscanf(optarg, "%[^_]_%d_%d", conf.ip_dead[conf.nport_dead], &conf.port_dead[conf.nport_dead], &conf.nchunk_dead[conf.nport_dead]);
	  //fprintf(stdout, "%d\n", conf.nchunk_dead[conf.nport_dead]);
	  conf.nport_dead++;
	  break;
	  
	case 'e':	  	  
	  sscanf(optarg, "%s", conf.hdr_fname);
	  //fprintf(stdout, "%s\n", conf.hdr_fname);
	  break;

	case 'f':
	  sscanf(optarg, "%lf", &conf.center_freq);
	  break;

	case 'g':
	  sscanf(optarg, "%d", &conf.nchan);
	  break;

	case 'i':
	  //sscanf(optarg, "%"SCNu64"::%"SCNu64"::%s::%lf::%"SCNu64"", &conf.df_sec, &conf.df_idf, conf.utc_start, &conf.mjd_start, &conf.picoseconds);
	  sscanf(optarg, "%"SCNu64"_%"SCNu64"_%[^_]_%lf_%"SCNu64"", &conf.df_sec, &conf.df_idf, conf.utc_start, &conf.mjd_start, &conf.picoseconds);
	  break;
	  
	case 'j':
	  sscanf(optarg, "%s", conf.dir);
	  break;
	}
    }
  //fprintf(stdout, "%"PRIu64"\t%"PRIu64"\t%s\t%f\t%"PRIu64"\n", conf.df_sec, conf.df_idf, conf.utc_start, conf.mjd_start, conf.picoseconds);
  return EXIT_SUCCESS;
}
