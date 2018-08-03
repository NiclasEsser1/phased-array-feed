#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include "multilog.h"
#include "dada_def.h"
#include "capture.h"
#include "sync.h"

multilog_t *runtime_log;

void usage()
{
  fprintf (stdout,
	   "paf_capture - capture PAF BMF raw data from NiC\n"
	   "\n"
	   "Usage: paf_capture [options]\n"
	   " -a Hexadecimal shared memory key for capture \n"
	   " -b BMF packet size\n"
	   " -c Start point of packet\n"
	   " -d Active IP adress and port, accept multiple values with -e value1 -e value2 ... the format of it is \"ip_port_nchunk_nchunk_cpu\" \n"
	   " -e Dead IP adress and port, accept multiple values with -e value1 -e value2 ... the format of it is \"ip_port_nchunk\" \n"
	   " -f The center frequency of captured data\n"
	   " -g Number of channels of current capture\n"
	   " -h Show help\n"
	   " -i Reference for the current capture, the format of it is \"dfsec_dfidf_utcstart_mjdstart_picoseconds\"\n"
	   " -j Which directory to put log file\n"
	   " -k The CPU for sync thread\n"
	   " -l The CPU for monitor thread\n"
	   " -m Bind thread to CPU or not\n"
	   " -n Time out for sockets\n"
	   " -o The number of chunks\n"
	   " -p The number of data frames in each buffer block of each frequency chunk\n"
	   " -q The number of data frames in each temp buffer of each frequency chunk\n"
	   " -r The number of data frames in period or each frequency chunk\n"
	   );
}

int main(int argc, char **argv)
{
  /* Initial part */
  int i, arg;
  conf_t conf;
  
  conf.nport_active = 0;
  conf.nport_dead   = 0;
  conf.sync_cpu     = 0;
  conf.monitor_cpu  = 0;
  conf.thread_bind  = 0; // Default do not bind thread to cpu
  for (i = 0; i < MPORT_CAPTURE; i++)
    conf.port_cpu[i] = 0;
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:g:hi:j:k:l:m:n:o:p:q:r:")) != -1)
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
	  sscanf(optarg, "%d", &conf.pktsz);
	  break;

	case 'c':	  	  
	  sscanf(optarg, "%d", &conf.pktoff);
	  break;

	case 'd':
	  sscanf(optarg, "%[^_]_%d_%d_%d_%d", conf.ip_active[conf.nport_active], &conf.port_active[conf.nport_active], &conf.nchunk_active_expect[conf.nport_active], &conf.nchunk_active_actual[conf.nport_active], &conf.port_cpu[conf.nport_active]);
	  fprintf(stdout, "%d\t%d\t%d\n", conf.nchunk_active_expect[conf.nport_active], conf.nchunk_active_actual[conf.nport_active], conf.port_cpu[conf.nport_active]);
	  conf.nport_active++;
	  break;
	  
	case 'e':
	  sscanf(optarg, "%[^_]_%d_%d", conf.ip_dead[conf.nport_dead], &conf.port_dead[conf.nport_dead], &conf.nchunk_dead[conf.nport_dead]);
	  conf.nport_dead++;
	  break;
	  
	case 'f':
	  sscanf(optarg, "%lf", &conf.center_freq);
	  break;

	case 'g':
	  sscanf(optarg, "%d", &conf.nchan);
	  break;

	case 'i':
	  sscanf(optarg, "%"SCNu64"_%"SCNu64"", &conf.sec_start, &conf.idf_start);
	  break;
	  
	case 'j':
	  sscanf(optarg, "%s", conf.dir);
	  break;
	  
	case 'k':
	  sscanf(optarg, "%d", &conf.sync_cpu);
	  break;
	  
	case 'l':
	  sscanf(optarg, "%d", &conf.monitor_cpu);
	  break;
	  
	case 'm':
	  sscanf(optarg, "%d", &conf.thread_bind);
	  break;
	  
	case 'n':
	  sscanf(optarg, "%d", &conf.sec_prd);
	  break;
	  
	case 'o':
	  sscanf(optarg, "%d", &conf.nchunk);
	  break;
	  
	case 'p':
	  sscanf(optarg, "%"SCNu64"", &conf.rbuf_ndf_chk);
	  break;
	  
	case 'q':
	  sscanf(optarg, "%"SCNu64"", &conf.tbuf_ndf_chk);
	  break;
	  
	case 'r':
	  sscanf(optarg, "%"SCNu64"", &conf.ndf_chk_prd);
	  break;
	}
    }
  
  /* Setup log interface */
  char fname_log[MSTR_LEN];
  FILE *fp_log = NULL;
  sprintf(fname_log, "%s/capture.log", conf.dir);
  fp_log = fopen(fname_log, "ab+"); 
  if(fp_log == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", fname_log);
      return EXIT_FAILURE;
    }
  runtime_log = multilog_open("capture", 1);
  multilog_add(runtime_log, fp_log);
  multilog(runtime_log, LOG_INFO, "CAPTURE START\n");
    
  /* To make sure that we are not going to bind all threads to one sigle CPU */
  if(!(conf.thread_bind == 0))
    {
      for (i = 0; i < MPORT_CAPTURE; i++)
	{
	  if (((conf.port_cpu[i] == conf.monitor_cpu)?0:1) == 1)
	    break;
	  if (((conf.port_cpu[i] == conf.sync_cpu)?0:1) == 1)
	    break;
	}
      if (i == MPORT_CAPTURE)
	{
	  fprintf(stdout, "We can not bind all threads into one single CPU!\n");
	  multilog(runtime_log, LOG_ERR, "We can not bind all threads into one single CPU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
    }
  
  /* Init capture */
  init_capture(&conf);

  /* Do the job */
  threads(&conf);
  
  /* Destory log interface */
  multilog(runtime_log, LOG_INFO, "CAPTURE END\n\n");
  multilog_close(runtime_log);
  fclose(fp_log);

  return EXIT_SUCCESS;
}
