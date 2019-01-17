#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include "multilog.h"
#include "dada_def.h"
#include "capture.h"

multilog_t *runtime_log;
extern uint64_t ndf_port[MPORT_CAPTURE];
extern uint64_t ndf_chk[MCHK_CAPTURE];

void usage()
{
  fprintf(stdout,
	  "capture_main - capture PAF BMF raw data from NiC\n"
	  "\n"
	  "Usage: paf_capture [options]\n"
	  " -a Hexadecimal shared memory key for capture \n"
	  " -b BMF packet size\n"
	  " -c Start point of packet\n"
	  " -d Alive IP adress and port, accept multiple values with -e value1 -e value2 ... the format of it is \"ip:port:nchunk_expected:nchunk_actual:cpu\" \n"
	  " -e Dead IP adress and port, accept multiple values with -e value1 -e value2 ... the format of it is \"ip:port:nchunk_expected\" \n"
	  " -f The center frequency of captured data\n"
	  " -g Frequency channels in each chunk\n"
	  " -h Show help\n"
	  " -i Reference information for the current capture, get from BMF packet header, epoch:sec:idf\n"
	  " -j Which directory to put runtime file\n"
	  " -k The CPU for buf control thread\n"
	  " -l The setup for the capture control, the format of it is \"cpt_ctrl:cpt_ctrl_cpu\"\n"
	  " -m Bind thread to CPU or not\n"
	  " -n Streaming period\n"
	  " -o The number of data frames in each buffer block of each frequency chunk\n"
	  " -p The number of data frames in each temp buffer of each frequency chunk\n"
	  " -q The number of data frames in each period of each frequency chunk\n"
	  " -r The name of header template for PSRDADA\n"
	  " -s The name of instrument \n"
	  " -t The source information, which is required for the case without capture control, in the format \"name:ra:dec\" \n"
	  " -u Force to pad band edge \n"
	   );
}

int main(int argc, char **argv)
{
  /* Initial part */
  int i, arg, source = 0;
  conf_t conf;
  
  conf.nport_alive   = 0;
  conf.nport_dead    = 0;
  conf.rbuf_ctrl_cpu = 0;
  conf.cpt_ctrl_cpu  = 0;
  conf.cpt_ctrl      = 0;  // Default do not control the capture during the runtime;
  conf.cpu_bind      = 0;  // Default do not bind thread to cpu
  conf.pad           = 0;  // Default do not pad
  sprintf(conf.source, "unset");
  sprintf(conf.ra, "unset");
  sprintf(conf.dec, "unset");
  
  for(i = 0; i < MPORT_CAPTURE; i++)
    conf.cpt_cpu[i] = 0;
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:g:hi:j:k:l:m:n:o:p:q:r:s:t:u:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  return EXIT_FAILURE;
	  
	case 'a':	  	  
	  if(sscanf(optarg, "%x", &conf.key) != 1)
	    {
	      fprintf(stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;

	case 'b':	  	  
	  sscanf(optarg, "%d", &conf.dfsz);
	  break;

	case 'c':	  	  
	  sscanf(optarg, "%d", &conf.dfoff);
	  break;

	case 'd':
	  sscanf(optarg, "%[^:]:%d:%d:%d:%d", conf.ip_alive[conf.nport_alive], &conf.port_alive[conf.nport_alive], &conf.nchk_alive_expect[conf.nport_alive], &conf.nchk_alive_actual[conf.nport_alive], &conf.cpt_cpu[conf.nport_alive]);
	  conf.nport_alive++;
	  break;
	  
	case 'e':
	  sscanf(optarg, "%[^:]:%d:%d", conf.ip_dead[conf.nport_dead], &conf.port_dead[conf.nport_dead], &conf.nchk_dead[conf.nport_dead]);
	  conf.nport_dead++;
	  break;
	  
	case 'f':
	  sscanf(optarg, "%lf", &conf.cfreq);
	  break;

	case 'g':
	  sscanf(optarg, "%d", &conf.nchan_chk);
	  break;

	case 'i':
	  sscanf(optarg, "%d:%"SCNu64":%"SCNu64"", &conf.ref.epoch, &conf.ref.sec, &conf.ref.idf_prd);
	  break;
	  
	case 'j':
	  sscanf(optarg, "%s", conf.dir);  // It should be different for different beams and the directory name should be setup by pipeline;
	  break;
	  
	case 'k':
	  sscanf(optarg, "%d", &conf.rbuf_ctrl_cpu);
	  break;
	  
	case 'l':
	  sscanf(optarg, "%d:%d", &conf.cpt_ctrl, &conf.cpt_ctrl_cpu);
	  break;
	  
	case 'm':
	  sscanf(optarg, "%d", &conf.cpu_bind);
	  break;
	  
	case 'n':
	  sscanf(optarg, "%d", &conf.prd);
	  break;
	  
	case 'o':
	  sscanf(optarg, "%"SCNu64"", &conf.rbuf_ndf_chk);
	  break;
	  
	case 'p':
	  sscanf(optarg, "%"SCNu64"", &conf.tbuf_ndf_chk);
	  break;
	  
	case 'q':
	  sscanf(optarg, "%"SCNu64"", &conf.ndf_chk_prd);
	  break;
	  
	case 'r':
	  sscanf(optarg, "%s", conf.hfname);
	  break;
	  
	case 's':
	  sscanf(optarg, "%s", conf.instrument);
	  break;
	  
	case 't':
	  {
	    source = 1;
	    sscanf(optarg, "%s:%s:%s", conf.source, conf.ra, conf.dec);
	    break;
	  }
	  
	case 'u':
	  sscanf(optarg, "%d", &conf.pad);
	  break;
	}
    }

  /* Setup log interface */
  char fname_log[MSTR_LEN];
  FILE *fp_log = NULL;
  sprintf(fname_log, "%s/capture.log", conf.dir);  // The file will be in different directory for different beam;
  fp_log = fopen(fname_log, "ab+"); 
  if(fp_log == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", fname_log);
      return EXIT_FAILURE;
    }
  runtime_log = multilog_open("capture", 1);
  multilog_add(runtime_log, fp_log);
  multilog(runtime_log, LOG_INFO, "CAPTURE START\n");

  /* Check the input */
  if((conf.cpt_ctrl == 0) && (source == 0))
    multilog(runtime_log, LOG_WARNING, "The target information will not be set, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
  /* To make sure that we are not going to bind all threads to one sigle CPU */
  if(!(conf.cpu_bind == 0))
    {
      for(i = 0; i < conf.nport_alive; i++)
	{
	  if(((conf.cpt_cpu[i] == conf.cpt_ctrl_cpu)?0:1) == 1)
	    break;
	  if(((conf.cpt_cpu[i] == conf.rbuf_ctrl_cpu)?0:1) == 1)
	    break;
	}
      if(i == conf.nport_alive)
	{
	  multilog(runtime_log, LOG_ERR, "We can not bind all threads into one single CPU, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  return EXIT_FAILURE;
	}
    }

  /* Init capture */
  if(init_capture(&conf))
    {      
      multilog(runtime_log, LOG_ERR, "Can not initial capture, has to abort, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not initial capture, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      multilog(runtime_log, LOG_INFO, "CAPTURE END\n\n");
      multilog_close(runtime_log);
      fclose(fp_log);
      return EXIT_FAILURE;
    }
  
  /* Do the job */
  threads(&conf);

  /* Destory capture */
  destroy_capture(conf);
  
  /* Destory log interface */
  multilog(runtime_log, LOG_INFO, "CAPTURE END\n\n");
  multilog_close(runtime_log);
  fclose(fp_log);

  return EXIT_SUCCESS;
}
