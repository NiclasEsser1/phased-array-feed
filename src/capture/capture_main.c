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
	  " -d Active IP adress and port, accept multiple values with -e value1 -e value2 ... the format of it is \"ip:port:nchunk_expected:nchunk_actual:cpu\" \n"
	  " -e Dead IP adress and port, accept multiple values with -e value1 -e value2 ... the format of it is \"ip:port:nchunk_expected\" \n"
	  " -f The center frequency of captured data\n"
	  " -g Number of channels of current capture\n"
	  " -h Show help\n"
	  " -i Reference information for the current capture, get from BMF packet header, epoch_ref:sec_ref:idf_ref\n"
	  " -j Which directory to put log file\n"
	  " -k The CPU for buf control thread\n"
	  " -l The CPU for capture control thread\n"
	  " -m Bind thread to CPU or not\n"
	  " -n Time out for sockets\n"
	  " -o The number of chunks\n"
	  " -p The number of data frames in each buffer block of each frequency chunk\n"
	  " -q The number of data frames in each temp buffer of each frequency chunk\n"
	  " -r The number of data frames in each period or each frequency chunk\n"
	  " -s The address to get control signal, currently uses unix socket\n"
	  " -t The name of header template for PSRDADA\n"
	  " -u The name of instrument \n"
	   );
}

int main(int argc, char **argv)
{
  /* Initial part */
  int i, arg;
  conf_t conf;
  
  conf.nport_active = 0;
  conf.nport_dead   = 0;
  conf.buf_ctrl_cpu     = 0;
  conf.capture_ctrl_cpu = 0;
  conf.thread_bind  = 0; // Default do not bind thread to cpu
  for(i = 0; i < MPORT_CAPTURE; i++)
    conf.port_cpu[i] = 0;
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
	  sscanf(optarg, "%d", &conf.pktsz);
	  break;

	case 'c':	  	  
	  sscanf(optarg, "%d", &conf.pktoff);
	  break;

	case 'd':
	  sscanf(optarg, "%[^:]:%d:%d:%d:%d", conf.ip_active[conf.nport_active], &conf.port_active[conf.nport_active], &conf.nchk_active_expect[conf.nport_active], &conf.nchk_active_actual[conf.nport_active], &conf.port_cpu[conf.nport_active]);
	  conf.nport_active++;
	  break;
	  
	case 'e':
	  sscanf(optarg, "%[^:]:%d:%d", conf.ip_dead[conf.nport_dead], &conf.port_dead[conf.nport_dead], &conf.nchk_dead[conf.nport_dead]);
	  conf.nport_dead++;
	  break;
	  
	case 'f':
	  sscanf(optarg, "%lf", &conf.center_freq);
	  break;

	case 'g':
	  sscanf(optarg, "%d", &conf.nchan);
	  break;

	case 'i':
	  sscanf(optarg, "%lf:%"SCNu64":%"SCNu64"", &conf.epoch_ref, &conf.sec_ref, &conf.idf_ref);
	  break;
	  
	case 'j':
	  sscanf(optarg, "%s", conf.dir);
	  break;
	  
	case 'k':
	  sscanf(optarg, "%d", &conf.buf_ctrl_cpu);
	  break;
	  
	case 'l':
	  sscanf(optarg, "%d", &conf.capture_ctrl_cpu);
	  break;
	  
	case 'm':
	  sscanf(optarg, "%d", &conf.thread_bind);
	  break;
	  
	case 'n':
	  sscanf(optarg, "%d", &conf.sec_prd);
	  break;
	  
	case 'o':
	  sscanf(optarg, "%d", &conf.nchk);
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
	  
	case 's':
	  sscanf(optarg, "%s", conf.ctrl_addr);
	  break;
	  
	case 't':
	  sscanf(optarg, "%s", conf.hfname);
	  break;
	  
	case 'u':
	  sscanf(optarg, "%s", conf.instrument);
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
      for(i = 0; i < MPORT_CAPTURE; i++)
	{
	  if(((conf.port_cpu[i] == conf.capture_ctrl_cpu)?0:1) == 1)
	    break;
	  if(((conf.port_cpu[i] == conf.buf_ctrl_cpu)?0:1) == 1)
	    break;
	}
      if(i == MPORT_CAPTURE)
	{
	  fprintf(stdout, "We can not bind all threads into one single CPU, has to abort!\n");
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

  fprintf(stdout, "HERE, ENF OF THREADS\n");
  
  /* Destory capture */
  destroy_capture(conf);
  
  /* Destory log interface */
  multilog(runtime_log, LOG_INFO, "CAPTURE END\n\n");
  multilog_close(runtime_log);
  fclose(fp_log);

  //for(i = 0; i < 6; i++)
  //  fprintf(stdout, "%"PRIu64"\t", ndf_port[i]);
  //fprintf(stdout, "\n");
  
  return EXIT_SUCCESS;
}
