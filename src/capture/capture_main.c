#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <string.h>
#include "dada_def.h"
#include "capture.h"
#include "log.h"

extern uint64_t        ndf_port[MPORT_CAPTURE];
extern uint64_t        ndf_chk[MCHK_CAPTURE];
extern pthread_mutex_t log_mutex;

void usage()
{
  fprintf(stdout,
	  "capture_main - capture PAF BMF raw data from NiC\n"
	  "\n"
	  "Usage: paf_capture [options]\n"
	  " -a Hexadecimal shared memory key for capture \n"
	  " -b Start point of packet\n"
	  " -c Alive IP adress and port, accept multiple values with -e value1 -e value2 ... the format of it is \"ip;port;nchunk_expected;nchunk_actual;cpu\" \n"
	  " -d Dead IP adress and port, accept multiple values with -e value1 -e value2 ... the format of it is \"ip;port;nchunk_expected\" \n"
	  " -e The center frequency of captured data\n"
	  " -f Reference information for the current capture, get from BMF packet header, epoch;sec;idf\n"
	  " -g Which directory to put runtime file\n"
	  " -h Show help\n"
	  " -i The CPU for buf control thread\n"
	  " -j The setup for the capture control, the format of it is \"cpt_ctrl;cpt_ctrl_cpu\"\n"
	  " -k Bind thread to CPU or not\n"
	  " -l The number of data frames in each buffer block of each frequency chunk\n"
	  " -m The number of data frames in each temp buffer of each frequency chunk\n"
	  " -n The name of header template for PSRDADA\n"
	  " -o The name of instrument \n"
	  " -p The source information, which is required for the case without capture control, in the format \"name;ra;dec\" \n"
	  " -q Force to pad band edge \n"
	   );
}

int main(int argc, char **argv)
{
  /* Initial part */
  int i, arg, source = 0;
  conf_t conf;
  char command_line[MSTR_LEN];
    
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
	      fprintf(stdout, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      fflush(stdout);
	      usage();
	      
	      exit(EXIT_FAILURE);
	    }
	  break;

	case 'b':	  	  
	  sscanf(optarg, "%d", &conf.dfoff);
	  break;

	case 'c':
	  sscanf(optarg, "%[^;];%d;%d;%d;%d", conf.ip_alive[conf.nport_alive], &conf.port_alive[conf.nport_alive], &conf.nchk_alive_expect[conf.nport_alive], &conf.nchk_alive_actual[conf.nport_alive], &conf.cpt_cpu[conf.nport_alive]);
	  conf.nport_alive++;
	  break;
	  
	case 'd':
	  sscanf(optarg, "%[^;];%d;%d", conf.ip_dead[conf.nport_dead], &conf.port_dead[conf.nport_dead], &conf.nchk_dead[conf.nport_dead]);
	  conf.nport_dead++;
	  break;
	  
	case 'e':
	  sscanf(optarg, "%lf", &conf.cfreq);
	  break;

	case 'f':
	  sscanf(optarg, "%d;%"SCNu64";%"SCNu64"", &conf.epoch_ref, &conf.sec_ref, &conf.idf_prd_ref);
	  break;
	  
	case 'g':
	  sscanf(optarg, "%s", conf.dir);  // It should be different for different beams and the directory name should be setup by pipeline;
	  break;
	  
	case 'i':
	  sscanf(optarg, "%d", &conf.rbuf_ctrl_cpu);
	  break;
	  
	case 'j':
	  sscanf(optarg, "%d;%d", &conf.cpt_ctrl, &conf.cpt_ctrl_cpu);
	  break;
	  
	case 'k':
	  sscanf(optarg, "%d", &conf.cpu_bind);
	  break;
	  	  
	case 'l':
	  sscanf(optarg, "%"SCNu64"", &conf.rbuf_ndf_chk);
	  break;
	  
	case 'm':
	  sscanf(optarg, "%"SCNu64"", &conf.tbuf_ndf_chk);
	  break;
	  
	case 'n':
	  sscanf(optarg, "%s", conf.hfname);
	  break;
	  
	case 'o':
	  sscanf(optarg, "%s", conf.instrument);
	  break;
	  
	case 'p':
	  {
	    source = 1;
	    sscanf(optarg, "%[^;];%[^;];%s", conf.source, conf.ra, conf.dec);
	    break;
	  }
	  
	case 'q':
	  sscanf(optarg, "%d", &conf.pad);
	  break;
	}
    }

  /* Setup log interface */
  char fname_log[MSTR_LEN];
  sprintf(fname_log, "%s/capture.log", conf.dir);  // The file will be in different directory for different beam;
  conf.logfile = paf_log_open(fname_log, "ab+");
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "CAPTURE START");

  /* Log the input parameters */
  for(i = 0; i < argc; i++)
    {
      strcat(command_line, " ");
      strcat(command_line, argv[i]);
    }
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "The command line is \"%s\"", command_line);    
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Hexadecimal shared memory key for capture is %x", conf.key);
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Start point of packet is %d", conf.dfoff);
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "We have %d alive ports, which are:", conf.nport_alive);
  for(i = 0; i < conf.nport_alive; i++)
    paf_log_add(conf.logfile, "INFO", 1, log_mutex, "    ip %s, port %d, expected frequency chunks %d and actual frequency chunks %d", conf.ip_alive[i], conf.port_alive[i], conf.nchk_alive_expect[i], conf.nchk_alive_actual[i]);

  if(conf.nport_dead == 0)
    paf_log_add(conf.logfile, "INFO", 1, log_mutex, "We do not have dead ports");
  else
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "We have %d dead ports, which are:", conf.nport_dead);
      for(i = 0; i < conf.nport_dead; i++)
	paf_log_add(conf.logfile, "INFO", 1, log_mutex, "    ip %s, port %d, expected frequency chunks %d", conf.ip_alive[i], conf.port_alive[i], conf.nchk_dead[i]);
    }
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "The center frequency for the capture is %f MHz", conf.cfreq);
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "The reference information for the capture is: epoch %d, seconds %"PRIu64" and location of packet in the period %"PRIu64"", conf.epoch_ref, conf.sec_ref, conf.idf_prd_ref);
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "The runtime information is %s", conf.dir);
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Buffer control thread runs on CPU %d", conf.rbuf_ctrl_cpu);
  if(conf.cpt_ctrl)
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "We will NOT enable sod at the beginning");
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Capture control thread runs on CPU %d", conf.cpt_ctrl_cpu);
    }
  else
    paf_log_add(conf.logfile, "INFO", 1, log_mutex, "We will enable sod at the beginning");
  if(conf.cpu_bind)
    paf_log_add(conf.logfile, "INFO", 1, log_mutex, "We will bind threads to CPUs");
  else
    paf_log_add(conf.logfile, "INFO", 1, log_mutex, "We will NOT bind threads to CPUs");
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Each ring buffer block has %"PRIu64" packets per frequency chunk", conf.rbuf_ndf_chk);
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Each temp buffer has %"PRIu64" packets per frequency chunk", conf.tbuf_ndf_chk);
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "The name of header template of PSRDADA is %s", conf.hfname);
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "The name of instrument of PSRDADA is %s", conf.instrument);
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "The source name is %s, RA is %s and DEC is %s", conf.source, conf.ra, conf.dec);
  if(conf.pad)
    paf_log_add(conf.logfile, "INFO", 1, log_mutex, "We will pad frequency chunks to fake full bandwidth data");
  else
    paf_log_add(conf.logfile, "INFO", 1, log_mutex, "We will NOT pad frequency chunks to fake full bandwidth data");
  
  /* Check the input */
  if((conf.cpt_ctrl == 0) && (source == 0))
    {
      paf_log_add(conf.logfile, "ERR", 1, log_mutex, "The target information will not be set, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);      
      paf_log_close(conf.logfile);
      exit(EXIT_FAILURE);
    }
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
	  paf_log_add(conf.logfile, "ERR", 1, log_mutex, "We can not bind all threads into one single CPU, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  paf_log_close(conf.logfile);	  
	  exit(EXIT_FAILURE);
	}
    }

  /* Init capture */
  init_capture(&conf);
  
  /* Do the job */
  threads(&conf);
  
  /* Destory capture */
  destroy_capture(conf);
  
  /* Destory log interface */
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "CAPTURE END");  
  paf_log_close(conf.logfile);

  return EXIT_SUCCESS;
}
