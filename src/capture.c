#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdarg.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <byteswap.h>
#include <linux/un.h>
#include <unistd.h>
#include <dirent.h>
#include <inttypes.h>

#include "dada_cuda.h"
#include "ipcbuf.h"
#include "capture.h"
#include "log.h"
#include "psrdada.h"

void usage()
{
  fprintf(stdout,
	  "capture_main - capture PAF BMF raw data from NiC and record it with PSRDADA ring buffer\n"
	  "Usage: paf_capture [options]\n"
	  " -h Show help\n"
	  " -a The network interface of BMF stream, \"ip_port\" \n"
	  " -b The status of BMF stream, \"nchunkExpected_nchunkActual\" \n"
	  " -c The reference of recorded BMF packets, \"epoch_sec_idf\" \n"
	  " -d The beam index of recorded BMF packets, \"RECEIVER\" in PSRDADA header \n"
	  " -e The center frequency of recorded BMF packets \n"
	  " -f The start position of recorded BMF packets, to record the header of BMF packets or not\n"	  
	  " -g The name of PSRDADA header template \n"
	  " -i Hexadecimal shared memory key of PSRDADA ring buffer \n"
	  " -j The number of data frames in each buffer block of each frequency chunk\n"
	  " -k The number of data frames in each temp buffer of each frequency chunk\n"
	  " -l The name of the directory for runtime files\n"
	  " -m The flag to enable \"debug\" mode, \"ipcbuf_disable_sod\" if debug mode is enabled\n\n"
	  );
}

int initialize_capture(conf_t *conf, int argc, char **argv)
{
  char fname_log[MSTR_LEN] = {'\0'};
  
  /* default arguments*/
  default_arguments(conf);
  
  /* Parse arguments */
  parse_arguments(conf, argc, argv);
  
  /* Setup log interface */
  DIR* dir = opendir(conf->dir); // First to check if the directory exists
  if(dir)
    closedir(dir);
  else
    {
      fprintf(stderr, "CAPTURE_ERROR: Failed to open %s with opendir or it does not exist, ", conf->dir);
      fprintf(stderr, "which happens at which happens at \"%s\", line [%d], has to abort\n\n", __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
  sprintf(fname_log, "%s/capture.log", conf->dir);  // Open the log file
  conf->log_file = log_open(fname_log, "ab+");
  if(conf->log_file == NULL)
    {
      fprintf(stderr, "CAPTURE_ERROR: Can not open log file %s, ", fname_log);
      fprintf(stderr, "which happends at \"%s\", line [%d], has to abort\n\n", __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
  log_add_wrap(*conf, "INFO", 1,  "CAPTURE START\n");
  
  /* check the command line and record it */
  verify_arguments(*conf, argc, argv);

  /* Setup other parameters */
  conf->tres_rbuf_blk = conf->ndf_per_chunk_rbuf * TRES_DF;  
  conf->dfsz_keep     = DFSZ - conf->dfsz_seek;
  conf->blksz_rbuf    = conf->nchunk_expect * conf->dfsz_keep * conf->ndf_per_chunk_rbuf;
  conf->tbufsz        = (conf->dfsz_keep) * conf->ndf_per_chunk_tbuf * conf->nchunk_expect;
  conf->tout.tv_sec   = floor(conf->tres_rbuf_blk);  // Time out if we do not receive data for one buffer block
  conf->tout.tv_usec  = 1.0E6*(conf->tres_rbuf_blk - conf->tout.tv_sec);
  
  log_add_wrap(*conf, "INFO", 1,  "tres_rbuf_blk is %f \n", conf->tres_rbuf_blk);
  log_add_wrap(*conf, "INFO", 1,  "dfsz_keep is %d \n", conf->dfsz_keep);
  log_add_wrap(*conf, "INFO", 1,  "blksz_rbuf is %"PRIu64" \n", conf->blksz_rbuf);
  log_add_wrap(*conf, "INFO", 1,  "tbufsz is %"PRIu64" \n", conf->tbufsz);
  if(conf->debug == 1)
    {
      conf->write = 2;
      log_add_wrap(*conf, "INFO", 1,  "We will disable_sod\n");
    }
  else
    {
      conf->write = 1;
      log_add_wrap(*conf, "INFO", 1,  "We will not disable_sod\n");
    }
  
  /* Create buffers */
  create_buffer(conf);

  /* Setup DADA header if it is necessary */
  if(conf->debug == 0)
    {
      /* Read DADA header from template file if necessary */
      if(read_dada_header_from_file(conf->dada_header_template, &(conf->dada_header)))
	{
	  fprintf(stderr, "CAPTURE_ERROR: Can not read dada header template file %s, ", conf->dada_header_template);
	  fprintf(stderr, "which happends at \"%s\", line [%d], has to abort\n\n", __FILE__, __LINE__);
	  exit(EXIT_FAILURE);	  
	}

      /* Update the dada header and get ready for next */
      update_dada_header(conf);
      if(write_dada_header(conf->header_block, conf->dada_header))
	{
	  fprintf(stderr, "CAPTURE_ERROR: Fail to write header block, ");
	  fprintf(stderr, "which happends at \"%s\", line [%d], has to abort\n\n", __FILE__, __LINE__);
	  exit(EXIT_FAILURE);	  
	}
    }

  return EXIT_SUCCESS;
}

int do_capture(conf_t conf)
{
  double freq, time_offset = 0;
  int sock, chunk_index, reuseaddr = 1;
  struct sockaddr_in sa = {0}, fromsa = {0};
  socklen_t fromlen = sizeof(fromsa);
  uint64_t df_in_period, df_in_period_ref, seconds_from_epoch, seconds_from_epoch_ref;
  uint64_t tbuf_loc, rbuf_loc, ndf_per_chunk_rtbuf;
  uint64_t ndf_in_rbuf_blk_expect, ndf_in_rbuf_blk = 0, ndf = 0, ndf_expect = 0;
  uint64_t ndf_in_tbuf = 0;
  int64_t df_from_ref;
  char *rbuf;
  
  /* Setup */
  seconds_from_epoch_ref = conf.seconds_from_epoch;
  df_in_period_ref       = conf.df_in_period;
  ndf_per_chunk_rtbuf    = conf.ndf_per_chunk_rbuf + conf.ndf_per_chunk_tbuf;
  ndf_in_rbuf_blk_expect = conf.ndf_per_chunk_rbuf * conf.nchunk_actual;
  rbuf                   = ipcbuf_get_next_write(conf.data_block);
  if(rbuf == NULL)
    {
      log_add_wrap(conf, "ERR", 0, "ipcbuf_get_next_write failed\n");
      log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: ipcbuf_get_next_write failed, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  log_add_wrap(conf, "INFO", 1,  "Setup do_capture function, done\n");
  
  /* Setup socket and bind it */
  sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&conf.tout, sizeof(conf.tout));
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuseaddr, sizeof(reuseaddr));
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(conf.port);
  sa.sin_addr.s_addr = inet_addr(conf.ip);    
  if(bind(sock, (struct sockaddr *)&sa, sizeof(sa)) == -1)
    {
      log_add_wrap(conf, "ERR", 0,  "Can not bind to %s_%d\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port));
      log_add_wrap(conf, "ERR", 1,  "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: Can not bind to %s_%d ", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port));
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);

      close(sock);
      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  log_add_wrap(conf, "INFO", 1,  "Setup data capture socket, done\n");
  
  /* Receive one packet and check where are we */
  if(recvfrom(sock, (void *)conf.dbuf, DFSZ, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
    {
      log_add_wrap(conf, "ERR", 0, "Can not receive data from %s_%d\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port));
      log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: Can not receive data from %s_%d ", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port));
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);

      close(sock);
      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  if(decode_df_header(conf.dbuf, &df_in_period, &seconds_from_epoch, &freq))
    {
      log_add_wrap(conf, "ERR", 0, "Can not decode BMF packet header\n");
      log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: Can not decode BMF packet header ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);

      close(sock);
      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  
  df_from_ref = ((int64_t)(df_in_period - df_in_period_ref) + ((double)seconds_from_epoch - (double)seconds_from_epoch_ref) / TRES_DF);
  if(df_from_ref > 0) // the reference has to be in future
    {
      log_add_wrap(conf, "ERR", 0, "The reference is in the past \n");
      log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: The reference is in the past ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);

      close(sock);
      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  log_add_wrap(conf, "INFO", 1,  "Check first packet, done\n");
  
  /* Tell pipeline that the capture is ready */
  fprintf(stdout, "CAPTURE_READY\n"); 
  fflush(stdout);

  /* Do the capture */
  while(true)
    {
      /* Receive packet */
      if(recvfrom(sock, (void *)conf.dbuf, DFSZ, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
	{
	  log_add_wrap(conf, "ERR", 0, "Can not receive data from %s_%d\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port));
	  log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  
	  fprintf(stderr, "CAPTURE_ERROR: Can not receive data from %s_%d ", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port));
	  fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);

	  close(sock);
	  destroy_capture(conf);
	  exit(EXIT_FAILURE);
	}
      
      /* Decode packet header */
      if(decode_df_header(conf.dbuf, &df_in_period, &seconds_from_epoch, &freq))
	{
	  log_add_wrap(conf, "ERR", 0, "Can not decode BMF packet header\n");
	  log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  
	  fprintf(stderr, "CAPTURE_ERROR: Can not decode BMF packet header ");
	  fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);

	  close(sock);
	  destroy_capture(conf);
	  exit(EXIT_FAILURE);
	}

      /* Get the location of the packet */
      df_from_ref = ((int64_t)(df_in_period - df_in_period_ref) + ((double)seconds_from_epoch - (double)seconds_from_epoch_ref) / TRES_DF);
      chunk_index = ((int)((freq - conf.freq + 0.5)/NCHAN_PER_CHUNK + 0.5 * conf.nchunk_expect));	
      if ((chunk_index<0) || (chunk_index >= conf.nchunk_expect))
	{      
	  log_add_wrap(conf, "ERR", 0, "Frequency chunk %d is outside the range [0 %d]\n", chunk_index, conf.nchunk_expect);
	  log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  
	  fprintf(stderr, "CAPTURE_ERROR: Frequency chunk %d is outside the range [0 %d], ", chunk_index, conf.nchunk_expect);
	  fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

	  close(sock);
	  destroy_capture(conf);
	  exit(EXIT_FAILURE);
	}

      /* Record data into current ring buffer block if it is covered by it */
      if((df_from_ref>=0) && (df_from_ref < (int64_t)conf.ndf_per_chunk_rbuf))  
	{
	  rbuf_loc = (uint64_t)((df_from_ref * conf.nchunk_expect + chunk_index) * conf.dfsz_keep);         // This is in TFTFP order
	  memcpy(rbuf + rbuf_loc, conf.dbuf + conf.dfsz_seek, conf.dfsz_keep);
	  ndf_in_rbuf_blk++; 
	}

      /* Record data into temp buffer if it is covered by it */
      if((df_from_ref >= (int64_t)conf.ndf_per_chunk_rbuf) && (df_from_ref < (int64_t)ndf_per_chunk_rtbuf))
	{		  
	  tbuf_loc  = (uint64_t)(((df_from_ref - conf.ndf_per_chunk_rbuf) * conf.nchunk_expect + chunk_index) * conf.dfsz_keep);
	  memcpy(conf.tbuf + tbuf_loc, conf.dbuf + conf.dfsz_seek, conf.dfsz_keep);
	  ndf_in_tbuf++;
	}

      /* 
	 Move to a new ring buffer block if temp buffer is half 
	 or data can not be covered either by current ring buffer block or temp buffer 
      */
      if((df_from_ref >= (int64_t)ndf_per_chunk_rtbuf) || (ndf_in_tbuf >= (int64_t)(0.5 * (conf.ndf_per_chunk_tbuf * conf.nchunk_expect))))
	{
	  log_add_wrap(conf, "INFO", 1,  "Change ring buffer block, start\n");
	  /* Close current ring buffer block */
	  if(ipcbuf_mark_filled(conf.data_block, conf.blksz_rbuf) < 0)  
	    {
	      log_add_wrap(conf, "ERR", 0, "ipcbuf_mark_filled failed\n");
	      log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
	      fprintf(stderr, "CAPTURE_ERROR: ipcbuf_mark_filled failed, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

	      close(sock);
	      destroy_capture(conf);
	      exit(EXIT_FAILURE);
	    }
	  log_add_wrap(conf, "INFO", 1,  "Close current buffer block, done\n");
	  
	  /*
	    To see if the buffer is full, quit if yes.
	    If we have a reader, there will be at least one buffer which is not full
	  */
	  if(ipcbuf_get_nfull(conf.data_block) >= (ipcbuf_get_nbufs(conf.data_block) - 1)) 
	    {	      
	      log_add_wrap(conf, "ERR", 0, "Buffers are all full\n");
	      log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      
	      fprintf(stderr, "CAPTURE_ERROR: buffers are all full, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      
	      close(sock);
	      destroy_capture(conf);
	      exit(EXIT_FAILURE);
	    }
	  log_add_wrap(conf, "INFO", 1,  "Check available buffer block, done\n");
	  
	  /* Open a new ring buffer block */
	  rbuf = ipcbuf_get_next_write(conf.data_block);  
	  if(rbuf == NULL)
	    {
	      log_add_wrap(conf, "ERR", 0, "ipcbuf_get_next_write failed\n");
	      log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      
	      fprintf(stderr, "CAPTURE_ERROR: ipcbuf_get_next_write failed, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

	      close(sock);
	      destroy_capture(conf);
	      exit(EXIT_FAILURE);
	    }
	  log_add_wrap(conf, "INFO", 1,  "Open new buffer block, done\n");
	  
	  /* Check the packet loss rate */
	  time_offset += conf.tres_rbuf_blk;
	  ndf_expect  += ndf_in_rbuf_blk_expect;
	  ndf         += (ndf_in_rbuf_blk + ndf_in_tbuf);
	  fprintf(stdout, "HERE 1 %"PRId64"\t%.0f\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\n", df_from_ref, 0.5 * (conf.ndf_per_chunk_tbuf * conf.nchunk_expect), ndf_per_chunk_rtbuf, ndf_in_tbuf, ndf_in_rbuf_blk + ndf_in_tbuf, ndf_in_rbuf_blk_expect);
	  
	  fprintf(stdout, "CAPTURE_STATUS: %f %E %E\n", time_offset, 1.0 - ndf/(double)ndf_expect, 1.0 - (ndf_in_rbuf_blk + ndf_in_tbuf)/(double)ndf_in_rbuf_blk_expect);
	  fflush(stdout);	  
	  log_add_wrap(conf, "INFO", 1,  "Check packet loss rate %f %E %E, done\n", time_offset, 1.0 - ndf/(double)ndf_expect, 1.0 - (ndf_in_rbuf_blk + ndf_in_tbuf)/(double)ndf_in_rbuf_blk_expect);

	  /* Update reference */
	  df_in_period_ref    += conf.ndf_per_chunk_rbuf;
	  if(df_in_period_ref >= NDF_PER_CHUNK_PER_PERIOD)
	    {
	      seconds_from_epoch_ref += PERIOD;
	      df_in_period_ref       -= NDF_PER_CHUNK_PER_PERIOD;
	    }
	  log_add_wrap(conf, "INFO", 1,  "Update reference, done\n");

	  /* Copy temp buffer to ring buffer block */
	  memcpy(rbuf, conf.tbuf, conf.ndf_per_chunk_tbuf * conf.nchunk_expect * conf.dfsz_keep);		  
	  log_add_wrap(conf, "INFO", 1,  "Copy temp buffer to ring buffer block, done\n");

	  /* Reset counters*/
	  ndf_in_rbuf_blk = 0;
	  ndf_in_tbuf     = 0;
	  
	  log_add_wrap(conf, "INFO", 1,  "Change ring buffer block, done\n");
	}      
    }
  
  return EXIT_SUCCESS;
}

int destroy_capture(conf_t conf)
{
  destroy_buffer(conf);
  
  if(conf.log_file)
    {
      log_add_wrap(conf, "INFO", 1,  "CAPTURE END\n");
      log_close(conf.log_file);
    }
  
  return EXIT_SUCCESS;
}

int parse_arguments(conf_t *conf, int argc, char **argv)
{
  int arg;
  /* read in argument from command line */
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:g:hi:j:k:l:m:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);
	  
	case 'a':
	  if(sscanf(optarg, "%[^_]_%d", conf->ip, &conf->port) != 2)
	    {
	      fprintf(stderr, "CAPTURE_ERROR: Could not get the network interface of BMF stream, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n\n", __FILE__, __LINE__);
	      usage();
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'b':
	  if(sscanf(optarg, "%d_%d", &conf->nchunk_expect, &conf->nchunk_actual) != 2)
	    {
	      fprintf(stderr, "CAPTURE_ERROR: Could not get the status of BMF stream, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n\n", __FILE__, __LINE__);
	      usage();
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'c':
	  if(sscanf(optarg, "%d_%"SCNd64"_%"SCNd64"", &conf->days_from_1970, &conf->seconds_from_epoch, &conf->df_in_period) != 3)
	    {
	      fprintf(stderr, "CAPTURE_ERROR: Could not get the reference of recorded BMF packets, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n\n", __FILE__, __LINE__);
	      usage();	      
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'd':
	  if(sscanf(optarg, "%s", conf->receiver) != 1)
	    {
	      fprintf(stderr, "CAPTURE_ERROR: Could not get the beam index of recorded BMF packets, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n\n", __FILE__, __LINE__);
	      usage();	      
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'e':
	  if(sscanf(optarg, "%lf", &conf->freq) != 1)	
	    {
	      fprintf(stderr, "CAPTURE_ERROR: Could not get the center frequency of recorded BMF packets, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n\n", __FILE__, __LINE__);
	      usage();	      
	      exit(EXIT_FAILURE);
	    }
	  break;

	case 'f':	  	  
	  if(sscanf(optarg, "%d", &conf->dfsz_seek) != 1)	
	    {
	      fprintf(stderr, "CAPTURE_ERROR: Could not get the start position of recorded BMF packets, ");
	      fprintf(stderr, " which happens at \"%s\", line [%d], has to abort.\n\n", __FILE__, __LINE__);
	      usage();	      
	      exit(EXIT_FAILURE);
	    }
	  break;

	case 'g':
	  if(sscanf(optarg, "%s", conf->dada_header_template) != 1)	
	    {
	      fprintf(stderr, "CAPTURE_ERROR: Could not get the name of PSRDADA header template, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n\n", __FILE__, __LINE__);
	      usage();	      
	      exit(EXIT_FAILURE);
	    }       							   	
	  break;
	  
	case 'i':	  	  
	  if(sscanf(optarg, "%x", &conf->key) != 1)
	    {
	      fprintf(stderr, "CAPTURE_ERROR: Could not parse key from %s, ", optarg);
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n\n", __FILE__, __LINE__);
	      usage();	      
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'j':
	  if(sscanf(optarg, "%"SCNu64"", &conf->ndf_per_chunk_rbuf) != 1)
	    {
	      fprintf(stderr, "CAPTURE_ERROR: Could not get the number of data frames ");
	      fprintf(stderr, "in each buffer block of each frequency chunk, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n\n", __FILE__, __LINE__);
	      usage();	      
	      exit(EXIT_FAILURE);
	    }	
	  break;
	  
	case 'k':
	  if(sscanf(optarg, "%"SCNu64"", &conf->ndf_per_chunk_tbuf) != 1)
	    {
	      fprintf(stderr, "CAPTURE_ERROR: Could not get the number of data frames ");
	      fprintf(stderr, "in each temp buffer of each frequency chunk, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n\n", __FILE__, __LINE__);
	      usage();	      
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'l':
	  if(sscanf(optarg, "%s", conf->dir) != 1)
	    {
	      fprintf(stderr, "CAPTURE_ERROR: Could not get the name of the directory for runtime files, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n\n", __FILE__, __LINE__);
	      usage();	      
	      exit(EXIT_FAILURE);
	    }
	  break;	 
	  	  
	case 'm':
	  if(sscanf(optarg, "%d", &conf->debug) != 1)
	    {
	      fprintf(stderr, "CAPTURE_ERROR: Could not get the \"debug\" mode flag, ");
	      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n\n", __FILE__, __LINE__);
	      usage();	      
	      exit(EXIT_FAILURE);
	    }
	  break;
	}
    }
  
  return EXIT_SUCCESS;
}

int verify_arguments(conf_t conf, int argc, char **argv)
{
  int i;
  char command[MSTR_LEN] = {'\0'};
  
  strcpy(command, argv[0]);
  for(i = 1; i < argc; i++)
    {
      strcat(command, " ");
      strcat(command, argv[i]);
    }
  log_add_wrap(conf, "INFO", 1,  "The command line is \"%s\"\n", command);
  log_add_wrap(conf, "INFO", 1,  "Hexadecimal shared memory key of PSRDADA ring buffer is %x\n", conf.key); // Check it when create HDU
  log_add_wrap(conf, "INFO", 1,  "We receive data from ip %s, port %d\n", conf.ip, conf.port);
  log_add_wrap(conf, "INFO", 1,  "We expect %d frequency chunks\n", conf.nchunk_expect);
  log_add_wrap(conf, "INFO", 1,  "We will get %d frequency chunks\n", conf.nchunk_actual);
  log_add_wrap(conf, "INFO", 1,  "The runtime files are in %s\n", conf.dir); // This has already been checked before

  if((atoi(conf.receiver)<0) || (atoi(conf.receiver)>=NBEAM_MAX)) // More careful check later
    {
      log_add_wrap(conf, "ERR", 0, "The beam index is %s\n", conf.receiver);
      log_add_wrap(conf, "ERR", 0, "It is not in range [0 %d)\n", NBEAM_MAX - 1);
      log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: The beam index is %s and ", conf.receiver);
      fprintf(stderr, "it is not in range [0 %d), ", NBEAM_MAX - 1);
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  log_add_wrap(conf, "INFO", 1,  "The beam index is %s\n", conf.receiver);
  
  if((conf.dfsz_seek != 0) && (conf.dfsz_seek != DF_HDRSZ)) // The seek bytes has to be 0 or DF_HDRSZ
    {
      log_add_wrap(conf, "ERR", 0, "The start point of packet is %d\n", conf.dfsz_seek);
      log_add_wrap(conf, "ERR", 0, "It should be 0 or %d\n", DF_HDRSZ);
      log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: The start point of packet is %d ,", conf.dfsz_seek);
      fprintf(stderr, "but it should be 0 or %d, ", DF_HDRSZ);
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  log_add_wrap(conf, "INFO", 1,  "Start position of packet is %d\n", conf.dfsz_seek);

  if(conf.freq > BAND_LIMIT_UP || conf.freq < BAND_LIMIT_DOWN)
    // The reference information has to be reasonable, later more careful check
    {
      log_add_wrap(conf, "ERR", 0,  "The center frequency %f is not in (%f %f)\n", conf.freq, BAND_LIMIT_DOWN, BAND_LIMIT_UP);
      log_add_wrap(conf, "ERR", 1,  "Which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
            
      fprintf(stderr, "CAPTURE_ERROR: cfreq %f is not in (%f %f), ", conf.freq, BAND_LIMIT_DOWN, BAND_LIMIT_UP);
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }  
  log_add_wrap(conf, "INFO", 1,  "The center frequency is %f MHz\n", conf.freq); 

  if((conf.days_from_1970 <= 0) || (conf.df_in_period >= NDF_PER_CHUNK_PER_PERIOD))
    // The reference information has to be reasonable, later more careful check
    {
      log_add_wrap(conf, "ERR", 0,  "The reference information is not right\n");
      log_add_wrap(conf, "ERR", 0,  "The days_from_1970 is %d and df_in_period is %"PRId64"\n", conf.days_from_1970, conf.df_in_period);
      log_add_wrap(conf, "ERR", 1,  "Which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: The reference information is not right, ");
      fprintf(stderr, "days_from_1970 is %d and df_in_period is %"PRId64", ", conf.days_from_1970, conf.df_in_period);
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  log_add_wrap(conf, "INFO", 1,  "The reference information for the capture is: \n");
  log_add_wrap(conf, "INFO", 1,  "epoch %d\n", conf.days_from_1970);
  log_add_wrap(conf, "INFO", 1,  "seconds %"PRId64"\n", conf.seconds_from_epoch);
  log_add_wrap(conf, "INFO", 1,  "location of packet in the period %"PRId64"\n", conf.df_in_period);

  if(conf.ndf_per_chunk_rbuf==0)  // The actual size of it will be checked later
    {      
      log_add_wrap(conf, "ERR", 0,  "ndf_per_chunk_rbuf shoule be a positive number\n");
      log_add_wrap(conf, "ERR", 0,  "It is %"PRIu64"\n");
      log_add_wrap(conf, "ERR", 1,  "Which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_rbuf, __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: ndf_per_chunk_rbuf shoule be a positive number, ");
      fprintf(stderr, "but it is %"PRIu64", ", conf.ndf_per_chunk_rbuf);
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      
      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  log_add_wrap(conf, "INFO", 1,  "Each ring buffer block has %"PRIu64" packets per frequency chunk\n", conf.ndf_per_chunk_rbuf);
  
  if(conf.ndf_per_chunk_tbuf==0)  // The actual size of it will be checked later
    {      
      log_add_wrap(conf, "ERR", 0, "ndf_per_chunk_tbuf shoule be a positive number\n");
      log_add_wrap(conf, "ERR", 0, "It is %"PRIu64"\n", conf.ndf_per_chunk_tbuf);
      log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: ndf_per_chunk_tbuf shoule be a positive number, ");
      fprintf(stderr, "but it is %"PRIu64", ", conf.ndf_per_chunk_tbuf);
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);

      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  log_add_wrap(conf, "INFO", 1, "Each temp buffer has %"PRIu64" packets per frequency chunk\n", conf.ndf_per_chunk_tbuf);

  if(conf.debug == 0)
    {
      if(access(conf.dada_header_template, F_OK ) != -1 )
	log_add_wrap(conf, "INFO", 1, "The name of header template of PSRDADA is %s\n", conf.dada_header_template);
      else
	{        
	  log_add_wrap(conf, "ERR", 1, "dada_header_template %s is not exist\n", conf.dada_header_template);
	  log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  
	  fprintf(stderr, "CAPTURE_ERROR: dada_header_template %s is not exist, ", conf.dada_header_template);
	  fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  
	  destroy_capture(conf);
	  exit(EXIT_FAILURE);
	}
    }
    
  return EXIT_SUCCESS;
}

int default_arguments(conf_t *conf)
{
  conf->dfsz_seek           = 0;       // Default to record the packet header
  conf->freq                = 0;       // Default with an impossible number
  conf->days_from_1970      = 0;       // Default with an impossible number  
  conf->seconds_from_epoch  = -1;      // Default with an impossible number
  conf->df_in_period        = -1;      // Default with an impossible number
  conf->port                = -1;      //
  conf->dfsz_seek           = DF_HDRSZ;// Default not record BMF packet header
  conf->debug               = 1;       // Default to run with disable_sod
  conf->ndf_per_chunk_rbuf = 0;  // Default with an impossible value
  conf->ndf_per_chunk_tbuf = 0;  // Default with an impossible value
  conf->dbregister         = 0;  // Default not register ring buffer
  conf->write              = 1;  // Default as a writer and start data at beginning
  
  memset(conf->ip, 0x00, sizeof(conf->ip));
  memset(conf->dir, 0x00, sizeof(conf->dir));
  memset(conf->receiver, 0x00, sizeof(conf->receiver));
  memset(conf->dada_header_template, 0x00, sizeof(conf->dada_header_template));

  sprintf(conf->ip, "unset"); // Default with "unset"  
  sprintf(conf->dir, "unset");             // Default with "unset"
  sprintf(conf->receiver, "unset");             // Default with "unset"
  sprintf(conf->dada_header_template, "unset"); // Default with "unset"
  
  return EXIT_SUCCESS;
}

int create_buffer(conf_t *conf)
{
  int write = conf->write;
  int dbregister = conf->dbregister;
  
  conf->hdu = dada_hdu_create_wrap(conf->key, write, dbregister);
  if(conf->hdu == NULL)
    {        
      log_add_wrap(*conf, "ERR", 0,  "Can not create ring buffer\n");
      log_add_wrap(*conf, "ERR", 1,  "Which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: Can not create ring buffer, ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      
      destroy_capture(*conf);
      exit(EXIT_FAILURE);
    }

  conf->data_block   = (ipcbuf_t *)(conf->hdu->data_block);
  conf->header_block = (ipcbuf_t *)(conf->hdu->header_block);
  conf->tbuf = (char *)malloc(NBYTE_CHAR * conf->tbufsz);
  conf->dbuf = (char *)malloc(NBYTE_CHAR * DFSZ);
  
  return EXIT_SUCCESS;
}

int destroy_buffer(conf_t conf)
{
  int write = conf.write;
  int dbregister = conf.dbregister;
  
  if(dada_hdu_destroy_wrap(conf.hdu, conf.key, write, dbregister))
    {        
      log_add_wrap(conf, "ERR", 0, "Can not destroy ring buffer\n");
      log_add_wrap(conf, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: Can not destroy ring buffer ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);

      if(conf.log_file)
	{
	  log_add_wrap(conf, "INFO", 1,  "CAPTURE END\n");
	  log_close(conf.log_file);
	}
      exit(EXIT_FAILURE);
    }
  conf.data_block = NULL;
  conf.header_block = NULL;
  
  if (conf.tbuf)
    free(conf.tbuf);
  if (conf.dbuf)
    free(conf.dbuf);
  
  return EXIT_SUCCESS;
}

int decode_df_header(char *dbuf, uint64_t *df_in_period, uint64_t *seconds_from_epoch, double *freq)
{
  uint64_t *ptr = NULL, writebuf;

  if(dbuf == NULL)
    {
      fprintf(stderr, "CAPTURE_ERROR: dbuf is NULL ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  ptr      = (uint64_t*)dbuf;
  writebuf = bswap_64(*ptr);
  *df_in_period = writebuf & 0x00000000ffffffff;
  *seconds_from_epoch = (writebuf & 0x3fffffff00000000) >> 32;

  writebuf = bswap_64(*(ptr + 2));
  *freq     = (double)((writebuf & 0x00000000ffff0000) >> 16);
  
  return EXIT_SUCCESS;
}

int update_dada_header(conf_t *conf)
{
  int nchan; 
  time_t seconds_from_1970;
  double seconds_in_period, scale;

  nchan = conf->nchunk_expect * NCHAN_PER_CHUNK; // Number of channels in real;
  scale = nchan / (double)conf->dada_header.nchan;
  
  seconds_in_period = TRES_DF * conf->df_in_period;  
  seconds_from_1970 = floor(seconds_in_period) + conf->seconds_from_epoch + SECDAY * conf->days_from_1970;

  conf->dada_header.freq  = conf->freq;
  conf->dada_header.nchan =  nchan;
  conf->dada_header.bw   *= scale;  
    
  memset(conf->dada_header.receiver, '\0', sizeof(conf->dada_header.receiver));
  strncpy(conf->dada_header.receiver, conf->receiver, strlen(conf->receiver));
  
  conf->dada_header.file_size        *= scale;
  conf->dada_header.bytes_per_second *= scale;

  conf->dada_header.mjd_start   = seconds_from_1970 / SECDAY + MJD1970;  // Float MJD start time without fraction second
  conf->dada_header.picoseconds = 1E6 * round(1.0E6 * (seconds_in_period - floor(seconds_in_period)));
  
  strftime (conf->dada_header.utc_start, MSTR_LEN, DADA_TIMESTR, gmtime(&seconds_from_1970)); // String start time without fraction second  

  log_add_wrap(*conf, "INFO", 1,  "scale is %f \n", scale);
  log_add_wrap(*conf, "INFO", 1,  "BW is %f MHz\n", conf->dada_header.bw);
  log_add_wrap(*conf, "INFO", 1,  "NCHAN is %d \n", conf->dada_header.nchan);
  log_add_wrap(*conf, "INFO", 1,  "FREQ is %f MHz\n", conf->dada_header.freq);
  log_add_wrap(*conf, "INFO", 1,  "RECEIVER is %s\n", conf->dada_header.receiver);
  log_add_wrap(*conf, "INFO", 1,  "MJD_START is %f \n", conf->dada_header.mjd_start);
  log_add_wrap(*conf, "INFO", 1,  "UTC_START is %s \n", conf->dada_header.utc_start);
  log_add_wrap(*conf, "INFO", 1,  "FILE_SIZE is %"PRIu64" \n", conf->dada_header.file_size);
  log_add_wrap(*conf, "INFO", 1,  "PICOSECONDS is %"PRIu64" \n", conf->dada_header.picoseconds);
  log_add_wrap(*conf, "INFO", 1,  "BYTES_PER_SECOND is %"PRIu64" \n", conf->dada_header.bytes_per_second);
  
  return EXIT_SUCCESS;
}

int log_add_wrap(conf_t conf, const char *type, int flush, const char *format, ...)
{
  va_list args;

  va_start(args, format);
  if(log_add(conf.log_file, type, flush, format, args))
    {
      va_end (args);

      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  va_end (args);
  
  return EXIT_SUCCESS;
}
