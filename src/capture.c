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
  log_add_wrap(*conf, "INFO", 1,  "CAPTURE START");
      
  /* check the command line and record it */
  verify_arguments(*conf, argc, argv);

  /* Create buffers */
  create_buffer(conf);

  /* Setup DADA header if it is necessary */
  if(conf->debug == 0)
    {
      /* Read DADA header from template file if necessary */
      if(read_dada_header_from_file(conf->dada_header_template, conf->dada_header))
	{
	  fprintf(stderr, "CAPTURE_ERROR: Can not read dada header template file %s, ", conf->dada_header_template);
	  fprintf(stderr, "which happends at \"%s\", line [%d], has to abort\n\n", __FILE__, __LINE__);
	  exit(EXIT_FAILURE);	  
	}    
      /* Update the dada header and get ready for next */
      update_dada_header(conf);
      if(write_dada_header(conf->header_block, *(conf->dada_header)))
	{
	  fprintf(stderr, "CAPTURE_ERROR: Fail to write header block, ");
	  fprintf(stderr, "which happends at \"%s\", line [%d], has to abort\n\n", __FILE__, __LINE__);
	  exit(EXIT_FAILURE);	  
	}
    }

  /* Setup other parameters */
  conf->time_res_blk = conf->ndf_per_chunk_rbuf * TIME_RES_DF;  
  conf->dfsz_keep    = DFSZ - conf->dfsz_seek;
  conf->blksz_rbuf   = conf->nchunk_expect * conf->dfsz_keep * conf->ndf_per_chunk_rbuf;
  conf->tbufsz       = (conf->dfsz_keep + 1) * conf->ndf_per_chunk_tbuf * conf->nchunk_expect;
  
  log_add_wrap(*conf, "INFO", 1,  "time_res_blk is %f \n", conf->time_res_blk);
  log_add_wrap(*conf, "INFO", 1,  "dfsz_keep is %d \n", conf->dfsz_keep);
  log_add_wrap(*conf, "INFO", 1,  "blksz_rbuf is %"PRIu64" \n", conf->blksz_rbuf);
  log_add_wrap(*conf, "INFO", 1,  "tbufsz is %"PRIu64" \n", conf->tbufsz);
  
  return EXIT_SUCCESS;
}

int destroy_capture(conf_t conf)
{
  destroy_buffer(conf);
  
  if(conf.log_file)
    {
      log_add(conf.log_file, "INFO", 1,  "CAPTURE END");
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
  log_add(conf.log_file, "INFO", 1,  "The command line is \"%s\"\n", command);
  log_add(conf.log_file, "INFO", 1,  "Hexadecimal shared memory key of PSRDADA ring buffer is %x\n", conf.key); // Check it when create HDU
  log_add(conf.log_file, "INFO", 1,  "We receive data from ip %s, port %d\n", conf.ip, conf.port);
  log_add(conf.log_file, "INFO", 1,  "We expect %d frequency chunks\n", conf.nchunk_expect);
  log_add(conf.log_file, "INFO", 1,  "We will get %d frequency chunks\n", conf.nchunk_actual);
  log_add(conf.log_file, "INFO", 1,  "The runtime files are in %s\n", conf.dir); // This has already been checked before
  
  if((atoi(conf.receiver)<0) || (atoi(conf.receiver)>=NBEAM_MAX)) // More careful check later
    {
      log_add(conf.log_file, "ERR", 0, "The beam index is %s\n", conf.dada_header->receiver);
      log_add(conf.log_file, "ERR", 0, "It is not in range [0 %d)\n", NBEAM_MAX - 1);
      log_add(conf.log_file, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort.", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: The beam index is %s and ", conf.dada_header->receiver);
      fprintf(stderr, "it is not in range [0 %d), ", NBEAM_MAX - 1);
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "The beam index is %s\n", conf.dada_header->receiver);
  
  if((conf.dfsz_seek != 0) && (conf.dfsz_seek != DF_HDRSZ)) // The seek bytes has to be 0 or DF_HDRSZ
    {
      log_add(conf.log_file, "ERR", 0, "The start point of packet is %d\n", conf.dfsz_seek);
      log_add(conf.log_file, "ERR", 0, "It should be 0 or %d\n", DF_HDRSZ);
      log_add(conf.log_file, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: The start point of packet is %d ,", conf.dfsz_seek);
      fprintf(stderr, "but it should be 0 or %d, ", DF_HDRSZ);
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "Start position of packet is %d", conf.dfsz_seek);

  if(conf.freq > BAND_LIMIT_UP || conf.freq < BAND_LIMIT_DOWN)
    // The reference information has to be reasonable, later more careful check
    {
      log_add(conf.log_file, "ERR", 0,  "The center frequency %f is not in (%f %f)\n", conf.dada_header->freq, BAND_LIMIT_DOWN, BAND_LIMIT_UP);
      log_add(conf.log_file, "ERR", 1,  "Which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
            
      fprintf(stderr, "CAPTURE_ERROR: cfreq %f is not in (%f %f), ", conf.dada_header->freq, BAND_LIMIT_DOWN, BAND_LIMIT_UP);
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }  
  log_add(conf.log_file, "INFO", 1,  "The center frequency is %f MHz", conf.dada_header->freq); 

  if((conf.days_from_1970 <= 0) || (conf.df_in_period >= NDF_PER_CHUNK_PER_PERIOD))
    // The reference information has to be reasonable, later more careful check
    {
      log_add(conf.log_file, "ERR", 0,  "The reference information is not right\n");
      log_add(conf.log_file, "ERR", 0,  "The days_from_1970 is %d and df_in_period is %"PRId64"\n", conf.days_from_1970, conf.df_in_period);
      log_add(conf.log_file, "ERR", 1,  "Which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: The reference information is not right, ");
      fprintf(stderr, "days_from_1970 is %d and df_in_period is %"PRId64", ", conf.days_from_1970, conf.df_in_period);
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "The reference information for the capture is: \n");
  log_add(conf.log_file, "INFO", 1,  "epoch %d\n", conf.days_from_1970);
  log_add(conf.log_file, "INFO", 1,  "seconds %"PRId64"\n", conf.seconds_from_epoch);
  log_add(conf.log_file, "INFO", 1,  "location of packet in the period %"PRId64"\n", conf.df_in_period);

  if(conf.ndf_per_chunk_rbuf==0)  // The actual size of it will be checked later
    {      
      log_add(conf.log_file, "ERR", 0,  "ndf_per_chunk_rbuf shoule be a positive number\n");
      log_add(conf.log_file, "ERR", 0,  "It is %"PRIu64"\n");
      log_add(conf.log_file, "ERR", 1,  "Which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_rbuf, __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: ndf_per_chunk_rbuf shoule be a positive number, ");
      fprintf(stderr, "but it is %"PRIu64", ", conf.ndf_per_chunk_rbuf);
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      
      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "Each ring buffer block has %"PRIu64" packets per frequency chunk", conf.ndf_per_chunk_rbuf);
  
  if(conf.ndf_per_chunk_tbuf==0)  // The actual size of it will be checked later
    {      
      log_add(conf.log_file, "ERR", 0, "ndf_per_chunk_tbuf shoule be a positive number\n");
      log_add(conf.log_file, "ERR", 0, "It is %"PRIu64"\n", conf.ndf_per_chunk_tbuf);
      log_add(conf.log_file, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: ndf_per_chunk_tbuf shoule be a positive number, ");
      fprintf(stderr, "but it is %"PRIu64", ", conf.ndf_per_chunk_tbuf);
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);

      destroy_capture(conf);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, "Each temp buffer has %"PRIu64" packets per frequency chunk", conf.ndf_per_chunk_tbuf);

  if(conf.debug == 0)
    {
      if(access(conf.dada_header_template, F_OK ) != -1 )
	log_add(conf.log_file, "INFO", 1, "The name of header template of PSRDADA is %s", conf.dada_header_template);
      else
	{        
	  log_add(conf.log_file, "ERR", 1, "dada_header_template %s is not exist\n", conf.dada_header_template);
	  log_add(conf.log_file, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	  
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
  conf->dada_header->freq    = 0;       // Default with an impossible number
  conf->days_from_1970      = 0;       // Default with an impossible number
  conf->seconds_from_epoch  = -1;      // Default with an impossible number
  conf->df_in_period        = -1;      // Default with an impossible number
  conf->port                = -1;      //
  conf->dfsz_seek           = DF_HDRSZ;// Default not record BMF packet header
  conf->debug               = 1;       // Default to run with disable_sod
  
  conf->ndf_per_chunk_rbuf = 0;  // Default with an impossible value
  conf->ndf_per_chunk_tbuf = 0;  // Default with an impossible value

  memset(conf->ip, 0x00, sizeof(conf->ip));
  memset(conf->dir, 0x00, sizeof(conf->dir));
  memset(conf->dada_header->receiver, 0x00, sizeof(conf->dada_header->receiver));
  memset(conf->dada_header_template, 0x00, sizeof(conf->dada_header_template));

  sprintf(conf->ip, "unset"); // Default with "unset"  
  sprintf(conf->dir, "unset");             // Default with "unset"
  sprintf(conf->dada_header->receiver, "unset");             // Default with "unset"
  sprintf(conf->dada_header_template, "unset"); // Default with "unset"
  
  return EXIT_SUCCESS;
}

int create_buffer(conf_t *conf)
{
  int write;
  int dbregister = 0;
  int create = 1;

  if (conf->debug == 0) 
    write = 1;
  else // disable_sod
    write = 2;
  
  if(dada_hdu(conf->hdu, conf->key, create, write, dbregister))
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
  int write = 1;
  int dbregister = 0;
  int create = 0;
  
  if(dada_hdu(conf.hdu, conf.key, create, write, dbregister))
    {        
      log_add(conf.log_file, "ERR", 0, "Can not destroy ring buffer\n");
      log_add(conf.log_file, "ERR", 1, "Which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      
      fprintf(stderr, "CAPTURE_ERROR: Can not destroy ring buffer ");
      fprintf(stderr, "which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);

      if(conf.log_file)
	{
	  log_add(conf.log_file, "INFO", 1,  "CAPTURE END");
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
  scale = nchan / (double)conf->dada_header->nchan;
  
  seconds_in_period = TIME_RES_DF * conf->df_in_period;  
  seconds_from_1970 = floor(seconds_in_period) + conf->seconds_from_epoch + SECDAY * conf->days_from_1970;

  conf->dada_header->freq     = conf->freq;
  conf->dada_header->nchan =  nchan;
  conf->dada_header->bw   *= scale;  

  memset(conf->dada_header->receiver, '\0', sizeof(conf->dada_header->receiver));
  strncpy(conf->dada_header->receiver, conf->receiver, strlen(conf->receiver));
  
  conf->dada_header->file_size        *= scale;
  conf->dada_header->bytes_per_second *= scale;

  conf->dada_header->mjd_start   = seconds_from_1970 / SECDAY + MJD1970;  // Float MJD start time without fraction second
  conf->dada_header->picoseconds = 1E6 * round(1.0E6 * (seconds_in_period - floor(seconds_in_period)));
  
  strftime (conf->dada_header->utc_start, MSTR_LEN, DADA_TIMESTR, gmtime(&seconds_from_1970)); // String start time without fraction second  

  log_add_wrap(*conf, "INFO", 1,  "scale is %f \n", scale);
  log_add_wrap(*conf, "INFO", 1,  "BW is %f MHz\n", conf->dada_header->bw);
  log_add_wrap(*conf, "INFO", 1,  "NCHAN is %d \n", conf->dada_header->nchan);
  log_add_wrap(*conf, "INFO", 1,  "FREQ is %f MHz\n", conf->dada_header->freq);
  log_add_wrap(*conf, "INFO", 1,  "RECEIVER is %s\n", conf->dada_header->receiver);
  log_add_wrap(*conf, "INFO", 1,  "MJD_START is %f \n", conf->dada_header->mjd_start);
  log_add_wrap(*conf, "INFO", 1,  "UTC_START is %s \n", conf->dada_header->utc_start);
  log_add_wrap(*conf, "INFO", 1,  "FILE_SIZE is %"PRIu64" \n", conf->dada_header->file_size);
  log_add_wrap(*conf, "INFO", 1,  "PICOSECONDS is %"PRIu64" \n", conf->dada_header->picoseconds);
  log_add_wrap(*conf, "INFO", 1,  "BYTES_PER_SECOND is %"PRIu64" \n", conf->dada_header->bytes_per_second);
  
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
