#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

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

#include "dada_cuda.h"
#include "ipcbuf.h"
#include "capture.h"
#include "log.h"
#include "psrdada.h"

int examine_record_arguments(conf_t conf, char **argv, int argc)
{
  int i;
  char command[MSTR_LEN] = {'\0'};
  
  strcpy(command, argv[0]);
  for(i = 1; i < argc; i++)
    {
      strcat(command, " ");
      strcat(command, argv[i]);
    }
  log_add(conf.log_file, "INFO", 1,  "The command line is \"%s\"", command);
  log_add(conf.log_file, "INFO", 1,  "Hexadecimal shared memory key for capture is %x", conf.key); // Check it when create HDU
  log_add(conf.log_file, "INFO", 1,  "We receive data from ip %s, port %d, expected frequency chunks %d and actual frequency chunks %d", conf.ip, conf.port, conf.nchunk_expect, conf.nchunk_actual);
  log_add(conf.log_file, "INFO", 1,  "The runtime information is %s", conf.dir); // This has already been checked before
  
  if((atoi(conf.dada_header.receiver)<0) || (atoi(conf.dada_header.receiver)>=NBEAM_MAX)) // More careful check later
    {
      fprintf(stderr, "CAPTURE_ERROR: The beam index is %s, which is not in range [0 %d), happens at \"%s\", line [%d], has to abort.\n", conf.dada_header.receiver, NBEAM_MAX - 1, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "The beam index is %s, which is not in range [0 %d), happens at \"%s\", line [%d], has to abort.", conf.dada_header.receiver, NBEAM_MAX - 1, __FILE__, __LINE__);
      
      fclose(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "We capture data from beam %s, counting from 0", conf.dada_header.receiver);
  
  if((conf.dfsz_seek != 0) && (conf.dfsz_seek != DF_HDRSZ)) // The seek bytes has to be 0 or DF_HDRSZ
    {
      fprintf(stderr, "CAPTURE_ERROR: The start point of packet is %d, but it should be 0 or %d, happens at \"%s\", line [%d], has to abort.\n", conf.dfsz_seek, DF_HDRSZ, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "The start point of packet is %d, but it should be 0 or %d, happens at \"%s\", line [%d], has to abort.", conf.dfsz_seek, DF_HDRSZ, __FILE__, __LINE__);
      
      fclose(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "Start point of packet is %d", conf.dfsz_seek);

  if(conf.dada_header.freq > BAND_LIMIT_UP || conf.dada_header.freq < BAND_LIMIT_DOWN)
    // The reference information has to be reasonable, later more careful check
    {
      fprintf(stderr, "CAPTURE_ERROR: cfreq %f is not in (%f %f), happens at \"%s\", line [%d], has to abort.\n",conf.dada_header.freq, BAND_LIMIT_DOWN, BAND_LIMIT_UP, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "cfreq %f is not in (%f %f), happens at \"%s\", line [%d], has to abort.",conf.dada_header.freq, BAND_LIMIT_DOWN, BAND_LIMIT_UP, __FILE__, __LINE__);
      
      fclose(conf.log_file);
      exit(EXIT_FAILURE);
    }  
  log_add(conf.log_file, "INFO", 1,  "The center frequency for the capture is %f MHz", conf.dada_header.freq); 

  if((conf.days_from_1970 <= 0) || (conf.df_in_period >= NDF_PER_CHUNK_PER_PERIOD))
    // The reference information has to be reasonable, later more careful check
    {
      fprintf(stderr, "CAPTURE_ERROR: The reference information is not right, days_from_1970 is %d and df_in_period is %"PRId64", happens at \"%s\", line [%d], has to abort.\n", conf.days_from_1970, conf.df_in_period, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "The reference information is not right, days_from_1970 is %d and df_in_period is %"PRId64", happens at \"%s\", line [%d], has to abort.", conf.days_from_1970, conf.df_in_period, __FILE__, __LINE__);
      
      fclose(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "The reference information for the capture is: epoch %d, seconds %"PRId64" and location of packet in the period %"PRId64"", conf.days_from_1970, conf.seconds_from_epoch, conf.df_in_period);

  if(conf.ndf_per_chunk_rbuf==0)  // The actual size of it will be checked later
    {      
      fprintf(stderr, "CAPTURE_ERROR: ndf_per_chunk_rbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_rbuf, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "ndf_per_chunk_rbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort", conf.ndf_per_chunk_rbuf, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "Each ring buffer block has %"PRIu64" packets per frequency chunk", conf.ndf_per_chunk_rbuf);
  
  if(conf.ndf_per_chunk_tbuf==0)  // The actual size of it will be checked later
    {      
      fprintf(stderr, "CAPTURE_ERROR: ndf_per_chunk_tbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_tbuf, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "ndf_per_chunk_tbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort", conf.ndf_per_chunk_tbuf, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1,  "Each temp buffer has %"PRIu64" packets per frequency chunk", conf.ndf_per_chunk_tbuf);

  if(access(conf.dada_header_template, F_OK ) != -1 )
    log_add(conf.log_file, "INFO", 1,  "The name of header template of PSRDADA is %s", conf.dada_header_template);
  else
    {        
      fprintf(stderr, "CAPTURE_ERROR: dada_header_template %s is not exist, which happens at \"%s\", line [%d], has to abort\n", conf.dada_header_template, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "dada_header_template %s is not exist, which happens at \"%s\", line [%d], has to abort", conf.dada_header_template, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
    
  return EXIT_SUCCESS;
}

int default_arguments(conf_t *conf)
{
  conf->dfsz_seek           = 0;       // Default to record the packet header
  conf->dada_header.freq    = 0;       // Default with an impossible number
  conf->days_from_1970      = 0;       // Default with an impossible number
  conf->seconds_from_epoch  = -1;      // Default with an impossible number
  conf->df_in_period        = -1;      // Default with an impossible number
  conf->port                = -1;      //
  conf->dfsz_seek           = DF_HDRSZ;// Default not record BMF packet header
  
  conf->ndf_per_chunk_rbuf = 0;  // Default with an impossible value
  conf->ndf_per_chunk_tbuf = 0;  // Default with an impossible value

  memset(conf->ip, 0x00, sizeof(conf->ip));
  memset(conf->dir, 0x00, sizeof(conf->dir));
  memset(conf->dada_header.receiver, 0x00, sizeof(conf->dada_header.receiver));
  memset(conf->dada_header_template, 0x00, sizeof(conf->dada_header_template));

  sprintf(conf->ip, "unset"); // Default with "unset"  
  sprintf(conf->dir, "unset");             // Default with "unset"
  sprintf(conf->dada_header.receiver, "unset");             // Default with "unset"
  sprintf(conf->dada_header_template, "unset"); // Default with "unset"
  
  return EXIT_SUCCESS;
}

int create_buffer(conf_t *conf)
{
  int write = 1;
  int dbregister = 0;
  int create = 1;

  if(dada_hdu(conf->hdu, conf->key, create, write, dbregister))
    {        
      fprintf(stderr, "CAPTURE_ERROR: Can not create buffer, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      log_add(conf->log_file, "ERR", 1,  "Can not create buffer, which happens at \"%s\", line [%d], has to abort",__FILE__, __LINE__);
      
      log_close(conf->log_file);
      exit(EXIT_FAILURE);
    }
  conf->data_block   = (ipcbuf_t *)(conf->hdu->data_block);
  conf->header_block = (ipcbuf_t *)(conf->hdu->header_block);
  conf->tbuf = (char *)malloc(conf->tbufsz * NBYTE_CHAR);
  
  return EXIT_SUCCESS;
}

int destory_buffer(conf_t conf)
{
  int write = 1;
  int dbregister = 0;
  int create = 0;
  
  if(dada_hdu(conf.hdu, conf.key, create, write, dbregister))
    {        
      fprintf(stderr, "CAPTURE_ERROR: Can not destory buffer, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1,  "Can not destory buffer, which happens at \"%s\", line [%d], has to abort",__FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  conf.data_block = NULL;
  conf.header_block = NULL;
  free(conf.tbuf);
  
  return EXIT_SUCCESS;
}
