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

#include "ipcbuf.h"
#include "capture.h"
#include "multilog.h"

int initialize_capture(int argc, char **argv, configuration_t *configuration)
{
  /* Parse the input arguments */
  parse_arguments(argc, argv, configuration);
  
  /* Initialize ring buffer and temporary buffer */
  initialize_buffer(configuration);

  /* Initialize socket */
  initialize_socket(configuration);
  
  return EXIT_SUCCESS;
}

void usage()
{
  fprintf(stdout,
	  "capture_main - capture PAF BMF raw data from NiC\n"
	  "\n"
	  "Usage: paf_capture [options]\n"
	  " -h Show help\n"
	  " -a Hexadecimal shared memory key for capture \n"
	  " -b The size of each beamformer packet, includs the header of it\n"
	  " -c Start point of beamformer packet to record, to decide record packet header or not\n"

	  " -d The number of data frames in each period of each frequency chunk\n"
	  " -e Frequency channels in each chunk\n"
	  " -f Beamformer streaming period\n"

	  " -g IP adress and port of data arrives, the format of it is \"ip:port:nchunk_expected:nchunk_actual\" \n"
	  " -i The center frequency of arriving data\n"
	  " -j Reference information for the current capture, get from BMF packet header, epoch:sec:idf\n"
	  " -k To directory to record runtime information\n"
	  
	  " -l The number of data frames in each buffer block of each frequency chunk\n"
	  " -m The number of data frames in each temp buffer of each frequency chunk\n"

	  " -n The name of header template for PSRDADA\n"
	  " -o The name of instrument \n"
	  );
}

int parse_arguments(int argc, char **argv, configuration_t *configuration)
{  
  int arg;
  int nparam_expect = 9;
  int nparam_actual = 0;

  /* Parse the input and check it */
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:g:hi:j:k:l:m:n:o:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);
	  
	case 'a':	  	  
	  if(sscanf(optarg, "%x", &configuration->rbuf_key) != 1)
	    {
	      fprintf(stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  nparam_actual++;
	  break;

	case 'b':
	  sscanf(optarg, "%d", &configuration->pktsize_bytes);
	  nparam_actual++;
	  break;

	case 'c':
	  sscanf(optarg, "%d", &configuration->offset_pktsize_bytes);
	  nparam_actual++;
	  break;
	  
	case 'd':
	  sscanf(optarg, "%"SCNu64"", &configuration->npkt_per_chunk_period);
	  nparam_actual++;
	  break;
	  
	case 'e':
	  sscanf(optarg, "%d", &configuration->nchan_per_chunk);
	  nparam_actual++;
	  break;
	  
	case 'f':
	  sscanf(optarg, "%d", &configuration->pkt_period_secs);
	  nparam_actual++;
	  break;
	  
	case 'g':
	  sscanf(optarg, "%[^:]:%d:%d:%d", configuration->ip, &configuration->port, &configuration->nchunk_expect, &configuration->nchunk_actual);
	  nparam_actual++;
	  break;

	case 'i':
	  sscanf(optarg, "%lf", &configuration->freq);
	  nparam_actual++;
	  break;

	case 'j':
	  sscanf(optarg, "%d:%"SCNu64":%"SCNu64"", &configuration->refpkt_epoch, &configuration->refpkt_secs, &configuration->refpkt_idx_period);
	  nparam_actual++;
	  break;
	  
	case 'k':
	  sscanf(optarg, "%s", configuration->runtime_dir);
	  nparam_actual++;
	  break;
	  
	case 'l':
	  sscanf(optarg, "%"SCNu64"", &configuration->npkt_per_chunk_rbuf);
	  nparam_actual++;
	  break;
	  
	case 'm':
	  sscanf(optarg, "%"SCNu64"", &configuration->npkt_per_chunk_tbuf);
	  nparam_actual++;
	  break;
	  
	case 'n':
	  sscanf(optarg, "%s", configuration->dada_hdr_fname);
	  nparam_actual++;
	  break;
	  
	case 'o':
	  sscanf(optarg, "%s", configuration->instrument_name);
	  nparam_actual++;
	  break;
	}
    }
  if(nparam_actual != nparam_expect)
    {
      fprintf(stderr, "%d parameters are missing!!!\n", nparam_expect - nparam_actual);
      usage();
      exit(EXIT_FAILURE);
    }

  /* Do some simple calculations */
  configuration->nchan_expect         = configuration->nchan_per_chunk * configuration->nchunk_expect;
  configuration->nchan_actual         = configuration->nchan_per_chunk * configuration->nchunk_actual;
  
  configuration->remind_pktsize_bytes = configuration->pktsize_bytes - configuration->offset_pktsize_bytes;
  configuration->pkt_tres_secs        = configuration->pkt_period_secs/(double)configuration->npkt_per_chunk_period;
  configuration->rbuf_tres_secs       = configuration->pkt_tres_secs * configuration->npkt_per_chunk_rbuf;
  configuration->refchunk_idx        = -(configuration->freq + 1.0)/configuration->nchan_per_chunk + 0.5 * configuration->nchan_expect;
  
  configuration->int_reftime_secs   = floor(configuration->pkt_tres_secs * configuration->refpkt_idx_period) + configuration->refpkt_secs + SECDAY * configuration->refpkt_epoch;
  configuration->frac_reftime_psecs = 1E6 * round(1.0E6 * (configuration->pkt_period_secs - floor(configuration->pkt_tres_secs * configuration->refpkt_idx_period)));
  
  configuration->timeout.tv_sec     = configuration->pkt_period_secs;
  configuration->timeout.tv_usec    = 0.0;
  
  configuration->rbufsize_bytes     = configuration->nchunk_expect * configuration->remind_pktsize_bytes * configuration->npkt_per_chunk_rbuf;
  configuration->tbufsize_bytes     = configuration->nchunk_expect * configuration->remind_pktsize_bytes * configuration->npkt_per_chunk_tbuf;
  configuration->tbuf_thred_pkts    = (uint64_t)(0.5 * configuration->nchan_expect * configuration->npkt_per_chunk_tbuf);
  
  /* Setup log interface */
  char fname_log[MAX_STRLEN];
  FILE *fp_log = NULL;
  sprintf(fname_log, "%s/capture.log", configuration->runtime_dir);  // The file will be in different directory for different beam;
  fp_log = fopen(fname_log, "ab+"); 
  if(fp_log == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", fname_log);
      exit(EXIT_FAILURE);
    }

  /* Log the information so far */
  configuration->runtime_log = multilog_open("capture", 1);
  multilog_add(configuration->runtime_log, fp_log);
  multilog(configuration->runtime_log, LOG_INFO, "CAPTURE START\n\n");
  multilog(configuration->runtime_log, LOG_INFO, "We are going to capture data from %s:%d.\n", configuration->ip, configuration->port);
  multilog(configuration->runtime_log, LOG_INFO, "The DADA key for the capture is %x.\n", configuration->rbuf_key);
  multilog(configuration->runtime_log, LOG_INFO, "The center frequency of the capture is %f.\n", configuration->freq);
  multilog(configuration->runtime_log, LOG_INFO, "The reference information of the capture is %"PRIu64"%d:%"PRIu64".\n", configuration->refpkt_epoch, configuration->refpkt_secs, configuration->refpkt_idx_period);
  multilog(configuration->runtime_log, LOG_INFO, "The reference time information is %lld:%"PRIu64".\n", (long long)configuration->int_reftime_secs, configuration->frac_reftime_psecs);
  multilog(configuration->runtime_log, LOG_INFO, "The packet time resolution and ring buffer time resolution are %f and %f seconds.\n", configuration->pkt_tres_secs, configuration->rbuf_tres_secs);
  multilog(configuration->runtime_log, LOG_INFO, "The reference frequency chunk index is %f.\n", configuration->refchunk_idx);  
  multilog(configuration->runtime_log, LOG_INFO, "The data header template of the capture is %s.\n", configuration->dada_hdr_fname);
  multilog(configuration->runtime_log, LOG_INFO, "The instrucment name will be %s in the DADA header.\n", configuration->instrument_name);
  multilog(configuration->runtime_log, LOG_INFO, "The beamformer packet size is %d bytes, but the data we record will offset by %d bytes.\n", configuration->pktsize_bytes, configuration->offset_pktsize_bytes);
  multilog(configuration->runtime_log, LOG_INFO, "%d bytes of each beamformer packet will be recorded.\n", configuration->remind_pktsize_bytes);
  multilog(configuration->runtime_log, LOG_INFO, "%d frequency chunks are expected, but we can only get %d frequency chunks.\n", configuration->nchunk_expect, configuration->nchunk_actual);
  multilog(configuration->runtime_log, LOG_INFO, "%d frequency channels are expected, but we can only get %d frequency channels.\n", configuration->nchan_expect, configuration->nchan_actual);
  multilog(configuration->runtime_log, LOG_INFO, "The stream period is %d seconds.\n", configuration->pkt_period_secs);
  multilog(configuration->runtime_log, LOG_INFO, "%"PRIu64" of packets in each period for each frequency chunk.\n", configuration->npkt_per_chunk_period);  
  multilog(configuration->runtime_log, LOG_INFO, "%"PRIu64" of packets in each ring buffer block for each frequency chunk.\n", configuration->npkt_per_chunk_rbuf);
  multilog(configuration->runtime_log, LOG_INFO, "%"PRIu64" of packets in each temporary buffer for each frequency chunk.\n", configuration->npkt_per_chunk_tbuf);
  multilog(configuration->runtime_log, LOG_INFO, "The size of ring buffer and temporart buffer are %"PRIu64" and %"PRIu64" bytes.\n\n", configuration->rbufsize_bytes, configuration->tbufsize_bytes);
  multilog(configuration->runtime_log, LOG_INFO, "The threshold for the buffer change is %"PRIu64" packets.\n\n", configuration->tbuf_thred_pkts);
  
  return EXIT_SUCCESS;
}

int initialize_buffer(configuration_t *configuration)
{
  /* Create HDU and check the size of buffer bolck */
  configuration->hdu = dada_hdu_create(configuration->runtime_log);
  dada_hdu_set_key(configuration->hdu, configuration->rbuf_key);
  if(dada_hdu_connect(configuration->hdu) < 0)
    {
      multilog(configuration->runtime_log, LOG_ERR, "Can not connect to hdu, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      multilog(configuration->runtime_log, LOG_INFO, "CAPTURE ABORT\n\n");
      
      fclose(configuration->fp_log);
      dada_hdu_destroy(configuration->hdu);
      exit(EXIT_FAILURE);    
    }
  
  /* make ourselves the write client */
  if(dada_hdu_lock_write(configuration->hdu) < 0) 
    {
      multilog(configuration->runtime_log, LOG_ERR, "Error locking HDU, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      multilog(configuration->runtime_log, LOG_INFO, "CAPTURE ABORT\n\n");
      
      fclose(configuration->fp_log);
      dada_hdu_disconnect(configuration->hdu);
      exit(EXIT_FAILURE);
    }
  
  /* DADA DB for data and header */
  configuration->db_data = (ipcbuf_t *)(configuration->hdu->data_block);
  configuration->db_hdr  = (ipcbuf_t *)(configuration->hdu->header_block);
  if((configuration->db_data == NULL) || (configuration->db_hdr == NULL))
    {
      multilog(configuration->runtime_log, LOG_ERR, "Can not get dada_db, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      multilog(configuration->runtime_log, LOG_INFO, "CAPTURE ABORT\n\n");
      
      fclose(configuration->fp_log);
      dada_hdu_destroy(configuration->hdu);
      exit(EXIT_FAILURE);  
    }
      
  /* Check the buffer size */
  if(configuration->rbufsize_bytes != ipcbuf_get_bufsz(configuration->db_data))  
    {
      multilog(configuration->runtime_log, LOG_ERR, "Buffer size mismatch, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      multilog(configuration->runtime_log, LOG_INFO, "CAPTURE ABORT\n\n");
      
      fclose(configuration->fp_log);
      dada_hdu_destroy(configuration->hdu);
      exit(EXIT_FAILURE);    
    }

  /* Disable start-of-data */
  if(ipcbuf_disable_sod(configuration->db_data) < 0)
    {
      multilog(configuration->runtime_log, LOG_ERR, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      multilog(configuration->runtime_log, LOG_INFO, "CAPTURE ABORT\n\n");
      
      fclose(configuration->fp_log);
      dada_hdu_destroy(configuration->hdu);
      exit(EXIT_FAILURE);
    }

  /* Create temporary buffer */
  configuration->tbuf = (char *)malloc(configuration->tbufsize_bytes * sizeof(char));  
  if(configuration->tbuf == NULL)
    {	     
      multilog(configuration->runtime_log, LOG_ERR, "can not malloc memory for temporary buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      multilog(configuration->runtime_log, LOG_INFO, "CAPTURE ABORT\n\n");
      
      fclose(configuration->fp_log);
      dada_hdu_destroy(configuration->hdu);
      exit(EXIT_FAILURE);
    }

  /* Get buffer to receive data from socket */
  configuration->pkt = (char *)malloc(sizeof(char) * configuration->pktsize_bytes);
  if(configuration->pkt == NULL)
    {
      multilog(configuration->runtime_log, LOG_ERR, "can not malloc memory, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      multilog(configuration->runtime_log, LOG_INFO, "CAPTURE ABORT\n\n");
      
      free(configuration->tbuf);
      fclose(configuration->fp_log);
      dada_hdu_destroy(configuration->hdu);
      exit(EXIT_FAILURE);
    }
  
  /* Get the first ring buffer block ready */
  configuration->rbuf = ipcbuf_get_next_write(configuration->db_data);
  if(configuration->rbuf == NULL)
    {	     
      multilog(configuration->runtime_log, LOG_ERR, "open_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      multilog(configuration->runtime_log, LOG_INFO, "CAPTURE ABORT\n\n");

      free(configuration->pkt);
      fclose(configuration->fp_log);
      dada_hdu_destroy(configuration->hdu);
      exit(EXIT_FAILURE);
    }
  
  return EXIT_SUCCESS;
}

int initialize_socket(configuration_t *configuration)
{
  struct sockaddr_in sa, fromsa;
  configuration->sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  setsockopt(configuration->sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&configuration->timeout, sizeof(configuration->timeout));  
  memset(&sa, 0x00, sizeof(sa));
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(configuration->port);
  sa.sin_addr.s_addr = inet_addr(configuration->ip);
  if(bind(configuration->sock, (struct sockaddr *)&sa, sizeof(sa)) == -1)
    {
      multilog(configuration->runtime_log, LOG_ERR,  "Can not bind to %s:%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
      multilog(configuration->runtime_log, LOG_INFO, "CAPTURE ABORT\n\n");
      
      free(configuration->pkt);
      free(configuration->tbuf);
      fclose(configuration->fp_log);
      dada_hdu_destroy(configuration->hdu);
      exit(EXIT_FAILURE);
    }

}
int do_capture(configuration_t configuration)
{
  socklen_t fromlen;
  struct sockaddr_in fromsa;
  uint64_t tbuf_loc, rbuf_loc;
  uint64_t writebuf, *ptr = NULL;
  double freq;
  uint64_t pkt_idx_period, pkt_secs;
  int64_t pkt_idx;
  int chunk_idx;
  uint64_t counter_rbuf = 0, counter_tbuf = 0;
  double elapsed_time = 0;
  uint64_t npkt_expect = 0, npkt_actual = 0;
  uint64_t npkt_expect_per_blk = configuration.npkt_per_chunk_rbuf * configuration.nchunk_actual;
  uint64_t npkt_actual_per_blk;
  
  /* Receive a packet */
  if(recvfrom(configuration.sock, (void *)configuration.pkt, configuration.pktsize_bytes, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
    {      
      multilog(configuration.runtime_log, LOG_ERR,  "Time out, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      multilog(configuration.runtime_log, LOG_INFO, "CAPTURE ABORT\n\n");
      
      free(configuration.pkt);
      free(configuration.tbuf);
      fclose(configuration.fp_log);
      dada_hdu_destroy(configuration.hdu);
      exit(EXIT_FAILURE);
    }

  /* index of packet and frequency chunks */
  ptr      = (uint64_t*)configuration.pkt;
  writebuf = bswap_64(*ptr);
  pkt_idx_period  = writebuf & 0x00000000ffffffff;
  pkt_secs = (writebuf & 0x3fffffff00000000) >> 32;
  writebuf = bswap_64(*(ptr + 2));
  freq     = (double)((writebuf & 0x00000000ffff0000) >> 16);
  
  pkt_idx   = (int64_t)(pkt_idx_period - configuration.refpkt_idx_period) + ((double)pkt_secs - (double)configuration.refpkt_secs) / configuration.pkt_tres_secs;
  chunk_idx = (int)(configuration.freq/configuration.nchan_per_chunk + configuration.refchunk_idx);

  /* Loop to receive data */
  while(true)
    {
      /* Only keep data in special range */
      if((configuration.npkt_per_chunk_rbuf + configuration.npkt_per_chunk_tbuf)>pkt_idx>=0)
	{
	  if(pkt_idx<configuration.npkt_per_chunk_rbuf) // Data going to ring buffer block
	    {
	      counter_rbuf ++;
	      rbuf_loc = (uint64_t)((pkt_idx * configuration.nchan_expect + chunk_idx) * configuration.remind_pktsize_bytes);
	      memcpy(configuration.rbuf + rbuf_loc, configuration.pkt + configuration.offset_pktsize_bytes, configuration.remind_pktsize_bytes);
	      continue;
	    }
	  else   // Data going to temporary buffer
	    {
	      counter_tbuf ++;
	      tbuf_loc = (uint64_t)(((pkt_idx - configuration.npkt_per_chunk_rbuf) * configuration.nchan_expect + chunk_idx) * configuration.remind_pktsize_bytes);
	      memcpy(configuration.tbuf + tbuf_loc, configuration.pkt + configuration.offset_pktsize_bytes, configuration.remind_pktsize_bytes);
	      
	      continue;
	    }

	  if(counter_tbuf > configuration.tbuf_thred_pkts) // Trigger the change
	    {
	      /* Close current ring buffer block */
	      if(ipcbuf_mark_filled(configuration.db_data, configuration.rbufsize_bytes) < 0)
		{
		  multilog(configuration.runtime_log, LOG_ERR, "close_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
		  multilog(configuration.runtime_log, LOG_INFO, "CAPTURE ABORT\n\n");
		  
		  free(configuration.pkt);
		  free(configuration.tbuf);
		  fclose(configuration.fp_log);
		  dada_hdu_destroy(configuration.hdu);
		  exit(EXIT_FAILURE);
		}
	      /*
		To see if the buffer is full, quit if yes.
		If we have a reader, there will be at least one buffer which is not full
	      */
	      if(ipcbuf_get_nfull(configuration.db_data) >= (ipcbuf_get_nbufs(configuration.db_data) - 1)) 
		{	     
		  multilog(configuration.runtime_log, LOG_ERR, "buffers are all full, which happens at \"%s\", line [%d], has to abort..\n", __FILE__, __LINE__);
		  multilog(configuration.runtime_log, LOG_INFO, "CAPTURE ABORT\n\n");
		  
		  free(configuration.pkt);
		  free(configuration.tbuf);
		  fclose(configuration.fp_log);
		  dada_hdu_destroy(configuration.hdu);
		  exit(EXIT_FAILURE);		  
		}

	      /* Get new ring buffer block and copy temporary memory into it */
	      configuration.rbuf = ipcbuf_get_next_write(configuration.db_data); 
	      if(configuration.rbuf == NULL)
		{
		  multilog(configuration.runtime_log, LOG_ERR, "open_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
		  multilog(configuration.runtime_log, LOG_INFO, "CAPTURE ABORT\n\n");
		  
		  free(configuration.pkt);
		  free(configuration.tbuf);
		  fclose(configuration.fp_log);
		  dada_hdu_destroy(configuration.hdu);
		  exit(EXIT_FAILURE);
		}
	      memcpy(configuration.rbuf, configuration.tbuf, configuration.tbufsize_bytes);
	      
	      /* Update reference point */
	      configuration.refpkt_idx_period += configuration.npkt_per_chunk_rbuf;
	      if(configuration.refpkt_idx_period >= configuration.npkt_per_chunk_period)       
		{
		  configuration.refpkt_secs       += configuration.pkt_period_secs;
		  configuration.refpkt_idx_period -= configuration.npkt_per_chunk_period;
		}

	      /* Check the traffic status */
	      npkt_actual_per_blk = counter_rbuf + counter_tbuf;
	      elapsed_time += configuration.pkt_tres_secs;
	      npkt_expect  += npkt_expect_per_blk;
	      npkt_actual  += npkt_actual_per_blk;

	      multilog(configuration.runtime_log, LOG_INFO, "%f seconds, loss rate in previous buffer block is %E, loss rate from beginning is %E\n", elapsed_time, 1.0 - npkt_actual_per_blk/(double)npkt_expect_per_blk, 1.0 - npkt_actual/(double)npkt_expect);
	      
	      counter_tbuf = 0;
	      counter_rbuf = 0;
	    }
	}
      
      /* Receive a packet */
      if(recvfrom(configuration.sock, (void *)configuration.pkt, configuration.pktsize_bytes, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
	{      
	  multilog(configuration.runtime_log, LOG_ERR,  "Time out, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  multilog(configuration.runtime_log, LOG_INFO, "CAPTURE ABORT\n\n");
	  
	  free(configuration.pkt);
	  free(configuration.tbuf);
	  fclose(configuration.fp_log);
	  dada_hdu_destroy(configuration.hdu);
	  exit(EXIT_FAILURE);
	}
      
      /* index of packet and frequency chunks */
      ptr      = (uint64_t*)configuration.pkt;
      writebuf = bswap_64(*ptr);
      pkt_idx_period  = writebuf & 0x00000000ffffffff;
      pkt_secs = (writebuf & 0x3fffffff00000000) >> 32;
      writebuf = bswap_64(*(ptr + 2));
      freq     = (double)((writebuf & 0x00000000ffff0000) >> 16);
      
      pkt_idx   = (int64_t)(pkt_idx_period - configuration.refpkt_idx_period) + ((double)pkt_secs - (double)configuration.refpkt_secs) / configuration.pkt_tres_secs;
      chunk_idx = (int)(configuration.freq/configuration.nchan_per_chunk + configuration.refchunk_idx);
    }
  
  return EXIT_SUCCESS;
}

int destroy_capture(configuration_t configuration)
{
  /* Finish the log */
  multilog(configuration.runtime_log, LOG_INFO, "CAPTURE DONE\n\n");
  
  if(configuration.fp_log != NULL)
    fclose(configuration.fp_log);

  /* destroy hdu */
  dada_hdu_destroy(configuration.hdu);

  /* Free temporary buffer */
  if (configuration.tbuf!=NULL)
    free(configuration.tbuf);

  /* Free buffer for data receiving */
  if(configuration.pkt!=NULL)
    free(configuration.pkt);

  /* Close socket */
  close(configuration.sock);
  
  return EXIT_SUCCESS;
}
