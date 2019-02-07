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

#include "ipcbuf.h"
#include "capture.h"
#include "log.h"

char *cbuf = NULL;
char *tbuf = NULL;
int quit   = 0; // 0 means no quit, 1 means quit normal, 2 means quit with problem;
int transit[MPORT_CAPTURE] = {0};

uint64_t ndf[MPORT_CAPTURE]  = {0};
uint64_t tail[MPORT_CAPTURE] = {0};
uint64_t ndf_advance[MPORT_CAPTURE];
uint64_t df_in_period_ref[MPORT_CAPTURE]; 
uint64_t seconds_from_epoch_ref[MPORT_CAPTURE];

pthread_mutex_t ref_mutex[MPORT_CAPTURE] = {PTHREAD_MUTEX_INITIALIZER};
pthread_mutex_t ndf_mutex[MPORT_CAPTURE] = {PTHREAD_MUTEX_INITIALIZER};
pthread_mutex_t log_mutex                = PTHREAD_MUTEX_INITIALIZER;

void *do_capture(void *conf)
{
  double freq;
  char *dbuf = NULL;
  int sock, chunk_index; 
  int64_t df_in_blk;
  conf_t *capture_conf = (conf_t *)conf;
  struct sockaddr_in sa = {0}, fromsa = {0};
  socklen_t fromlen = sizeof(fromsa);
  uint64_t df_in_period, seconds_from_epoch, tbuf_loc, cbuf_loc, writebuf, *ptr = NULL;
  register int nchunk = capture_conf->nchunk;
  register double chunk_index0 = capture_conf->chunk_index0;
  register double time_res_df = capture_conf->time_res_df;

  register int dfsz_seek = capture_conf->dfsz_seek;
  register int dfsz_keep = capture_conf->dfsz_keep;
  register uint64_t ndf_per_chunk_rbuf = capture_conf->ndf_per_chunk_rbuf;
  register uint64_t rbuf_ndf_per_chunk_tbuf = capture_conf->ndf_per_chunk_rbuf + capture_conf->ndf_per_chunk_tbuf;
  register int thread_index = capture_conf->thread_index;
  
  dbuf = (char *)malloc(sizeof(char) * DFSZ);
  log_add(capture_conf->log_file, "INFO", 1, log_mutex, "In funtion thread id = %ld, %d, %d", (long)pthread_self(), capture_conf->thread_index, thread_index);
  
  sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&capture_conf->tout, sizeof(capture_conf->tout));  
  memset(&sa, 0x00, sizeof(sa));
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(capture_conf->port_alive[thread_index]);
  sa.sin_addr.s_addr = inet_addr(capture_conf->ip_alive[thread_index]);
  
  if(bind(sock, (struct sockaddr *)&sa, sizeof(sa)) == -1)
    {
      log_add(capture_conf->log_file, "ERR", 1, log_mutex, "Can not bind to %s_%d, which happens at \"%s\", line [%d], has to abort",
		  inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
      fprintf(stderr, "Can not bind to %s_%d, which happens at \"%s\", line [%d], has to abort.\n",
	      inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
      
      quit = 2;
      close(sock);
      free(dbuf);      
      pthread_exit(NULL);
    }

  /* 
     If the reference time is in future, 
     we receive packets but drop them;
  */
  do{    
    if(recvfrom(sock, (void *)dbuf, DFSZ, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
      {
	log_add(capture_conf->log_file, "ERR", 1, log_mutex, "Can not receive data from %s_%d, which happens at \"%s\", line [%d], has to abort",
		    inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
	fprintf(stderr,	"Can not receive data from %s_%d, which happens at \"%s\", line [%d], has to abort.\n",
		inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
	
	quit = 2;
	close(sock);
	free(dbuf);
	pthread_exit(NULL);
      }

    ptr = (uint64_t*)dbuf;
    writebuf = bswap_64(*ptr);
    df_in_period = writebuf & 0x00000000ffffffff;
    seconds_from_epoch = (writebuf & 0x3fffffff00000000) >> 32;
        
    pthread_mutex_lock(&ref_mutex[thread_index]);
    df_in_blk = (int64_t)(df_in_period - df_in_period_ref[thread_index]) + ((double)seconds_from_epoch - (double)seconds_from_epoch_ref[thread_index]) / time_res_df;
    pthread_mutex_unlock(&ref_mutex[thread_index]);
  }while(df_in_blk<0);
  ndf_advance[thread_index] = df_in_blk;
  
  /* Tell other process (e.g., baseband2filterbank) that the current thread is ready */
  fprintf(stdout, "CAPTURE_READY\n"); 
  fflush(stdout);

  /* loop to capture packets */
  while(!quit)
    {
      if(recvfrom(sock, (void *)dbuf, DFSZ, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
      	{
	  log_add(capture_conf->log_file, "ERR", 1, log_mutex, "Can not receive data from %s_%d, which happens at \"%s\", line [%d], has to abort",
		      inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
	  fprintf(stderr, "Can not receive data from %s_%d, which happens at \"%s\", line [%d], has to abort.\n",
		  inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
	  
      	  quit = 2;
	  close(sock);
      	  free(dbuf);
      	  pthread_exit(NULL);
      	}

      ptr      = (uint64_t*)dbuf;
      writebuf = bswap_64(*ptr);
      df_in_period = writebuf & 0x00000000ffffffff;
      seconds_from_epoch = (writebuf & 0x3fffffff00000000) >> 32;
      writebuf = bswap_64(*(ptr + 2));
      freq     = (double)((writebuf & 0x00000000ffff0000) >> 16);
  
      chunk_index = (int)(freq/NCHAN_PER_CHUNK + chunk_index0);
      if (chunk_index<0 || chunk_index > (capture_conf->nchunk-1))
	{      
	  log_add(capture_conf->log_file, "ERR", 1, log_mutex, "Frequency chunk is outside the range [0 %d], which happens at \"%s\", line [%d], has to abort", capture_conf->nchunk, __FILE__, __LINE__);
	  fprintf(stderr, "Frequency chunk is outside the range [0 %d], which happens at \"%s\", line [%d], has to abort.\n",
		  capture_conf->nchunk, __FILE__, __LINE__);
	  
	  quit = 2;
	  close(sock);
	  free(dbuf);
	  pthread_exit(NULL);
	}
      
      pthread_mutex_lock(&ref_mutex[thread_index]);
      df_in_blk = (int64_t)(df_in_period - df_in_period_ref[thread_index]) + ((double)seconds_from_epoch - (double)seconds_from_epoch_ref[thread_index]) / time_res_df;
      pthread_mutex_unlock(&ref_mutex[thread_index]);

      if(df_in_blk>=0) // Only check the "current" packets
	{
	  if(df_in_blk < ndf_per_chunk_rbuf)  // packets belong to current ring buffer block
	    {
	      transit[thread_index] = 0;      // tell buffer control thread that current capture thread has updated reference time
	      cbuf_loc = (uint64_t)((df_in_blk * nchunk + chunk_index) * dfsz_keep);         // This is in TFTFP order
	      memcpy(cbuf + cbuf_loc, dbuf + dfsz_seek, dfsz_keep);
	      
	      pthread_mutex_lock(&ndf_mutex[thread_index]); 
	      ndf[thread_index]++; // Packet counter
	      pthread_mutex_unlock(&ndf_mutex[thread_index]);
	    }
	  else
	    {
	      log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Cross the boundary");

	      transit[thread_index] = 1; // tell buffer control thread that current capture thread is crossing the boundary
	      if(df_in_blk < rbuf_ndf_per_chunk_tbuf)
		{		  
		  tail[thread_index] = (uint64_t)((df_in_blk - ndf_per_chunk_rbuf) * nchunk + chunk_index); // This is in TFTFP order
		  tbuf_loc           = (uint64_t)(tail[thread_index] * (dfsz_keep + 1));
		  
		  tail[thread_index]++;  // have to ++, otherwise we will miss the last available data frame in tbuf;
		  
		  tbuf[tbuf_loc] = 'Y';
		  memcpy(tbuf + tbuf_loc + 1, dbuf + dfsz_seek, dfsz_keep);
		  
		  pthread_mutex_lock(&ndf_mutex[thread_index]);
		  ndf[thread_index]++;
		  pthread_mutex_unlock(&ndf_mutex[thread_index]);
		}
	      
	    }
	}
      else
	transit[thread_index] = 0;
    }
  log_add(capture_conf->log_file, "INFO", 1, log_mutex, "DONE the capture thread, which happens at \"%s\", line [%d]", __FILE__, __LINE__);
  
  /* Exit */
  quit = 1;
  
  free(dbuf);
  close(sock);
  pthread_exit(NULL);
}

int initialize_capture(conf_t *conf)
{
  int i, sock, nblk_behind = 0;
  time_t seconds_from_1970;

  struct sockaddr_in sa = {0}, fromsa = {0};
  socklen_t fromlen = sizeof(fromsa);
  int64_t df_in_blk = -1;
  uint64_t df_in_period, seconds_from_epoch, writebuf, *ptr = NULL;
  char *dbuf = NULL;
  time_t now;
  int beam_index;
  
  /* Unix socket for capture control*/
  if(conf->capture_ctrl)
    sprintf(conf->capture_ctrl_addr, "%s/capture.socket", conf->dir);  
   
  /* Init status */
  conf->nchunk       = 0;
  conf->nchunk_alive = 0;
  for(i = 0; i < conf->nport_alive; i++)
    {
      conf->nchunk        += conf->nchunk_alive_expect_on_port[i];
      conf->nchunk_alive  += conf->nchunk_alive_actual_on_port[i];
    }
  for(i = 0; i < conf->nport_dead; i++)
    conf->nchunk       += conf->nchunk_dead_on_port[i];
  if(conf->pad == 1)
    conf->nchunk = NCHUNK_FULL_BAND;
  if(conf->nchunk_alive == 0)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "None alive chunks, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "None alive chunks, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      fclose(conf->log_file);
      exit(EXIT_FAILURE);
    }
  
  conf->time_res_df  = (double)PERIOD/(double)NDF_PER_CHUNK_PER_PERIOD;
  conf->time_res_blk = conf->time_res_df * (double)conf->ndf_per_chunk_rbuf;
  conf->nchan        = conf->nchunk * NCHAN_PER_CHUNK;
  conf->chunk_index0 = -(conf->center_freq + 1.0)/(double)NCHAN_PER_CHUNK + 0.5 * conf->nchunk;

  log_add(conf->log_file, "INFO", 1, log_mutex, "time_res_df is %f", conf->time_res_df);
  log_add(conf->log_file, "INFO", 1, log_mutex, "time_res_blk is %f", conf->time_res_blk);
  log_add(conf->log_file, "INFO", 1, log_mutex, "nchan is %d", conf->nchan);
  log_add(conf->log_file, "INFO", 1, log_mutex, "chunk_index0 is %f", conf->chunk_index0);
  
  conf->seconds_from_1970 = floor(conf->time_res_df * conf->df_in_period) + conf->seconds_from_epoch + SECDAY * conf->days_from_1970;
  time(&now);
  if(abs(conf->seconds_from_1970 - now) > 10 * PERIOD)  // No plan to offset the reference time by 10 times of period
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "the reference information offset from current time by 10 times of period, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "the reference information offset from current time by 10 times of period, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      fclose(conf->log_file);
      exit(EXIT_FAILURE);      
    }
  conf->picoseconds_ref   = 1E6 * round(1.0E6 * (PERIOD - floor(conf->time_res_df * conf->df_in_period)));
  log_add(conf->log_file, "INFO", 1, log_mutex, "picoseconds_ref is %"PRIu64"", conf->picoseconds_ref);
  log_add(conf->log_file, "INFO", 1, log_mutex, "seconds_from_1970 is %lun", (unsigned long)conf->seconds_from_1970);
  
  conf->tout.tv_sec     = PERIOD;
  conf->tout.tv_usec    = 0;
  
  /* Create HDU and check the size of buffer bolck */
  conf->dfsz_keep  = DFSZ - conf->dfsz_seek;
  conf->blksz_rbuf = conf->nchunk * conf->dfsz_keep * conf->ndf_per_chunk_rbuf;  // The required ring buffer block size in byte;
  log_add(conf->log_file, "INFO", 1, log_mutex, "blksz_rbuf is %"PRIu64"", conf->blksz_rbuf);
  
  conf->hdu        = dada_hdu_create(NULL);
  dada_hdu_set_key(conf->hdu, conf->key);
  if(dada_hdu_connect(conf->hdu) < 0)
    { 
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      destroy_capture(*conf);
      log_close(conf->log_file);
      exit(EXIT_FAILURE);    
    }
  
  if(dada_hdu_lock_write(conf->hdu) < 0) // make ourselves the write client
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Error locking HDU, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      destroy_capture(*conf);
      log_close(conf->log_file);
      exit(EXIT_FAILURE);
    }
  conf->data_block   = (ipcbuf_t *)(conf->hdu->data_block);
  conf->header_block = (ipcbuf_t *)(conf->hdu->header_block);
  
  if(conf->blksz_rbuf != ipcbuf_get_bufsz(conf->data_block))  // Check the ring buffer blovk size
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Buffer size mismatch, %"PRIu64" vs %"PRIu64", which happens at \"%s\", line [%d], has to abort", conf->blksz_rbuf, ipcbuf_get_bufsz(conf->data_block), __FILE__, __LINE__);
      fprintf(stderr, "Buffer size mismatch, %"PRIu64" vs %"PRIu64", which happens at \"%s\", line [%d], has to abort.\n", conf->blksz_rbuf, ipcbuf_get_bufsz(conf->data_block), __FILE__, __LINE__);
      
      destroy_capture(*conf);
      log_close(conf->log_file);
      exit(EXIT_FAILURE);    
    }

  if(conf->capture_ctrl)
    {
      if(ipcbuf_disable_sod(conf->data_block) < 0)
	{
	  log_add(conf->log_file, "ERR", 1, log_mutex, "Can not write data before start, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  	 
	  destroy_capture(*conf);
	  log_close(conf->log_file);	  
	  exit(EXIT_FAILURE);
	}
    }
  else  // For the case which enables the SOD at the beginning
    {
      seconds_from_1970 = conf->seconds_from_1970;
      conf->picoseconds = conf->picoseconds_ref;
      conf->mjd_start   = seconds_from_1970 / SECDAY + MJD1970;                       // Float MJD start time without fraction second
      strftime (conf->utc_start, MSTR_LEN, DADA_TIMESTR, gmtime(&seconds_from_1970)); // String start time without fraction second 
      register_dada_header(*conf);
    }
  conf->tbufsz = (conf->dfsz_keep + 1) * conf->ndf_per_chunk_tbuf * conf->nchunk;
  log_add(conf->log_file, "INFO", 1, log_mutex, "tbufsz is %"PRIu64"", conf->tbufsz);
  tbuf = (char *)malloc(conf->tbufsz * sizeof(char));// init temp buffer
  
  /* 
     Get the first ring buffer block ready for capture, 
     to catch up if the reference time is before current time,
     write empty buffer blocks if we need to catch up;
     Also check the beam_index and seconds_from_epoch here;
  */
  dbuf = (char *)malloc(sizeof(char) * DFSZ);  
  sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&conf->tout, sizeof(conf->tout));  
  memset(&sa, 0x00, sizeof(sa));
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(conf->port_alive[0]);
  sa.sin_addr.s_addr = inet_addr(conf->ip_alive[0]);

  if(bind(sock, (struct sockaddr *)&sa, sizeof(sa)) == -1)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not bind to %s_%d, which happens at \"%s\", line [%d], has to abort", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
      fprintf(stderr, "Can not bind to %s_%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
      
      free(dbuf);
      close(sock);
      destroy_capture(*conf);
      log_close(conf->log_file);
      exit(EXIT_FAILURE);
    }
  
  if(recvfrom(sock, (void *)dbuf, DFSZ, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "Can not receive data from %s_%d, which happens at \"%s\", line [%d], has to abort", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
      fprintf(stderr, "Can not receive data from %s_%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);

      free(dbuf);
      close(sock);
      destroy_capture(*conf);
      log_close(conf->log_file);
      exit(EXIT_FAILURE);
    }
  
  ptr = (uint64_t*)dbuf;
  writebuf = bswap_64(*ptr);
  df_in_period = writebuf & 0x00000000ffffffff;
  seconds_from_epoch = (writebuf & 0x3fffffff00000000) >> 32;
  writebuf = bswap_64(*(ptr + 2));
  beam_index = writebuf & 0x000000000000ffff;
  if(beam_index != conf->beam_index) // Check the beam index
    {      
      log_add(conf->log_file, "ERR", 1, log_mutex, "beam_index here is %d, but the input is %d, which happens at \"%s\", line [%d], has to abort", beam_index, conf->beam_index, __FILE__, __LINE__);
      fprintf(stderr, "beam_index here is %d, but the input is %d, which happens at \"%s\", line [%d], has to abort\n", beam_index, conf->beam_index, __FILE__, __LINE__);
      
      free(dbuf);
      close(sock);
      destroy_capture(*conf);
      log_close(conf->log_file);
      exit(EXIT_FAILURE);
    }
  if((int)((double)seconds_from_epoch - (double)conf->seconds_from_epoch) % PERIOD) // The difference in seconds should be multiple times of period    
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "the difference in seconds is not multiple times of period, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "the difference in seconds is not multiple times of period, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      free(dbuf);
      close(sock);
      destroy_capture(*conf);
      log_close(conf->log_file);
      exit(EXIT_FAILURE);
    }
  
  df_in_blk = (int64_t)(df_in_period - conf->df_in_period) + ((double)seconds_from_epoch - (double)conf->seconds_from_epoch) / conf->time_res_df;
  if(df_in_blk>0) // If the reference time is before the current time;
    {
      nblk_behind = (int)floor(df_in_blk/(double)conf->ndf_per_chunk_rbuf);
      for(i = 0; i < nblk_behind; i++)
	{      
	  cbuf = ipcbuf_get_next_write(conf->data_block); // Open a ring buffer block
	  if(cbuf == NULL)
	    {
	      log_add(conf->log_file, "ERR", 1, log_mutex, "open_buffer failed, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	      fprintf(stderr, "open_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      
	      free(dbuf);
	      close(sock);
	      destroy_capture(*conf);
	      log_close(conf->log_file);
	      exit(EXIT_FAILURE);
	    }
	  
	  if(ipcbuf_mark_filled(conf->data_block, conf->blksz_rbuf) < 0) // write nothing to it
	    {
	      log_add(conf->log_file, "ERR", 1, log_mutex, "close_buffer failed, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	      fprintf(stderr, "close_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      
	      free(dbuf);
	      close(sock);
	      destroy_capture(*conf);
	      log_close(conf->log_file);
	      exit(EXIT_FAILURE);
	    }
	  
	  conf->df_in_period += conf->ndf_per_chunk_rbuf;
	  if(conf->df_in_period >= NDF_PER_CHUNK_PER_PERIOD)
	    {
	      conf->seconds_from_epoch  += PERIOD;
	      conf->df_in_period        -= NDF_PER_CHUNK_PER_PERIOD;
	    }
	}
    }
  free(dbuf);
  close(sock);
  log_add(conf->log_file, "INFO", 1, log_mutex, "nblk_behind is %d", nblk_behind);
  log_add(conf->log_file, "INFO", 1, log_mutex, "reference info is %"PRIu64"\t%"PRIu64"", conf->seconds_from_epoch, conf->df_in_period);
  
  cbuf = ipcbuf_get_next_write(conf->data_block);
  if(cbuf == NULL)
    {
      log_add(conf->log_file, "ERR", 1, log_mutex, "open_buffer failed, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "open_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      log_close(conf->log_file);
      destroy_capture(*conf);
      exit(EXIT_FAILURE);
    }

  /* finally setup the reference information for each thread */
  for(i = 0; i < conf->nport_alive; i++)
    {
      seconds_from_epoch_ref[i] = conf->seconds_from_epoch;
      df_in_period_ref[i]       = conf->df_in_period;
    }
  
  return EXIT_SUCCESS;
}

int destroy_capture(conf_t conf)
{
  int i;
  
  for(i = 0; i < MPORT_CAPTURE; i++)
    {
      pthread_mutex_destroy(&ref_mutex[i]);
      pthread_mutex_destroy(&ndf_mutex[i]);
    }
  pthread_mutex_destroy(&log_mutex);

  if(conf.data_block)
    dada_hdu_unlock_write(conf.hdu);
  dada_hdu_destroy(conf.hdu); // it has disconnect
  
  return EXIT_SUCCESS;
}

int register_dada_header(conf_t conf)
{
  char *hdrbuf = NULL;
  
  /* Register header */
  hdrbuf = ipcbuf_get_next_write(conf.header_block);
  if(!hdrbuf)
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  if(!conf.dada_header_template)
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Please specify header file, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Please specify header file, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }  
  if(fileread(conf.dada_header_template, hdrbuf, DADA_HDRSZ) < 0)
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Error reading header file, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error reading header file, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  
  /* Setup DADA header with given values */
  if(ascii_header_set(hdrbuf, "UTC_START", "%s", conf.utc_start) < 0)  
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Error setting UTC_START, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error setting UTC_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  
  if(ascii_header_set(hdrbuf, "RA", "%s", conf.ra) < 0)  
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Error setting RA, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error setting RA, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  
  if(ascii_header_set(hdrbuf, "DEC", "%s", conf.dec) < 0)  
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Error setting DEC, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error setting DEC, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  
  if(ascii_header_set(hdrbuf, "SOURCE", "%s", conf.source) < 0)  
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Error setting SOURCE, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error setting SOURCE, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);      
      exit(EXIT_FAILURE);
    }
  
  if(ascii_header_set(hdrbuf, "INSTRUMENT", "%s", conf.instrument_name) < 0)  
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Error setting INSTRUMENT, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error setting INSTRUMENT, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);     
      exit(EXIT_FAILURE);
    }
  
  if(ascii_header_set(hdrbuf, "PICOSECONDS", "%"PRIu64, conf.picoseconds) < 0)  
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Error setting PICOSECONDS, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error setting PICOSECONDS, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }    
  if(ascii_header_set(hdrbuf, "FREQ", "%.6lf", conf.center_freq) < 0)
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Error setting FREQ, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error setting FREQ, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  if(ascii_header_set(hdrbuf, "MJD_START", "%.10lf", conf.mjd_start) < 0)
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Error setting MJD_START, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error setting MJD_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  if(ascii_header_set(hdrbuf, "NCHAN", "%d", conf.nchan) < 0)
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Error setting NCHAN, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error setting NCHAN, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  
  if(ascii_header_set(hdrbuf, "BEAM_ID", "%d", conf.beam_index) < 0)
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Error setting BEAM_ID, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error setting BEAM_ID, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }  
  if(ascii_header_get(hdrbuf, "RESOLUTION", "%lf", &conf.freq_res) < 0)
    {
      log_add(conf.log_file, "ERR", 1,  log_mutex, "Error getting RESOLUTION, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error getting RESOLUTION, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  conf.bandwidth = conf.freq_res * conf.nchan;
  if(ascii_header_set(hdrbuf, "BW", "%.6lf", conf.bandwidth) < 0)
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Error setting BW, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error setting BW, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  /* donot set header parameters anymore - acqn. doesn't start */
  if(ipcbuf_mark_filled(conf.header_block, DADA_HDRSZ) < 0)
    {
      log_add(conf.log_file, "ERR", 1, log_mutex, "Error header_fill, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error header_fill, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_capture(conf);
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }

  return EXIT_SUCCESS;	      
}


int threads(conf_t *conf)
{
  int i, ret[MPORT_CAPTURE + 2], nthread;
  pthread_t thread[MPORT_CAPTURE + 2];
  pthread_attr_t attr;
  cpu_set_t cpus;
  int nport_alive = conf->nport_alive;
  conf_t thread_conf[MPORT_CAPTURE];
    
  for(i = 0; i < nport_alive; i++)
    /* 
       Create threads. Capture threads and ring buffer control thread are essential, the capture control thread is created when it is required to;
       If we create a capture control thread, we can control the start and end of data during runtime, the header of DADA buffer will be setup each time we start the data, which means without rerun the pipeline, we can get multiple capture runs;
       If we do not create a capture thread, the data will start at the begining and we need to setup the header at that time, we can only do one capture without rerun the pipeline;
    */
    {
      thread_conf[i] = *conf;
      thread_conf[i].thread_index = i;
      
      if(!(conf->cpu_bind == 0))
	{
	  pthread_attr_init(&attr);  
	  CPU_ZERO(&cpus);
	  CPU_SET(conf->capture_cpu[i], &cpus);
	  pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
	  ret[i] = pthread_create(&thread[i], &attr, do_capture, (void *)&thread_conf[i]);
	  pthread_attr_destroy(&attr);
	}
      else
	ret[i] = pthread_create(&thread[i], &attr, do_capture, (void *)&thread_conf[i]);
    }

  if(!(conf->cpu_bind == 0)) 
    {            
      pthread_attr_init(&attr);
      CPU_ZERO(&cpus);
      CPU_SET(conf->rbuf_ctrl_cpu, &cpus);      
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);	
      ret[nport_alive] = pthread_create(&thread[nport_alive], &attr, buf_control, (void *)conf);
      pthread_attr_destroy(&attr);

      if(conf->capture_ctrl)
	{
	  pthread_attr_init(&attr);
	  CPU_ZERO(&cpus);
	  CPU_SET(conf->capture_ctrl_cpu, &cpus);      
	  pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);	
	  ret[nport_alive + 1] = pthread_create(&thread[nport_alive + 1], &attr, capture_control, (void *)conf);
	  pthread_attr_destroy(&attr);
	}
    }
  else
    {
      ret[nport_alive] = pthread_create(&thread[nport_alive], NULL, buf_control, (void *)conf);
      if(conf->capture_ctrl)
	ret[nport_alive + 1] = pthread_create(&thread[nport_alive + 1], NULL, capture_control, (void *)conf);
    }

  if (conf->capture_ctrl)
    nthread = nport_alive + 2;
  else
    nthread = nport_alive + 1;

  log_add(conf->log_file, "INFO", 1, log_mutex, "Join threads? Before it");
  for(i = 0; i < nthread; i++)   // Join threads and unbind cpus
    pthread_join(thread[i], NULL);

  log_add(conf->log_file, "INFO", 1, log_mutex, "Join threads? The last quit is %d", quit);
    
  return EXIT_SUCCESS;
}

void *buf_control(void *conf)
{
  conf_t *capture_conf = (conf_t *)conf;
  int i, chunk_index, idf, transited = 0, nchunk = capture_conf->nchunk;
  uint64_t cbuf_loc, tbuf_loc, ntail, rbuf_nblk = 0;
  uint64_t ndf_actual = 0, ndf_expect = 0;
  uint64_t ndf_blk_actual = 0, ndf_blk_expect = 0;
  double sleep_time = 0.5 * capture_conf->time_res_blk;
  unsigned int sleep_sec = (int)sleep_time;
  useconds_t sleep_usec  = 1.0E6 * (sleep_time - sleep_sec);
  
  while(!quit)
    {
      /*
	To see if we need to move to next buffer block or quit 
	If all ports have packet arrive after current ring buffer block, start to change the block 
      */
      while((!transited) && (!quit))
	{
	  transited = transit[0];
	  for(i = 1; i < capture_conf->nport_alive; i++) // When all ports are on the transit status
	    //transited = transited && transit[i]; // all happen, take action
	    transited = transited || transit[i]; // one happens, take action
	}
      
      if(quit == 1)
	{
	  log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Quit just after the buffer transit state change");
	  pthread_exit(NULL);
	}
      if(quit == 2)
	{
	  log_add(capture_conf->log_file, "ERR", 1, log_mutex, "Quit just after the buffer transit state change, which happens at \"%s\", line [%d]", __FILE__, __LINE__);
	  fprintf(stderr, "Quit with ERROR just after the buffer transit state change, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  
	  pthread_exit(NULL);
	}
      log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Just after buffer transit state change");
      
      /* Check the traffic of previous buffer cycle */
      rbuf_nblk ++; // From the time we actually put data into ring buffer
      ndf_blk_expect = 0;
      ndf_blk_actual = 0;
      for(i = 0; i < capture_conf->nport_alive; i++)
	{
	  pthread_mutex_lock(&ndf_mutex[i]); 
	  ndf_blk_actual += ndf[i];
	  ndf[i] = 0; 
	  pthread_mutex_unlock(&ndf_mutex[i]);
	}
      ndf_actual += ndf_blk_actual;
      if(rbuf_nblk==1)
	{
	  for(i = 0; i < capture_conf->nport_alive; i++)
	    ndf_blk_expect += (capture_conf->ndf_per_chunk_rbuf - ndf_advance[i]) * capture_conf->nchunk_alive_actual_on_port[i];
	}
      else
	ndf_blk_expect += capture_conf->ndf_per_chunk_rbuf * capture_conf->nchunk_alive; // Only for current buffer
      ndf_expect += ndf_blk_expect;

      log_add(capture_conf->log_file, "INFO", 1, log_mutex, "%s starts from port %d, packet loss rate %d %f %E %E", capture_conf->ip_alive[0], capture_conf->port_alive[0], rbuf_nblk * capture_conf->time_res_blk, (1.0 - ndf_actual/(double)ndf_expect), (1.0 - ndf_blk_actual/(double)ndf_blk_expect));
      log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Packets counters, %"PRIu64" %"PRIu64" %"PRIu64" %"PRIu64"", ndf_actual, ndf_expect, ndf_blk_actual, ndf_blk_expect);
      
      fprintf(stdout, "CAPTURE_STATUS %d %f %E %E\n", capture_conf->process_index, rbuf_nblk * capture_conf->time_res_blk, (1.0 - ndf_actual/(double)ndf_expect), (1.0 - ndf_blk_actual/(double)ndf_blk_expect)); // Pass the status to stdout
      fflush(stdout);
      log_add(capture_conf->log_file, "INFO", 1, log_mutex, "After fflush stdout");
      log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Before mark filled");
      
      /* Close current buffer */
      if(ipcbuf_mark_filled(capture_conf->data_block, capture_conf->blksz_rbuf) < 0)
	{
	  log_add(capture_conf->log_file, "ERR", 1, log_mutex, "close_buffer failed, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  fprintf(stderr, "close_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  
	  quit = 2;
	  pthread_exit(NULL);
	}
      log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Mark filled done");
      
      /*
	To see if the buffer is full, quit if yes.
	If we have a reader, there will be at least one buffer which is not full
      */
      if(ipcbuf_get_nfull(capture_conf->data_block) >= (ipcbuf_get_nbufs(capture_conf->data_block) - 1)) 
	{
	  log_add(capture_conf->log_file, "ERR", 1, log_mutex, "buffers are all full, which happens at \"%s\", line [%d], has to abort.", __FILE__, __LINE__);
	  fprintf(stderr, "buffers are all full, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  	  
	  quit = 2;
	  pthread_exit(NULL);
	}
      log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Available buffer block check done");
      
      /* Get new buffer block */
      cbuf = ipcbuf_get_next_write(capture_conf->data_block);
      log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Get next write done");
      if(cbuf == NULL)
	{
	  log_add(capture_conf->log_file, "ERR", 1, log_mutex, "open_buffer failed, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  fprintf(stderr, "open_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  
	  quit = 2;
	  pthread_exit(NULL);
	}
      
      /* Update reference point */
      for(i = 0; i < capture_conf->nport_alive; i++)
	{
	  // Update the reference hdr, once capture thread get the updated reference, the data will go to the next block or be dropped;
	  // We have to put a lock here as partial update of reference hdr will be a trouble to other threads;
	  
	  pthread_mutex_lock(&ref_mutex[i]);
	  log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Start to change the reference information %"PRIu64" %"PRIu64"", seconds_from_epoch_ref[i], df_in_period_ref[i]);
	  df_in_period_ref[i] += capture_conf->ndf_per_chunk_rbuf;
	  if(df_in_period_ref[i] >= NDF_PER_CHUNK_PER_PERIOD)
	    {
	      seconds_from_epoch_ref[i]  += PERIOD;
	      df_in_period_ref[i] -= NDF_PER_CHUNK_PER_PERIOD;
	    }
	  log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Finish the change of reference information %"PRIu64" %"PRIu64"", seconds_from_epoch_ref[i], df_in_period_ref[i]);
	  pthread_mutex_unlock(&ref_mutex[i]);
	}
      
      /* To see if we need to copy data from temp buffer into ring buffer */
      log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Just before the second transit state change");
      while(transited && (!quit))
	{
	  transited = transit[0];
	  for(i = 1; i < capture_conf->nport_alive; i++)
	    //transited = transited || transit[i]; // all happen, take action
	    transited = transited && transit[i]; // one happens, take action
	}      
      if(quit == 1)
	{
	  log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Quit just after the second transit state change");
	  pthread_exit(NULL);
	}    
      if(quit == 2)
	{
	  log_add(capture_conf->log_file, "ERR", 1, log_mutex, "Quit just after the second transit state change, which happens at \"%s\", line [%d]", __FILE__, __LINE__);
	  fprintf(stderr, "Quit with ERROR just after the second transit state change, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  pthread_exit(NULL);
	}

      ntail = 0;
      for(i = 0; i < capture_conf->nport_alive; i++)
	ntail = (tail[i] > ntail) ? tail[i] : ntail;
      
      log_add(capture_conf->log_file, "INFO", 1, log_mutex, "The location of the last packet in temp buffer is %"PRIu64"", ntail);
      for(i = 0; i < ntail; i++)
	{
	  tbuf_loc = (uint64_t)(i * (capture_conf->dfsz_keep + 1));	      
	  if(tbuf[tbuf_loc] == 'Y')
	    {		  
	      cbuf_loc = (uint64_t)(i * capture_conf->dfsz_keep);  // This is for the TFTFP order temp buffer copy;	      
	      memcpy(cbuf + cbuf_loc, tbuf + tbuf_loc + 1, capture_conf->dfsz_keep);		  
	      tbuf[tbuf_loc + 1] = 'N';  // Make sure that we do not copy the data later;
	      // If we do not do that, we may have too many data frames to copy later
	    }
	}
      for(i = 0; i < capture_conf->nport_alive; i++)
	tail[i] = 0;  // Reset the location of tbuf;
      
      sleep(sleep_sec);
      usleep(sleep_usec);
    }
  
  /* Exit */
  if(ipcbuf_mark_filled(capture_conf->data_block, capture_conf->blksz_rbuf) < 0)
    {
      log_add(capture_conf->log_file, "ERR", 1, log_mutex, "ipcio_close_block failed, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "ipcio_close_block failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      quit = 2;
      pthread_exit(NULL);
    }
  
  log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Normale quit of buffer control thread");
  pthread_exit(NULL);
}


void *capture_control(void *conf)
{  
  int sock, i;
  struct sockaddr_un sa = {0}, fromsa = {0};
  socklen_t fromlen = sizeof(fromsa);
  conf_t *capture_conf = (conf_t *)conf;
  char capture_control_command[MSTR_LEN], capture_control_keyword[MSTR_LEN];
  int64_t start_buf;
  int64_t available_buf;
  double seconds_offset; // Offset from the reference time;
  uint64_t picoseconds_offset; // The seconds_offset fraction part in picoseconds
  int msg_len;
  time_t seconds_from_1970;
  
  /* Create an unix socket for control */
  if((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1)
    {
      log_add(capture_conf->log_file, "ERR", 1, log_mutex, "Error setting NCHAN, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "Error setting NCHAN, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      quit = 1;
      pthread_exit(NULL);
    }
  
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&capture_conf->tout, sizeof(capture_conf->tout));
  memset(&sa, 0, sizeof(struct sockaddr_un));
  sa.sun_family = AF_UNIX;
  strncpy(sa.sun_path, capture_conf->capture_ctrl_addr, strlen(capture_conf->capture_ctrl_addr));
  unlink(capture_conf->capture_ctrl_addr);
  
  log_add(capture_conf->log_file, "INFO", 1, log_mutex, capture_conf->capture_ctrl_addr);
  
  if(bind(sock, (struct sockaddr*)&sa, sizeof(sa)) == -1)
    {
      log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Can not bind to file socket, which happens at \"%s\", line [%d]", __FILE__, __LINE__);
       
      quit = 1;
      close(sock);
      pthread_exit(NULL);
 
    }

  fprintf(stdout, "CAPTURE_READY\n"); // Tell other process that the thread is ready
  fflush(stdout);
  
  while(!quit)
    {
      msg_len = 0;
      while(!(msg_len > 0) && !quit)
	{
	  memset(capture_control_command, 0, MSTR_LEN);
	  msg_len = recvfrom(sock, (void *)capture_control_command, MSTR_LEN, 0, (struct sockaddr*)&fromsa, &fromlen);
	}
      
      if(quit == 1)
	{
	  log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Quit just after checking the unix socket");
	  close(sock);
	  pthread_exit(NULL);
	}
      if(quit == 2)
	{
	  log_add(capture_conf->log_file, "ERR", 1, log_mutex, "Quit just after checking the unix socket, which happens at \"%s\", line [%d]", __FILE__, __LINE__);
	  fprintf(stderr, "Quit with ERROR just after checking the unix socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	  close(sock);
	  pthread_exit(NULL);
	}
      
      if(strstr(capture_control_command, "END-OF-CAPTURE") != NULL)
	{
	  log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Got END-OF-CAPTURE signal, has to quit");      
	  
	  quit = 1;	      
	  if(ipcbuf_is_writing(capture_conf->data_block))
	    ipcbuf_enable_eod(capture_conf->data_block);
	  log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Got END-OF-CAPTURE signal, after ENABLE_EOD");      
	  
	  close(sock);
	  pthread_exit(NULL);
	}  
      if(strstr(capture_control_command, "END-OF-DATA") != NULL)
	{
	  log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Got END-OF-DATA signal, has to enable eod");
	  ipcbuf_enable_eod(capture_conf->data_block);
	  memset(capture_control_command, 0, MSTR_LEN);
	}
	  
      if(strstr(capture_control_command, "START-OF-DATA") != NULL)
	{
	  log_add(capture_conf->log_file, "INFO", 1, log_mutex, "Got START-OF-DATA signal, has to enable sod");
	  
	  sscanf(capture_control_command, "%[^_]_%[^_]_%[^_]_%[^_]_%"SCNd64"", capture_control_keyword, capture_conf->source, capture_conf->ra, capture_conf->dec, &start_buf); // Read the start buffer from socket or get the minimum number from the buffer, we keep starting at the begining of buffer block;
	  available_buf = ipcbuf_get_write_count(capture_conf->data_block);
	  start_buf = (start_buf > available_buf) ? start_buf : available_buf; // To make sure the start buffer is valuable, to get the most recent buffer
	  log_add(capture_conf->log_file, "INFO", 1, log_mutex, "The data is enabled at %"PRIu64" buffer block", start_buf);
	  ipcbuf_enable_sod(capture_conf->data_block, (uint64_t)start_buf, 0);
	  
	  /* To get time stamp for current header */
	  seconds_offset = start_buf * capture_conf->time_res_blk; // Only work with buffer number
	  picoseconds_offset = 1E6 * (round(1.0E6 * (seconds_offset - floor(seconds_offset))));
	  
	  capture_conf->picoseconds = picoseconds_offset +capture_conf->picoseconds_ref;
	  seconds_from_1970 = capture_conf->seconds_from_1970 + floor(seconds_offset);
	  if(!(capture_conf->picoseconds < 1E12))
	    {
	      seconds_from_1970 += 1;
	      capture_conf->picoseconds -= 1E12;
	    }
	  strftime (capture_conf->utc_start, MSTR_LEN, DADA_TIMESTR, gmtime(&seconds_from_1970)); // String start time without fraction second 
	  capture_conf->mjd_start = seconds_from_1970 / SECDAY + MJD1970;                         // Float MJD start time without fraction second
	  
	  /*
	    To see if the buffer is full, quit if yes.
	    If we have a reader, there will be at least one buffer which is not full
	  */
	  if(ipcbuf_get_nfull(capture_conf->header_block) >= (ipcbuf_get_nbufs(capture_conf->header_block) - 1)) 
	    {
	      log_add(capture_conf->log_file, "ERR", 1, log_mutex, "buffers are all full, has to abort, which happens at \"%s\", line [%d]", __FILE__, __LINE__);
	      fprintf(stderr, "buffers are all full, has to abort, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      
	      quit = 2;
	      close(sock);
	      pthread_exit(NULL);
	    }
	  
	  /* setup dada header here */
	  if(register_dada_header(*capture_conf))
	    {		  
	      quit = 1;
	      close(sock);
	      pthread_exit(NULL);
	    }
	  memset(capture_control_command, 0, MSTR_LEN);
	}
    }
  
  quit = 1;
  close(sock);
  pthread_exit(NULL);
}

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
  log_add(conf.log_file, "INFO", 1, log_mutex, "The command line is \"%s\"", command);
  
  if((conf.process_index<0) || (conf.process_index>=NPROCESS_PER_NODE_MAX))
    {
      fprintf(stderr, "The process index is %d, which is not in range [0 %d), happens at \"%s\", line [%d], has to abort.\n", conf.process_index, NPROCESS_PER_NODE_MAX - 1, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "The process index is %d, which is not in range [0 %d), happens at \"%s\", line [%d], has to abort.", conf.process_index, NPROCESS_PER_NODE_MAX - 1, __FILE__, __LINE__);
      
      fclose(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "We capture data with process %d", conf.process_index);
  
  if((conf.beam_index<0) || (conf.beam_index>=NBEAM_MAX)) // More careful check later
    {
      fprintf(stderr, "The beam index is %d, which is not in range [0 %d), happens at \"%s\", line [%d], has to abort.\n", conf.beam_index, NBEAM_MAX - 1, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "The beam index is %d, which is not in range [0 %d), happens at \"%s\", line [%d], has to abort.", conf.beam_index, NBEAM_MAX - 1, __FILE__, __LINE__);
      
      fclose(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "We capture data with beam %d", conf.beam_index);
  
  log_add(conf.log_file, "INFO", 1, log_mutex, "Hexadecimal shared memory key for capture is %x", conf.key); // How to check this?
  
  if((conf.dfsz_seek != 0) && (conf.dfsz_seek != DF_HDRSZ)) // The seek bytes has to be 0 or DF_HDRSZ
    {
      fprintf(stderr, "The start point of packet is %d, but it should be 0 or %d, happens at \"%s\", line [%d], has to abort.\n", conf.dfsz_seek, DF_HDRSZ, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "The start point of packet is %d, but it should be 0 or %d, happens at \"%s\", line [%d], has to abort.", conf.dfsz_seek, DF_HDRSZ, __FILE__, __LINE__);
      
      fclose(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "Start point of packet is %d", conf.dfsz_seek);

  if(conf.nport_alive == 0)  // Can not work without alive ports
    {
      fprintf(stderr, "We do not have alive port, happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "We do not have alive port, happens at \"%s\", line [%d], has to abort.", __FILE__, __LINE__);
      
      fclose(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "We have %d alive ports, which are:", conf.nport_alive);
  for(i = 0; i < conf.nport_alive; i++)
    log_add(conf.log_file, "INFO", 1, log_mutex, "    ip %s, port %d, expected frequency chunks %d and actual frequency chunks %d", conf.ip_alive[i], conf.port_alive[i], conf.nchunk_alive_expect_on_port[i], conf.nchunk_alive_actual_on_port[i]);

  if(conf.nport_dead == 0)  // Does not matter how many dead ports we have
    log_add(conf.log_file, "INFO", 1, log_mutex, "We do not have dead ports");
  else
    {
      log_add(conf.log_file, "WARN", 1, log_mutex, "We have %d dead ports, which are:", conf.nport_dead);
      for(i = 0; i < conf.nport_dead; i++)
	log_add(conf.log_file, "WARN", 1, log_mutex, "    ip %s, port %d, expected frequency chunks %d", conf.ip_dead[i], conf.port_dead[i], conf.nchunk_dead_on_port[i]);
    }

  if(conf.center_freq > BAND_LIMIT_UP || conf.center_freq < BAND_LIMIT_DOWN)
    // The reference information has to be reasonable, later more careful check
    {
      fprintf(stderr, "center_freq %f is not in (%f %f), happens at \"%s\", line [%d], has to abort.\n",conf.center_freq, BAND_LIMIT_DOWN, BAND_LIMIT_UP, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "center_freq %f is not in (%f %f), happens at \"%s\", line [%d], has to abort.",conf.center_freq, BAND_LIMIT_DOWN, BAND_LIMIT_UP, __FILE__, __LINE__);
      
      fclose(conf.log_file);
      exit(EXIT_FAILURE);
    }  
  log_add(conf.log_file, "INFO", 1, log_mutex, "The center frequency for the capture is %f MHz", conf.center_freq); 

  if((conf.days_from_1970 <= 0) || (conf.seconds_from_epoch <= -1) || (conf.df_in_period < 0) || (conf.df_in_period >= NDF_PER_CHUNK_PER_PERIOD))
    // The reference information has to be reasonable, later more careful check
    {
      fprintf(stderr, "The reference information is not right, days_from_1970 is %d, seconds_from_epoch is %"PRId64" and df_in_period is %"PRId64", happens at \"%s\", line [%d], has to abort.\n", conf.days_from_1970, conf.seconds_from_epoch, conf.df_in_period, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "The reference information is not right, days_from_1970 is %d, seconds_from_epoch is %"PRId64" and df_in_period is %"PRId64", happens at \"%s\", line [%d], has to abort.", conf.days_from_1970, conf.seconds_from_epoch, conf.df_in_period, __FILE__, __LINE__);
      
      fclose(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "The reference information for the capture is: epoch %d, seconds %"PRId64" and location of packet in the period %"PRId64"", conf.days_from_1970, conf.seconds_from_epoch, conf.df_in_period);

  log_add(conf.log_file, "INFO", 1, log_mutex, "The runtime information is %s", conf.dir); // This has already been checked before

  if(conf.cpu_bind)  
    {      
      log_add(conf.log_file, "INFO", 1, log_mutex, "Buffer control thread runs on CPU %d", conf.rbuf_ctrl_cpu);
      
      for(i = 0; i < conf.nport_alive; i++) // To check the bind information for capture threads
	{	  
	  if(((conf.capture_cpu[i] == conf.capture_ctrl_cpu)?0:1) == 1)
	    break;
	  if(((conf.capture_cpu[i] == conf.rbuf_ctrl_cpu)?0:1) == 1)
	    break;
	}
      if(i == conf.nport_alive)  // We can not bind all capture threads to one cpu
	{
	  log_add(conf.log_file, "ERR", 1, log_mutex, "We can not bind all \"capture\" threads into one single CPU, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  fprintf(stderr, "We can not bind all threads into one single CPU, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  
	  log_close(conf.log_file);	  
	  exit(EXIT_FAILURE);
	}      
              
      if(conf.capture_ctrl)
	log_add(conf.log_file, "INFO", 1, log_mutex, "We will NOT enable sod at the beginning, Capture control thread runs on CPU %d", conf.capture_ctrl_cpu);
      else
	{      
	  if((conf.source == "unset") && ((conf.ra == "unset") || (conf.dec == "unset")))
	    {
	      fprintf(stderr, "We have to setup target information if we do not enable capture control, which happens at \"%s\", line [%d], has to abort\n", __FILE__, __LINE__);
	      log_add(conf.log_file, "ERR", 1, log_mutex, "We have to setup target information if we do not enable capture control, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	      
	      log_close(conf.log_file);
	      exit(EXIT_FAILURE);
	    }      
	  log_add(conf.log_file, "WARN", 1, log_mutex, "We will enable sod at the beginning");
	  log_add(conf.log_file, "INFO", 1, log_mutex, "The source name is %s, RA is %s and DEC is %s", conf.source, conf.ra, conf.dec);
	}  
    }
  else
    log_add(conf.log_file, "WARN", 1, log_mutex, "We will NOT bind threads to CPUs");

  if(conf.ndf_per_chunk_rbuf<=0)  // The actual size of it will be checked later
    {      
      fprintf(stderr, "ndf_per_chunk_rbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_rbuf, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "ndf_per_chunk_rbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort", conf.ndf_per_chunk_rbuf, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "Each ring buffer block has %"PRIu64" packets per frequency chunk", conf.ndf_per_chunk_rbuf);
  
  if(conf.ndf_per_chunk_tbuf<=0)  // The actual size of it will be checked later
    {      
      fprintf(stderr, "ndf_per_chunk_tbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort\n", conf.ndf_per_chunk_tbuf, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "ndf_per_chunk_tbuf shoule be a positive number, but it is %"PRIu64", which happens at \"%s\", line [%d], has to abort", conf.ndf_per_chunk_tbuf, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  log_add(conf.log_file, "INFO", 1, log_mutex, "Each temp buffer has %"PRIu64" packets per frequency chunk", conf.ndf_per_chunk_tbuf);

  if(access(conf.dada_header_template, F_OK ) != -1 )
    log_add(conf.log_file, "INFO", 1, log_mutex, "The name of header template of PSRDADA is %s", conf.dada_header_template);
  else
    {        
      fprintf(stderr, "dada_header_template %s is not exist, which happens at \"%s\", line [%d], has to abort\n", conf.dada_header_template, __FILE__, __LINE__);
      log_add(conf.log_file, "ERR", 1, log_mutex, "dada_header_template %s is not exist, which happens at \"%s\", line [%d], has to abort", conf.dada_header_template, __FILE__, __LINE__);
      
      log_close(conf.log_file);
      exit(EXIT_FAILURE);
    }
  
  log_add(conf.log_file, "INFO", 1, log_mutex, "The name of instrument is %s", conf.instrument_name); // Instrument name is not critical
  
  if(conf.pad)
    log_add(conf.log_file, "INFO", 1, log_mutex, "We will pad frequency chunks to fake full bandwidth data");
  else
    log_add(conf.log_file, "INFO", 1, log_mutex, "We will NOT pad frequency chunks to fake full bandwidth data");
  
  return EXIT_SUCCESS;
}

int default_arguments(conf_t *conf)
{
  int i;
  
  conf->dfsz_seek      = 0;      // Default to record the packet header
  conf->center_freq    = 0;      // Default with an impossible number
  conf->days_from_1970 = 0;      // Default with an impossible number
  conf->seconds_from_epoch = -1; // Default with an impossible number
  conf->df_in_period   = -1;     // Default with an impossible number

  conf->nport_alive   = 0;       // Default no alive port
  conf->nport_dead    = 0;       // Default no dead port
  conf->rbuf_ctrl_cpu = 0;       // Default to control the buffer with cpu 0
  conf->capture_ctrl_cpu = 0;    // Default to control the capture with cpu 0
  conf->capture_ctrl     = 0;    // Default do not control the capture during the runtime;
  conf->cpu_bind       = 0;      // Default do not bind thread to cpu
  conf->pad            = 0;      // Default do not pad
  conf->process_index  = -1;     // Default with an impossible value
  conf->beam_index     = -1;     // Default with an impossible value
  conf->ndf_per_chunk_rbuf = 0;  // Default with an impossible value
  conf->ndf_per_chunk_tbuf = 0;  // Default with an impossible value
  
  memset(conf->source, 0x00, sizeof(conf->source));
  memset(conf->ra, 0x00, sizeof(conf->ra));
  memset(conf->dec, 0x00, sizeof(conf->dec));
  memset(conf->dir, 0x00, sizeof(conf->dir));
  memset(conf->dada_header_template, 0x00, sizeof(conf->dada_header_template));
  memset(conf->instrument_name, 0x00, sizeof(conf->instrument_name));
  
  sprintf(conf->source, "unset");          // Default with "unset"
  sprintf(conf->ra, "unset");              // Default with "unset"
  sprintf(conf->dec, "unset");             // Default with "unset"
  sprintf(conf->dir, "unset");             // Default with "unset"
  sprintf(conf->dada_header_template, "unset"); // Default with "unset"
  sprintf(conf->instrument_name, "unset"); // Default with "unset"
  
  for(i = 0; i < MPORT_CAPTURE; i++)
    {      
      sprintf(conf->ip_alive[i], "unset"); // Default with "unset"
      sprintf(conf->ip_dead[i], "unset");  // Default with "unset"
      conf->port_dead[i] = 0;
      conf->port_alive[i] = 0;
      conf->capture_cpu[i] = 0;
      conf->nchunk_dead_on_port[i] = 0;
      conf->nchunk_alive_expect_on_port[i] = 0;
      conf->nchunk_alive_actual_on_port[i] = 0;
    }
  return EXIT_SUCCESS;
}
