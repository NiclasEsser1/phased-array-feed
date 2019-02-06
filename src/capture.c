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

#include "ipcbuf.h"
#include "capture.h"
#include "log.h"

char     *cbuf = NULL;
char     *tbuf = NULL;
int       quit = 0; // 0 means no quit, 1 means quit normal, 2 means quit with problem;

uint64_t ndf[MPORT_CAPTURE] = {0};
int      transit[MPORT_CAPTURE] = {0};
uint64_t tail[MPORT_CAPTURE] = {0};
uint64_t df_sec_ref[MPORT_CAPTURE];
uint64_t idf_prd_ref[MPORT_CAPTURE]; 

pthread_mutex_t ref_mutex[MPORT_CAPTURE]  = {PTHREAD_MUTEX_INITIALIZER};
pthread_mutex_t ndf_mutex[MPORT_CAPTURE] = {PTHREAD_MUTEX_INITIALIZER};
extern pthread_mutex_t log_mutex;

int init_buf(conf_t *conf)
{
  int i;

  /* Create HDU and check the size of buffer bolck */
  conf->required_dfsz = DFSZ - conf->dfoff;
  
  conf->rbufsz = conf->nchk * conf->required_dfsz * conf->rbuf_ndf_chk;  // The required buffer block size in byte;
  conf->hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(conf->hdu, conf->key);
  if(dada_hdu_connect(conf->hdu) < 0)
    { 
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Can not connect to hdu, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      
      for(i = 0; i < MPORT_CAPTURE; i++)
	{
	  pthread_mutex_destroy(&ref_mutex[i]);
	  pthread_mutex_destroy(&ndf_mutex[i]);
	}
      
      pthread_mutex_destroy(&log_mutex);
      dada_hdu_destroy(conf->hdu);
      paf_log_close(conf->logfile);
      
      exit(EXIT_FAILURE);    
    }
  
  if(dada_hdu_lock_write(conf->hdu) < 0) // make ourselves the write client
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Error locking HDU, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      
      for(i = 0; i < MPORT_CAPTURE; i++)
	{
	  pthread_mutex_destroy(&ref_mutex[i]);
	  pthread_mutex_destroy(&ndf_mutex[i]);
	}
      pthread_mutex_destroy(&log_mutex);
      
      dada_hdu_disconnect(conf->hdu);
      paf_log_close(conf->logfile);
      
      exit(EXIT_FAILURE);
    }
  conf->db_data = (ipcbuf_t *)(conf->hdu->data_block);
  conf->db_hdr  = (ipcbuf_t *)(conf->hdu->header_block);
  
  if(conf->rbufsz != ipcbuf_get_bufsz(conf->db_data))  // Check the buffer size
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Buffer size mismatch, %"PRIu64" vs %"PRIu64", which happens at \"%s\", line [%d], has to abort", conf->rbufsz, ipcbuf_get_bufsz(conf->db_data), __FILE__, __LINE__);
      
      for(i = 0; i < MPORT_CAPTURE; i++)
	{
	  pthread_mutex_destroy(&ref_mutex[i]);
	  pthread_mutex_destroy(&ndf_mutex[i]);
	}
      
      pthread_mutex_destroy(&log_mutex);
      dada_hdu_unlock_write(conf->hdu);
      paf_log_close(conf->logfile);
      
      exit(EXIT_FAILURE);    
    }

  if(conf->cpt_ctrl)
    {
      if(ipcbuf_disable_sod(conf->db_data) < 0)
	{
	  paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Can not write data before start, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  	  
	  for(i = 0; i < MPORT_CAPTURE; i++)
	    {
	      pthread_mutex_destroy(&ref_mutex[i]);
	      pthread_mutex_destroy(&ndf_mutex[i]);
	    }
	  
	  pthread_mutex_destroy(&log_mutex);
	  dada_hdu_unlock_write(conf->hdu);
	  paf_log_close(conf->logfile);
	  
	  exit(EXIT_FAILURE);
	}
    }
  else
    {
      conf->sec_int     = conf->sec_int_ref;
      conf->picoseconds = conf->picoseconds_ref;
      conf->mjd_start   = conf->sec_int / SECDAY + MJD1970;                       // Float MJD start time without fraction second
      strftime (conf->utc_start, MSTR_LEN, DADA_TIMESTR, gmtime(&conf->sec_int)); // String start time without fraction second 
      dada_header(*conf);
    }
  conf->tbufsz = (conf->required_dfsz + 1) * conf->tbuf_ndf_chk * conf->nchk;
  tbuf = (char *)malloc(conf->tbufsz * sizeof(char));// init temp buffer
  
  /* Get the buffer block ready, first to catch up */
  int sock;
  struct sockaddr_in sa = {0}, fromsa = {0};
  socklen_t fromlen = sizeof(fromsa);
  int64_t idf_blk = -1;
  uint64_t idf_prd, df_sec;
  char *df = NULL;
  uint64_t writebuf, *ptr = NULL;
  int nblk_delay = 0;
  
  df = (char *)malloc(sizeof(char) * DFSZ);
  
  sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&conf->tout, sizeof(conf->tout));  
  memset(&sa, 0x00, sizeof(sa));
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(conf->port_alive[0]);
  sa.sin_addr.s_addr = inet_addr(conf->ip_alive[0]);
  
  if(bind(sock, (struct sockaddr *)&sa, sizeof(sa)) == -1)
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Can not bind to %s;%d, which happens at \"%s\", line [%d], has to abort", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
      
      /* Force to quit if we have time out */
      free(df);
      paf_log_close(conf->logfile);
      exit(EXIT_FAILURE);
    }
  
  if(recvfrom(sock, (void *)df, DFSZ, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Can not receive data from %s;%d, which happens at \"%s\", line [%d], has to abort", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);

      free(df);
      paf_log_close(conf->logfile);
      exit(EXIT_FAILURE);
    }
  
  ptr = (uint64_t*)df;
  writebuf = bswap_64(*ptr);
  idf_prd = writebuf & 0x00000000ffffffff;
  df_sec = (writebuf & 0x3fffffff00000000) >> 32;
  
  idf_blk = (int64_t)(idf_prd - idf_prd_ref[0]) + ((double)df_sec - (double)df_sec_ref[0]) / conf->df_res;
  if(idf_blk>0) // The reference time is really out of data
    {
      nblk_delay = (int)floor(idf_blk/(double)conf->rbuf_ndf_chk);
      for(i = 0; i < nblk_delay; i++)
	{      
	  cbuf = ipcbuf_get_next_write(conf->db_data);
	  if(cbuf == NULL)
	    {
	      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "open_buffer failed, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	      free(df);
	      paf_log_close(conf->logfile);
	      exit(EXIT_FAILURE);
	    }
	  
	  if(ipcbuf_mark_filled(conf->db_data, conf->rbufsz) < 0)
	    {
	      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "close_buffer failed, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);

	      free(df);
	      paf_log_close(conf->logfile);
	      exit(EXIT_FAILURE);
	    }
	  
	  conf->idf_prd0 += conf->rbuf_ndf_chk;
	  if(conf->idf_prd0 >= NDF_CHK_PRD)
	    {
	      conf->df_sec0  += PRD;
	      conf->idf_prd0 -= NDF_CHK_PRD;
	    }
	}
      for(i = 0; i < conf->nport_alive; i++)
	{
	  df_sec_ref[i]  = conf->df_sec0;
	  idf_prd_ref[i] = conf->idf_prd0;
	}
    }
  free(df);
  close(sock);
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "nblk_delay is %d", nblk_delay);
  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "reference info is %"PRIu64"\t%"PRIu64"", conf->df_sec0, conf->idf_prd0);
  
  cbuf = ipcbuf_get_next_write(conf->db_data);
  if(cbuf == NULL)
    {
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "open_buffer failed, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf->logfile);      
      exit(EXIT_FAILURE);
    }
  
  return EXIT_SUCCESS;
}

void *capture(void *conf)
{
  char *df = NULL;
  conf_t *captureconf = (conf_t *)conf;
  int sock, ichk; 
  struct sockaddr_in sa = {0}, fromsa = {0};
  socklen_t fromlen = sizeof(fromsa);
  int64_t idf_blk = -1;
  uint64_t idf_prd, df_sec;
  double freq;
  int epoch;
  uint64_t tbuf_loc, cbuf_loc;
  register int nchk = captureconf->nchk;
  register double ichk0 = captureconf->ichk0;
  register double df_res = captureconf->df_res;
  struct timespec start, now;
  uint64_t writebuf, *ptr = NULL;

  register int dfoff = captureconf->dfoff;
  register int required_dfsz = captureconf->required_dfsz;
  register uint64_t rbuf_ndf_chk = captureconf->rbuf_ndf_chk;
  register uint64_t rbuf_tbuf_ndf_chk = captureconf->rbuf_ndf_chk + captureconf->tbuf_ndf_chk;
  register int ithread = captureconf->ithread;
  
  df = (char *)malloc(sizeof(char) * DFSZ);
  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "In funtion thread id = %ld, %d, %d", (long)pthread_self(), captureconf->ithread, ithread);
    
  sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&captureconf->tout, sizeof(captureconf->tout));  
  memset(&sa, 0x00, sizeof(sa));
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(captureconf->port_alive[ithread]);
  sa.sin_addr.s_addr = inet_addr(captureconf->ip_alive[ithread]);
  
  if(bind(sock, (struct sockaddr *)&sa, sizeof(sa)) == -1)
    {
      paf_log_add(captureconf->logfile, "ERR", 1, log_mutex, "Can not bind to %s;%d, which happens at \"%s\", line [%d], has to abort", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
            
      /* Force to quit if we have time out */
      quit = 2;
      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "free(df) in bind %d", captureconf->ithread);
      free(df);
      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "done free(df) in bind %d", captureconf->ithread);
      
      pthread_exit(NULL);
    }

  do{    
    if(recvfrom(sock, (void *)df, DFSZ, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
      {
	paf_log_add(captureconf->logfile, "ERR", 1, log_mutex, "Can not receive data from %s;%d, which happens at \"%s\", line [%d], has to abort", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
	
	/* Force to quit if we have time out */
	quit = 2;
	paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "free(df) in recvfrom %d", captureconf->ithread);
	free(df);
	paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "done free(df) in recvfrom %d", captureconf->ithread);
	
	pthread_exit(NULL);
      }

    /* Get header information from bmf packet */    
    ptr = (uint64_t*)df;
    writebuf = bswap_64(*ptr);
    idf_prd = writebuf & 0x00000000ffffffff;
    df_sec = (writebuf & 0x3fffffff00000000) >> 32;
        
    pthread_mutex_lock(&ref_mutex[ithread]);
    idf_blk = (int64_t)(idf_prd - idf_prd_ref[ithread]) + ((double)df_sec - (double)df_sec_ref[ithread]) / df_res;
    pthread_mutex_unlock(&ref_mutex[ithread]);
  }while(idf_blk<0);
  fprintf(stdout, "CAPTURE_READY\n"); // Tell other process that the capture is ready
  fflush(stdout);
    
  while(!quit)
    {
      if(recvfrom(sock, (void *)df, DFSZ, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
      	{
	  paf_log_add(captureconf->logfile, "ERR", 1, log_mutex, "Can not receive data from %s;%d, which happens at \"%s\", line [%d], has to abort", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
	  
	  /* Force to quit if we have time out */
      	  quit = 2;
	  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "free(df) in recvfrom, while loop, %d", captureconf->ithread);
      	  free(df);
	  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "done free(df) in recvfrom, while loop, %d", captureconf->ithread);

      	  pthread_exit(NULL);
      	}

      /* Get header information from bmf packet */    
      ptr = (uint64_t*)df;
      writebuf = bswap_64(*ptr);
      idf_prd = writebuf & 0x00000000ffffffff;
      df_sec = (writebuf & 0x3fffffff00000000) >> 32;
      writebuf = bswap_64(*(ptr + 2));
      freq = (double)((writebuf & 0x00000000ffff0000) >> 16);
  
      ichk = (int)(freq/NCHAN_CHK + ichk0);
      if (ichk<0 || ichk > (captureconf->nchk-1))
	{      
	  paf_log_add(captureconf->logfile, "ERR", 1, log_mutex, "Frequency chunk is outside the range [0 %d], which happens at \"%s\", line [%d], has to abort", captureconf->nchk, __FILE__, __LINE__);
	  
	  /* Force to quit if we have time out */
	  quit = 2;
	  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "free(df) in ichk, while loop, %d", captureconf->ithread);
	  free(df);
	  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "done free(df) in ichk, while loop, %d", captureconf->ithread);
	  
	  pthread_exit(NULL);
	}
      
      pthread_mutex_lock(&ref_mutex[ithread]);
      idf_blk = (int64_t)(idf_prd - idf_prd_ref[ithread]) + ((double)df_sec - (double)df_sec_ref[ithread]) / df_res;
      pthread_mutex_unlock(&ref_mutex[ithread]);

      if(idf_blk>=0)
	{
	  if(idf_blk < rbuf_ndf_chk)  // Put data into current ring buffer block
	    {
	      transit[ithread] = 0; // The reference is already updated.
	      /* Put data into current ring buffer block if it is before rbuf_ndf_chk; */
	      //cbuf_loc = (uint64_t)((idf + ichk * rbuf_ndf_chk) * required_dfsz);   // This should give us FTTFP (FTFP) order
	      cbuf_loc = (uint64_t)((idf_blk * nchk + ichk) * required_dfsz); // This is in TFTFP order
	      memcpy(cbuf + cbuf_loc, df + dfoff, required_dfsz);
	      
	      pthread_mutex_lock(&ndf_mutex[ithread]);
	      ndf[ithread]++;
	      pthread_mutex_unlock(&ndf_mutex[ithread]);
	    }
	  else
	    {
	      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Cross the boundary");

	      transit[ithread] = 1; // The reference should be updated very soon
	      if(idf_blk < rbuf_tbuf_ndf_chk)
		{		  
		  tail[ithread] = (uint64_t)((idf_blk - rbuf_ndf_chk) * nchk + ichk); // This is in TFTFP order
		  tbuf_loc      = (uint64_t)(tail[ithread] * (required_dfsz + 1));
		  
		  tail[ithread]++;  // Otherwise we will miss the last available data frame in tbuf;
		  
		  tbuf[tbuf_loc] = 'Y';
		  memcpy(tbuf + tbuf_loc + 1, df + dfoff, required_dfsz);
		  
		  pthread_mutex_lock(&ndf_mutex[ithread]);
		  ndf[ithread]++;
		  pthread_mutex_unlock(&ndf_mutex[ithread]);
		}
	      
	    }
	}
      else
	transit[ithread] = 0;
    }
  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "DONE the capture thread, which happens at \"%s\", line [%d]", __FILE__, __LINE__);
  
  /* Exit */
  quit = 1;
  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "free(df) after while loop, %d", captureconf->ithread);
  free(df);
  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "done free(df) after while loop, %d", captureconf->ithread);
  close(sock);
  
  pthread_exit(NULL);
}

int init_capture(conf_t *conf)
{
  int i;

  if(conf->cpt_ctrl)
    sprintf(conf->cpt_ctrl_addr, "%s/capture.socket", conf->dir);  // The file will be in different directory for different beam;

  conf->nchk       = 0;
  conf->nchk_alive = 0;
  
  /* Init status */
  for(i = 0; i < conf->nport_alive; i++)
    {
      df_sec_ref[i]  = conf->df_sec0;
      idf_prd_ref[i] = conf->idf_prd0;
      conf->nchk        += conf->nchk_alive_expect[i];
      conf->nchk_alive  += conf->nchk_alive_actual[i];
    }
  for(i = 0; i < conf->nport_dead; i++)
    conf->nchk       += conf->nchk_dead[i];
  
  if(conf->pad == 1)
    conf->nchk = NCHK_FULL_BAND;
  
  conf->df_res       = (double)PRD/(double)NDF_CHK_PRD;
  conf->blk_res      = conf->df_res * (double)conf->rbuf_ndf_chk;
  conf->nchan        = conf->nchk * NCHAN_CHK;
  conf->ichk0        = -(conf->cfreq + 1.0)/NCHAN_CHK + 0.5 * conf->nchk;
  
  conf->sec_int_ref     = floor(conf->df_res * conf->idf_prd0) + conf->df_sec0 + SECDAY * conf->epoch0;
  conf->picoseconds_ref = 1E6 * round(1.0E6 * (PRD - floor(conf->df_res * conf->idf_prd0)));

  conf->tout.tv_sec     = PRD;
  conf->tout.tv_usec    = 0;
  
  init_buf(conf);  // Initi ring buffer, must be here;
  
  return EXIT_SUCCESS;
}

int destroy_capture(conf_t conf)
{
  int i;
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Inside destroy_capture");
  
  for(i = 0; i < MPORT_CAPTURE; i++)
    {
      pthread_mutex_destroy(&ref_mutex[i]);
      pthread_mutex_destroy(&ndf_mutex[i]);
    }
  pthread_mutex_destroy(&log_mutex);
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Finish destroy_mutex");
  dada_hdu_unlock_write(conf.hdu);
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Finish dada_hdu_unlock");
  dada_hdu_disconnect(conf.hdu);
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Finish dada_hdu_disconnect");
  dada_hdu_destroy(conf.hdu);
  paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Finish destroy_capture");
  
  return EXIT_SUCCESS;
}

int dada_header(conf_t conf)
{
  char *hdrbuf = NULL;
  
  /* Register header */
  hdrbuf = ipcbuf_get_next_write(conf.db_hdr);
  if(!hdrbuf)
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);
      
      exit(EXIT_FAILURE);
    }
  if(!conf.hfname)
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Please specify header file, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);
      
      exit(EXIT_FAILURE);
    }  
  if(fileread(conf.hfname, hdrbuf, DADA_HDRSZ) < 0)
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Error reading header file, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);

      exit(EXIT_FAILURE);
    }
  
  /* Setup DADA header with given values */
  if(ascii_header_set(hdrbuf, "UTC_START", "%s", conf.utc_start) < 0)  
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Error setting UTC_START, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);
      
      exit(EXIT_FAILURE);
    }
  
  if(ascii_header_set(hdrbuf, "RA", "%s", conf.ra) < 0)  
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Error setting RA, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);
      
      exit(EXIT_FAILURE);
    }
  
  if(ascii_header_set(hdrbuf, "DEC", "%s", conf.dec) < 0)  
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Error setting DEC, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);

      exit(EXIT_FAILURE);
    }
  
  if(ascii_header_set(hdrbuf, "SOURCE", "%s", conf.source) < 0)  
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Error setting SOURCE, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);
      
      exit(EXIT_FAILURE);
    }
  
  if(ascii_header_set(hdrbuf, "INSTRUMENT", "%s", conf.instrument) < 0)  
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Error setting INSTRUMENT, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);
      
      exit(EXIT_FAILURE);
    }
  
  if(ascii_header_set(hdrbuf, "PICOSECONDS", "%"PRIu64, conf.picoseconds) < 0)  
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Error setting PICOSECONDS, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);
      
      exit(EXIT_FAILURE);
    }    
  if(ascii_header_set(hdrbuf, "FREQ", "%.6lf", conf.cfreq) < 0)
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Error setting FREQ, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);

      exit(EXIT_FAILURE);
    }
  if(ascii_header_set(hdrbuf, "MJD_START", "%.10lf", conf.mjd_start) < 0)
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Error setting MJD_START, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);
      
      exit(EXIT_FAILURE);
    }
  if(ascii_header_set(hdrbuf, "NCHAN", "%d", conf.nchan) < 0)
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Error setting NCHAN, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);
      
      exit(EXIT_FAILURE);
    }  
  if(ascii_header_get(hdrbuf, "RESOLUTION", "%lf", &conf.chan_res) < 0)
    {
      paf_log_add(conf.logfile, "INFO", 1,  log_mutex, "Error getting RESOLUTION, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);

      exit(EXIT_FAILURE);
    }
  conf.bw = conf.chan_res * conf.nchan;
  if(ascii_header_set(hdrbuf, "BW", "%.6lf", conf.bw) < 0)
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Error setting BW, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);
      
      exit(EXIT_FAILURE);
    }
  /* donot set header parameters anymore - acqn. doesn't start */
  if(ipcbuf_mark_filled(conf.db_hdr, DADA_HDRSZ) < 0)
    {
      paf_log_add(conf.logfile, "INFO", 1, log_mutex, "Error header_fill, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      paf_log_close(conf.logfile);
      
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
  conf_t conf_thread[MPORT_CAPTURE];
    
  for(i = 0; i < nport_alive; i++)
    // Create threads. Capture threads and ring buffer control thread are essential, the capture control thread is created when it is required to;
    // If we create a capture control thread, we can control the start and end of data during runtime, the header of DADA buffer will be setup each time we start the data, which means without rerun the pipeline, we can get multiple capture runs;
    // If we do not create a capture thread, the data will start at the begining and we need to setup the header at that time, we can only do one capture without rerun the pipeline;
    {
      conf_thread[i] = *conf;
      conf_thread[i].ithread = i;
      
      if(!(conf->cpu_bind == 0))
	{
	  pthread_attr_init(&attr);  
	  CPU_ZERO(&cpus);
	  CPU_SET(conf->cpt_cpu[i], &cpus);
	  pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
	  ret[i] = pthread_create(&thread[i], &attr, capture, (void *)&conf_thread[i]);
	  pthread_attr_destroy(&attr);
	}
      else
	ret[i] = pthread_create(&thread[i], &attr, capture, (void *)&conf_thread[i]);
    }

  if(!(conf->cpu_bind == 0)) 
    {            
      pthread_attr_init(&attr);
      CPU_ZERO(&cpus);
      CPU_SET(conf->rbuf_ctrl_cpu, &cpus);      
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);	
      ret[nport_alive] = pthread_create(&thread[nport_alive], &attr, buf_control, (void *)conf);
      pthread_attr_destroy(&attr);

      if(conf->cpt_ctrl)
	{
	  pthread_attr_init(&attr);
	  CPU_ZERO(&cpus);
	  CPU_SET(conf->cpt_ctrl_cpu, &cpus);      
	  pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);	
	  ret[nport_alive + 1] = pthread_create(&thread[nport_alive + 1], &attr, capture_control, (void *)conf);
	  pthread_attr_destroy(&attr);
	}
    }
  else
    {
      ret[nport_alive] = pthread_create(&thread[nport_alive], NULL, buf_control, (void *)conf);
      if(conf->cpt_ctrl)
	ret[nport_alive + 1] = pthread_create(&thread[nport_alive + 1], NULL, capture_control, (void *)conf);
    }

  if (conf->cpt_ctrl)
    nthread = nport_alive + 2;
  else
    nthread = nport_alive + 1;

  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "Join threads? Before it");
  for(i = 0; i < nthread; i++)   // Join threads and unbind cpus
    pthread_join(thread[i], NULL);

  paf_log_add(conf->logfile, "INFO", 1, log_mutex, "Join threads? The last quit is %d", quit);
    
  return EXIT_SUCCESS;
}

void *buf_control(void *conf)
{
  conf_t *captureconf = (conf_t *)conf;
  int i, nchk = captureconf->nchk, transited = 0;
  uint64_t cbuf_loc, tbuf_loc, ntail;
  int ichk, idf;
  uint64_t rbuf_nblk = 0;
  uint64_t ndf_actual = 0, ndf_expect = 0;
  uint64_t ndf_blk_actual = 0, ndf_blk_expect = 0;
  double sleep_time = 0.5 * captureconf->blk_res;
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
	  for(i = 1; i < captureconf->nport_alive; i++) // When all ports are on the transit status
	    //transited = transited && transit[i]; // all happen, take action
	    transited = transited || transit[i]; // one happens, take action
	}
      
      if(quit == 1)
	{
	  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Quit just after the buffer transit state change");
	  pthread_exit(NULL);
	}
      if(quit == 2)
	{
	  paf_log_add(captureconf->logfile, "ERR", 1, log_mutex, "Quit just after the buffer transit state change");
	  
	  pthread_exit(NULL);
	}
      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Just after buffer transit state change");
      
      /* Check the traffic of previous buffer cycle */
      rbuf_nblk = ipcbuf_get_write_count(captureconf->db_data) + 1;
      ndf_blk_expect = 0;
      ndf_blk_actual = 0;
      for(i = 0; i < captureconf->nport_alive; i++)
	{
	  pthread_mutex_lock(&ndf_mutex[i]); 
	  ndf_blk_actual += ndf[i];
	  ndf[i] = 0; 
	  pthread_mutex_unlock(&ndf_mutex[i]);
	}
      ndf_actual += ndf_blk_actual;
      if(rbuf_nblk==1)
	{
	  for(i = 0; i < captureconf->nport_alive; i++)
	    ndf_blk_expect += captureconf->rbuf_ndf_chk * captureconf->nchk_alive_actual[i];
	}
      else
	ndf_blk_expect += captureconf->rbuf_ndf_chk * captureconf->nchk_alive; // Only for current buffer
      ndf_expect += ndf_blk_expect;

      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "%s starts from port %d, packet loss rate %d %f %E %E", captureconf->ip_alive[0], captureconf->port_alive[0], rbuf_nblk * captureconf->blk_res, (1.0 - ndf_actual/(double)ndf_expect), (1.0 - ndf_blk_actual/(double)ndf_blk_expect));
      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Packets counters, %"PRIu64" %"PRIu64" %"PRIu64" %"PRIu64"", ndf_actual, ndf_expect, ndf_blk_actual, ndf_blk_expect);
      
      fprintf(stdout, "CAPTURE_STATUS %d %f %E %E\n", captureconf->process_index, rbuf_nblk * captureconf->blk_res, (1.0 - ndf_actual/(double)ndf_expect), (1.0 - ndf_blk_actual/(double)ndf_blk_expect)); // Pass the status to stdout
      fflush(stdout);
      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "After fflush stdout");
      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Before mark filled");
      
      /* Close current buffer */
      if(ipcbuf_mark_filled(captureconf->db_data, captureconf->rbufsz) < 0)
	{
	  paf_log_add(captureconf->logfile, "ERR", 1, log_mutex, "close_buffer failed, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  quit = 2;
	  pthread_exit(NULL);
	}
      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Mark filled done");
      
      /*
	To see if the buffer is full, quit if yes.
	If we have a reader, there will be at least one buffer which is not full
      */
      if(ipcbuf_get_nfull(captureconf->db_data) >= (ipcbuf_get_nbufs(captureconf->db_data) - 1)) 
	{
	  paf_log_add(captureconf->logfile, "ERR", 1, log_mutex, "buffers are all full, which happens at \"%s\", line [%d], has to abort.", __FILE__, __LINE__);
	  
	  quit = 2;
	  pthread_exit(NULL);
	}
      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Available buffer block check done");
      
      /* Get new buffer block */
      cbuf = ipcbuf_get_next_write(captureconf->db_data);
      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Get next write done");
      if(cbuf == NULL)
	{
	  paf_log_add(captureconf->logfile, "ERR", 1, log_mutex, "open_buffer failed, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
	  
	  quit = 2;
	  pthread_exit(NULL);
	}
      
      /* Update reference point */
      for(i = 0; i < captureconf->nport_alive; i++)
	{
	  // Update the reference hdr, once capture thread get the updated reference, the data will go to the next block or be dropped;
	  // We have to put a lock here as partial update of reference hdr will be a trouble to other threads;
	  
	  pthread_mutex_lock(&ref_mutex[i]);
	  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Start to change the reference information %"PRIu64" %"PRIu64"", df_sec_ref[i], idf_prd_ref[i]);
	  idf_prd_ref[i] += captureconf->rbuf_ndf_chk;
	  if(idf_prd_ref[i] >= NDF_CHK_PRD)
	    {
	      df_sec_ref[i]  += PRD;
	      idf_prd_ref[i] -= NDF_CHK_PRD;
	    }
	  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Finish the change of reference information %"PRIu64" %"PRIu64"", df_sec_ref[i], idf_prd_ref[i]);
	  pthread_mutex_unlock(&ref_mutex[i]);
	}
      
      /* To see if we need to copy data from temp buffer into ring buffer */
      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Just before the second transit state change");
      while(transited && (!quit))
	{
	  transited = transit[0];
	  for(i = 1; i < captureconf->nport_alive; i++)
	    //transited = transited || transit[i]; // all happen, take action
	    transited = transited && transit[i]; // one happens, take action
	}      
      if(quit == 1)
	{
	  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Quit just after the second transit state change");

	  //pthread_detach(pthread_self());
	  pthread_exit(NULL);
	  //return NULL;
	}    
      if(quit == 2)
	{
	  paf_log_add(captureconf->logfile, "ERR", 1, log_mutex, "Quit just after the second transit state change");

	  //pthread_detach(pthread_self());
	  pthread_exit(NULL);
	  //return NULL;
	}

      ntail = 0;
      for(i = 0; i < captureconf->nport_alive; i++)
	ntail = (tail[i] > ntail) ? tail[i] : ntail;
      
      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "The location of the last packet in temp buffer is %"PRIu64"", ntail);
      for(i = 0; i < ntail; i++)
	{
	  tbuf_loc = (uint64_t)(i * (captureconf->required_dfsz + 1));	      
	  if(tbuf[tbuf_loc] == 'Y')
	    {		  
	      cbuf_loc = (uint64_t)(i * captureconf->required_dfsz);  // This is for the TFTFP order temp buffer copy;
	      
	      //idf = (int)(i / nchk);
	      //ichk = i - idf * nchk;
	      //cbuf_loc = (uint64_t)(ichk * captureconf->rbuf_ndf_chk + idf) * captureconf->required_dfsz;  // This is for the FTFP order temp buffer copy;		
	      
	      memcpy(cbuf + cbuf_loc, tbuf + tbuf_loc + 1, captureconf->required_dfsz);		  
	      tbuf[tbuf_loc + 1] = 'N';  // Make sure that we do not copy the data later;
	      // If we do not do that, we may have too many data frames to copy later
	    }
	}
      for(i = 0; i < captureconf->nport_alive; i++)
	tail[i] = 0;  // Reset the location of tbuf;
      
      sleep(sleep_sec);
      usleep(sleep_usec);
    }
  
  /* Exit */
  if(ipcbuf_mark_filled(captureconf->db_data, captureconf->rbufsz) < 0)
    {
      paf_log_add(captureconf->logfile, "ERR", 1, log_mutex, "ipcio_close_block failed, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      quit = 2;

      //pthread_detach(pthread_self());
      pthread_exit(NULL);
      //return NULL;
    }
  
  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Normale quit of buffer control thread");
  //pthread_detach(pthread_self());
  pthread_exit(NULL);
  //return NULL;
}


void *capture_control(void *conf)
{  
  int sock, i;
  struct sockaddr_un sa = {0}, fromsa = {0};
  socklen_t fromlen = sizeof(fromsa);
  conf_t *captureconf = (conf_t *)conf;
  char command_line[MSTR_LEN], command[MSTR_LEN];
  int64_t start_buf;
  int64_t available_buf;
  double sec_offset; // Offset from the reference time;
  uint64_t picoseconds_offset; // The sec_offset fraction part in picoseconds
  int msg_len;

  /* Create an unix socket for control */
  if((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1)
    {
      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Error setting NCHAN, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      
      quit = 1;
      pthread_exit(NULL);
    }
  
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&captureconf->tout, sizeof(captureconf->tout));
  memset(&sa, 0, sizeof(struct sockaddr_un));
  sa.sun_family = AF_UNIX;
  strncpy(sa.sun_path, captureconf->cpt_ctrl_addr, strlen(captureconf->cpt_ctrl_addr));
  unlink(captureconf->cpt_ctrl_addr);
  
  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, captureconf->cpt_ctrl_addr);
  
  if(bind(sock, (struct sockaddr*)&sa, sizeof(sa)) == -1)
    {
      paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Can not bind to file socket, which happens at \"%s\", line [%d]", __FILE__, __LINE__);
       
      quit = 1;
      close(sock);
      pthread_exit(NULL);
 
    }

  fprintf(stdout, "CAPTURE_READY\n"); // Tell other process that the capture is ready
  fflush(stdout);
  
  while(!quit)
    {
      msg_len = 0;
      while(!(msg_len > 0) && !quit)
	{
	  //fprintf(stdout, "%s\n", command_line);
	  //fflush(stdout);
	  memset(command_line, 0, MSTR_LEN);
	  msg_len = recvfrom(sock, (void *)command_line, MSTR_LEN, 0, (struct sockaddr*)&fromsa, &fromlen);
	}
      if(quit)
	{	  
	  close(sock);
	  //pthread_detach(pthread_self());
	  pthread_exit(NULL);
	  //return NULL;
	}
      //fprintf(stdout, "%s passed\n", command_line);
      //fprintf(stdout, "%s inside\n", command_line);
      
      if(strstr(command_line, "END-OF-CAPTURE") != NULL)
	{
	  //fprintf(stdout, "%s inside\n", command_line);
	  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Got END-OF-CAPTURE signal, has to quit");      
	  
	  quit = 1;	      
	  if(ipcbuf_is_writing(captureconf->db_data))
	    ipcbuf_enable_eod(captureconf->db_data);
	  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Got END-OF-CAPTURE signal, after ENABLE_EOD");      
	  
	  close(sock);
	  //pthread_detach(pthread_self());
	  pthread_exit(NULL);
	  //exit(EXIT_SUCCESS);
	  //return NULL;
	}  
      if(strstr(command_line, "END-OF-DATA") != NULL)
	{
	  //fprintf(stdout, "%s inside\n", command_line);
	  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Got END-OF-DATA signal, has to enable eod");
	  ipcbuf_enable_eod(captureconf->db_data);
	  memset(command_line, 0, MSTR_LEN);
	}
	  
      if(strstr(command_line, "START-OF-DATA") != NULL)
	{
	  //fprintf(stdout, "%s inside\n", command_line);
	  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "Got START-OF-DATA signal, has to enable sod");
	  
	  sscanf(command_line, "%[^_]_%[^_]_%[^_]_%[^_]_%"SCNd64"", command, captureconf->source, captureconf->ra, captureconf->dec, &start_buf); // Read the start buffer from socket or get the minimum number from the buffer, we keep starting at the begining of buffer block;
	  //available_buf = ipcbuf_get_write_count(captureconf->db_data) - ipcbuf_get_nbufs(captureconf->db_hdr);
	  //if (available_buf <0) // In case we start at the very beginning
	  //available_buf = 0;
	  available_buf = ipcbuf_get_write_count(captureconf->db_data);
	  start_buf = (start_buf > available_buf) ? start_buf : available_buf; // To make sure the start buffer is valuable, to get the most recent buffer
	  paf_log_add(captureconf->logfile, "INFO", 1, log_mutex, "The data is enabled at %"PRIu64" buffer block", start_buf);
	  ipcbuf_enable_sod(captureconf->db_data, (uint64_t)start_buf, 0);
	  //fprintf(stdout, "%s pass SOD\n", command_line);
	  
	  /* To get time stamp for current header */
	  sec_offset = start_buf * captureconf->blk_res; // Only work with buffer number
	  picoseconds_offset = 1E6 * (round(1.0E6 * (sec_offset - floor(sec_offset))));
	  
	  captureconf->picoseconds = picoseconds_offset +captureconf->picoseconds_ref;
	  captureconf->sec_int = captureconf->sec_int_ref + floor(sec_offset);
	  if(!(captureconf->picoseconds < 1E12))
	    {
	      captureconf->sec_int += 1;
	      captureconf->picoseconds -= 1E12;
	    }
	  strftime (captureconf->utc_start, MSTR_LEN, DADA_TIMESTR, gmtime(&captureconf->sec_int)); // String start time without fraction second 
	  captureconf->mjd_start = captureconf->sec_int / SECDAY + MJD1970;                         // Float MJD start time without fraction second
	  
	  /*
	    To see if the buffer is full, quit if yes.
	    If we have a reader, there will be at least one buffer which is not full
	  */
	  if(ipcbuf_get_nfull(captureconf->db_hdr) >= (ipcbuf_get_nbufs(captureconf->db_hdr) - 1)) 
	    {
	      paf_log_add(captureconf->logfile, "ERR", 1, log_mutex, "buffers are all full, has to abort");
	      
	      quit = 2;
	      pthread_exit(NULL);
	    }
	  
	  /* setup dada header here */
	  if(dada_header(*captureconf))
	    {		  
	      quit = 1;
	      close(sock);
	      //pthread_detach(pthread_self());
	      pthread_exit(NULL);
	      //return NULL;
	    }
	  memset(command_line, 0, MSTR_LEN);
	}
    }
  
  quit = 1;
  close(sock);
  //pthread_detach(pthread_self());
  pthread_exit(NULL);
  //return NULL;
}
