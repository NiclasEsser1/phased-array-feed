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
#include "log.h"

char     *cbuf = NULL;
char     *tbuf = NULL;

int      quit = 0;

uint64_t ndf[MPORT_CAPTURE] = {0};

int      transit[MPORT_CAPTURE] = {0};
uint64_t tail[MPORT_CAPTURE] = {0};
uint64_t df_sec_ref[MPORT_CAPTURE];
uint64_t idf_prd_ref[MPORT_CAPTURE]; 

pthread_mutex_t ref_mutex[MPORT_CAPTURE]  = {PTHREAD_MUTEX_INITIALIZER};
pthread_mutex_t ndf_mutex[MPORT_CAPTURE] = {PTHREAD_MUTEX_INITIALIZER};
pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

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
      paf_log_add(conf->logfile, "ERR", 1, log_mutex, "Buffer size mismatch, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      
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
  
  /* Get the buffer block ready */
  uint64_t block_id = 0;
  uint64_t write_blkid;
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
  struct sockaddr_in sa, fromsa;
  socklen_t fromlen;// = sizeof(fromsa);
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
      quit = 1;
      free(df);
      paf_log_close(captureconf->logfile);
      conf = (void *)captureconf;
      pthread_exit(NULL);
      return NULL;
    }

  while(!quit)
    {
      if(recvfrom(sock, (void *)df, DFSZ, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
      	{
	  paf_log_add(captureconf->logfile, "ERR", 1, log_mutex, "Can not receive data from %s;%d, which happens at \"%s\", line [%d], has to abort", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
	  
	  /* Force to quit if we have time out */
      	  quit = 1;	  
      	  free(df);
	  paf_log_close(captureconf->logfile);
      	  conf = (void *)captureconf;
      	  pthread_exit(NULL);
      	  return NULL;
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
	  quit = 1;	  
	  free(df);
	  paf_log_close(captureconf->logfile);
	  conf = (void *)captureconf;
	  pthread_exit(NULL);
	  return NULL;
	}
      
      pthread_mutex_lock(&ref_mutex[ithread]);
      //idf_blk = (int64_t)(idf_prd - ref[ithread].idf_prd) + ((double)df_sec - (double)ref[ithread].df_sec) / df_res;
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
    
  /* Exit */
  quit = 1;
  free(df);
  conf = (void *)captureconf;
  close(sock);
  pthread_exit(NULL);

  return NULL;
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
      idf_prd_ref[i] = conf->idf_prd0 + conf->rbuf_ndf_chk;
      conf->nchk        += conf->nchk_alive_expect[i];
      conf->nchk_alive  += conf->nchk_alive_actual[i];
    }
  for(i = 0; i < conf->nport_dead; i++)
    conf->nchk       += conf->nchk_dead[i];
  
  if(conf->pad == 1)
    conf->nchk = 48;
  
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
  
  for(i = 0; i < MPORT_CAPTURE; i++)
    {
      pthread_mutex_destroy(&ref_mutex[i]);
      pthread_mutex_destroy(&ndf_mutex[i]);
    }
  
  pthread_mutex_destroy(&log_mutex);
  dada_hdu_unlock_write(conf.hdu);
  dada_hdu_disconnect(conf.hdu);
  dada_hdu_destroy(conf.hdu);
  
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
