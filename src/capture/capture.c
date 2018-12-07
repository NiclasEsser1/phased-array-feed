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

#include "ipcbuf.h"
#include "capture.h"
#include "multilog.h"

extern multilog_t *runtime_log;

char *cbuf = NULL;
char *tbuf = NULL;

int quit = 0;
int ithread_extern = 0;

uint64_t ndf_port[MPORT_CAPTURE] = {0};

int transit[MPORT_CAPTURE] = {0};
uint64_t tail[MPORT_CAPTURE] = {0};
hdr_t hdr0[MPORT_CAPTURE]; // It is the reference for packet counter, we do not need mutex lock for this
hdr_t hdr_ref[MPORT_CAPTURE];
hdr_t hdr_current[MPORT_CAPTURE];

pthread_mutex_t ithread_mutex = PTHREAD_MUTEX_INITIALIZER;

pthread_mutex_t ndf_port_mutex[MPORT_CAPTURE] = {PTHREAD_MUTEX_INITIALIZER};

pthread_mutex_t hdr_ref_mutex[MPORT_CAPTURE]     = {PTHREAD_MUTEX_INITIALIZER};
pthread_mutex_t hdr_current_mutex[MPORT_CAPTURE] = {PTHREAD_MUTEX_INITIALIZER};

int init_buf(conf_t *conf)
{
  int i, nbufs;
  ipcbuf_t *db = NULL;

  /* Create HDU and check the size of buffer bolck */
  conf->required_pktsz = conf->pktsz - conf->pktoff;
  
  conf->rbufsz = conf->nchk * conf->required_pktsz * conf->rbuf_ndf_chk;  // The required buffer block size in byte;
  conf->hdu = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu, conf->key);
  if(dada_hdu_connect(conf->hdu) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Can not connect to hdu, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
      pthread_mutex_destroy(&ithread_mutex);
      for(i = 0; i < MPORT_CAPTURE; i++)
	{
	  pthread_mutex_destroy(&hdr_ref_mutex[i]);
	  pthread_mutex_destroy(&hdr_current_mutex[i]);
	  pthread_mutex_destroy(&ndf_port_mutex[i]);
	}
      
      dada_hdu_destroy(conf->hdu);
      return EXIT_FAILURE;    
    }
  
  if(dada_hdu_lock_write(conf->hdu) < 0) // make ourselves the write client
    {
      multilog(runtime_log, LOG_ERR, "Error locking HDU, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
  
      pthread_mutex_destroy(&ithread_mutex);
      for(i = 0; i < MPORT_CAPTURE; i++)
	{
	  pthread_mutex_destroy(&hdr_ref_mutex[i]);
	  pthread_mutex_destroy(&hdr_current_mutex[i]);
	  pthread_mutex_destroy(&ndf_port_mutex[i]);
	}
      
      dada_hdu_disconnect(conf->hdu);
      return EXIT_FAILURE;
    }
  
  db = (ipcbuf_t *)conf->hdu->data_block;
  if(conf->rbufsz != ipcbuf_get_bufsz(db))  // Check the buffer size
    {
      multilog(runtime_log, LOG_ERR, "Buffer size mismatch, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
        
      pthread_mutex_destroy(&ithread_mutex);
      for(i = 0; i < MPORT_CAPTURE; i++)
	{
	  pthread_mutex_destroy(&hdr_ref_mutex[i]);
	  pthread_mutex_destroy(&hdr_current_mutex[i]);
	  pthread_mutex_destroy(&ndf_port_mutex[i]);
	}
      
      dada_hdu_unlock_write(conf->hdu);
      return EXIT_FAILURE;    
    }
  if(ipcbuf_disable_sod(db) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
  
      pthread_mutex_destroy(&ithread_mutex);
      for(i = 0; i < MPORT_CAPTURE; i++)
  	{
  	  pthread_mutex_destroy(&hdr_ref_mutex[i]);
  	  pthread_mutex_destroy(&hdr_current_mutex[i]);
  	  pthread_mutex_destroy(&ndf_port_mutex[i]);
  	}
      
      dada_hdu_unlock_write(conf->hdu);
      return EXIT_FAILURE;
    }
  conf->tbufsz = (conf->required_pktsz + 1) * conf->tbuf_ndf_chk * conf->nchk;
  tbuf = (char *)malloc(conf->tbufsz * sizeof(char));// init temp buffer
  
  return EXIT_SUCCESS;
}

void *capture(void *conf)
{
  char *df = NULL;
  conf_t *captureconf = (conf_t *)conf;
  int sock, ithread, pktsz, pktoff, required_pktsz, ichk; 
  struct sockaddr_in sa, fromsa;
  struct timeval tout={captureconf->sec_prd, 0};  // Force to timeout if we could not receive data frames for one period.
  socklen_t fromlen;// = sizeof(fromsa);
  int64_t idf;
  uint64_t tbuf_loc, cbuf_loc;
  hdr_t hdr;
  double elapsed_time;
  struct timespec start, stop;

  init_hdr(&hdr); 
  
  pktsz          = captureconf->pktsz;
  pktoff         = captureconf->pktoff;
  required_pktsz = captureconf->required_pktsz;
  df = (char *)malloc(sizeof(char) * pktsz);
  
  /* Get right socker for current thread */
  pthread_mutex_lock(&ithread_mutex);
  ithread = ithread_extern;
  ithread_extern++;
  pthread_mutex_unlock(&ithread_mutex);
  
  sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tout, sizeof(tout));  
  memset(&sa, 0x00, sizeof(sa));
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(captureconf->port_active[ithread]);
  sa.sin_addr.s_addr = inet_addr(captureconf->ip_active[ithread]);
  if(bind(sock, (struct sockaddr *)&sa, sizeof(sa)) == -1)
    {
      multilog(runtime_log, LOG_ERR,  "Can not bind to %s:%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
      fprintf(stderr, "Can not bind to %s:%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
      
      /* Force to quit if we have time out */
      quit = 1;
      free(df);
      conf = (void *)captureconf;
      pthread_exit(NULL);
      return NULL;
    }

  if(recvfrom(sock, (void *)df, pktsz, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)  // This is the reference for the counter, which is different from the start of capture
    {
      multilog(runtime_log, LOG_ERR,  "Can not receive data from %s:%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
      fprintf(stderr, "Can not receive data from %s:%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
      
      /* Force to quit if we have time out */
      quit = 1;      
      free(df);
      conf = (void *)captureconf;
      pthread_exit(NULL);
      return NULL;
    }
  hdr_keys(df, &hdr0[ithread]);               // Get header information, which will be used to get the location of packets
  fprintf(stdout, "%d\t%"PRIu64"\t%"PRIu64"\n", ithread, hdr0[ithread].sec, hdr0[ithread].idf);

  while(!quit)
    {
      if(recvfrom(sock, (void *)df, pktsz, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
	{
	  multilog(runtime_log, LOG_ERR,  "Can not receive data from %s:%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
	  fprintf(stderr, "Can not receive data from %s:%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);

	  /* Force to quit if we have time out */
	  quit = 1;	  
	  free(df);
	  conf = (void *)captureconf;
	  pthread_exit(NULL);
	  return NULL;
	}
      hdr_keys(df, &hdr);               // Get header information, which will be used to get the location of packets
      
      clock_gettime(CLOCK_REALTIME, &start);
      pthread_mutex_lock(&hdr_current_mutex[ithread]);
      hdr_current[ithread] = hdr;
      pthread_mutex_unlock(&hdr_current_mutex[ithread]);
      clock_gettime(CLOCK_REALTIME, &stop);
      elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
      
      pthread_mutex_lock(&hdr_ref_mutex[ithread]);
      acquire_idf(hdr.idf, hdr.sec, hdr_ref[ithread].idf, hdr_ref[ithread].sec, captureconf->sec_prd, captureconf->ndf_chk_prd, &idf);	
      pthread_mutex_unlock(&hdr_ref_mutex[ithread]);
      
      if(acquire_ichk(hdr.freq, captureconf->center_freq, captureconf->nchan_chk, captureconf->nchk, &ichk))
      	{	  
      	  multilog(runtime_log, LOG_ERR,  "Frequency chunk index < 0 || > %d, which happens at \"%s\", line [%d], has to abort.\n", MCHK_CAPTURE,__FILE__, __LINE__);
      	  fprintf(stderr, "Frequency chunk index < 0 || > %d, which happens at \"%s\", line [%d], has to abort.\n", MCHK_CAPTURE,__FILE__, __LINE__);
      
      	  /* Force to quit if we have time out */
	  quit = 1;
      	  free(df);
      	  conf = (void *)captureconf;
      	  pthread_exit(NULL);
      	  return NULL;
      	}
      
      if(idf < 0) // discard the data frame if it does not fit to current buffer
	continue;
      else
      	{
	  if(idf >= captureconf->rbuf_ndf_chk) // Start the buffer block change
	    {
	      transit[ithread] = 1;
	      
	      if(idf >= (captureconf->rbuf_ndf_chk + captureconf->tbuf_ndf_chk)) // Discard the packet which does not fit to temp buffer
		continue;
	      else   // Put data into temp buffer
		{
		  tail[ithread] = (uint64_t)((idf - captureconf->rbuf_ndf_chk) * captureconf->nchk + ichk); // This is in TFTFP order
		  tbuf_loc      = (uint64_t)(tail[ithread] * (required_pktsz + 1));
		  
		  tail[ithread]++;  // Otherwise we will miss the last available data frame in tbuf;
		  
		  tbuf[tbuf_loc] = 'Y';
		  memcpy(tbuf + tbuf_loc + 1, df + pktoff, required_pktsz);
		  
		  pthread_mutex_lock(&ndf_port_mutex[ithread]);
		  ndf_port[ithread]++;
		  pthread_mutex_unlock(&ndf_port_mutex[ithread]);
		}
	    }
	  else  // Put data into current ring buffer block
	    {
	      transit[ithread] = 0;
	      
	      // Put data into current ring buffer block if it is before rbuf_ndf_chk;
	      //cbuf_loc = (uint64_t)((idf + ichk * captureconf->rbuf_ndf_chk) * required_pktsz);   // This should give us FTTFP (FTFP) order
	      cbuf_loc = (uint64_t)((idf * captureconf->nchk + ichk) * required_pktsz); // This is in TFTFP order
	      memcpy(cbuf + cbuf_loc, df + pktoff, required_pktsz);
	      
	      pthread_mutex_lock(&ndf_port_mutex[ithread]);
	      ndf_port[ithread]++;
	      pthread_mutex_unlock(&ndf_port_mutex[ithread]);
	    }
	}
    }
    
  /* Exit */
  free(df);
  conf = (void *)captureconf;
  close(sock);
  pthread_exit(NULL);

  return NULL;
}

int init_capture(conf_t *conf)
{
  int i;

  init_buf(conf);  // Initi ring buffer
  
  /* Init status */
  for(i = 0; i < conf->nport_active; i++)
    {
      hdr_ref[i].sec = conf->sec_ref;
      hdr_ref[i].idf = conf->idf_ref;
    }
    
  /* Get the buffer block ready */
  uint64_t block_id = 0;
  uint64_t write_blkid;
  cbuf = ipcbuf_get_next_write((ipcbuf_t*)conf->hdu->data_block);
  if(cbuf == NULL)
    {	     
      multilog(runtime_log, LOG_ERR, "open_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "open_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  conf->df_res   = (double)conf->sec_prd/(double)conf->ndf_chk_prd;
  conf->blk_res  = conf->df_res * (double)conf->rbuf_ndf_chk;

  conf->nchan_chk = conf->nchan/conf->nchk;
  
  return EXIT_SUCCESS;
}

int acquire_ichk(double freq, double center_freq, int nchan_chk, int nchk, int *ichk)
{
  *ichk = (int)((freq - center_freq + 0.5)/nchan_chk + nchk/2);

  if ((*ichk < 0) || (*ichk >= nchk))
    {
      quit = 1;
      return EXIT_FAILURE;
    }
  
  return EXIT_SUCCESS;
}

int acquire_idf(uint64_t idf, uint64_t sec, uint64_t idf_ref, uint64_t sec_ref, int sec_prd, uint64_t ndf_chk_prd, int64_t *idf_buf)
{
  *idf_buf = (int64_t)(idf - idf_ref) + (int64_t)(sec - sec_ref) / sec_prd * ndf_chk_prd;
  return EXIT_SUCCESS;
}

int destroy_capture(conf_t conf)
{
  int i;
  
  pthread_mutex_destroy(&ithread_mutex);
  for(i = 0; i < MPORT_CAPTURE; i++)
    {
      pthread_mutex_destroy(&hdr_ref_mutex[i]);
      pthread_mutex_destroy(&hdr_current_mutex[i]);
      pthread_mutex_destroy(&ndf_port_mutex[i]);
    }

  dada_hdu_unlock_write(conf.hdu);
  dada_hdu_disconnect(conf.hdu);
  dada_hdu_destroy(conf.hdu);
  
  return EXIT_SUCCESS;
}
