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
#include "hdr.h"
#include "sync.h"

extern multilog_t *runtime_log;

char *cbuf = NULL;
char *tbuf = NULL;

int quit;
int force_switch;
int ithread_extern;

int transit[MPORT_CAPTURE];
uint64_t tail[MPORT_CAPTURE];
hdr_t hdr_ref[MPORT_CAPTURE];

pthread_mutex_t ithread_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutexattr_t ithread_mutex_attr;

pthread_mutex_t quit_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutexattr_t quit_mutex_attr;

pthread_mutex_t force_switch_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutexattr_t force_switch_mutex_attr;

pthread_mutex_t hdr_ref_mutex[MPORT_CAPTURE] = {PTHREAD_MUTEX_INITIALIZER};
pthread_mutexattr_t hdr_ref_mutex_attr[MPORT_CAPTURE];

int init_buf(conf_t *conf)
{
  int i, nbufs;
  ipcbuf_t *db = NULL;

  /* Create HDU and check the size of buffer bolck */
  conf->required_pktsz = conf->pktsz - conf->pktoff;
  conf->rbufsz = conf->nchunk * conf->required_pktsz * conf->rbuf_ndf_chk;  // The required buffer block size in byte;
  conf->hdu = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu, conf->key);
  if(dada_hdu_connect(conf->hdu) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Can not connect to hdu, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  db = (ipcbuf_t *) conf->hdu->data_block;
  if(conf->rbufsz != ipcbuf_get_bufsz((ipcbuf_t *)db))  // Check the buffer size
    {
      multilog(runtime_log, LOG_ERR, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Buffer size mismatch, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;    
    }
  if (dada_hdu_lock_write (conf->hdu) < 0) // make ourselves the write client 
    {
      multilog (runtime_log, LOG_ERR, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if(ipcbuf_disable_sod((ipcbuf_t *)db) < 0)  // Disable SOD by default
    {
      multilog (runtime_log, LOG_ERR, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  conf->tbufsz = (conf->required_pktsz + 1) * conf->tbuf_ndf_chk * conf->nchunk;
  tbuf = (char *)malloc(conf->tbufsz * sizeof(char));// init temp buffer
  
  ///* Register header */
  //char *hdrbuf = NULL;
  //hdrbuf = ipcbuf_get_next_write (conf->hdu->header_block);
  //if(!hdrbuf)
  //  {
  //    multilog(runtime_log, LOG_ERR, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    return EXIT_FAILURE;
  //  }
  //if(!conf->hdr_fname)
  //  {
  //    multilog(runtime_log, LOG_ERR, "Please specify header file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    fprintf(stderr, "Please specify header file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    return EXIT_FAILURE;
  //  }  
  //if(fileread(conf->hdr_fname, hdrbuf, DADA_HDR_SIZE) < 0)
  //  {
  //    multilog(runtime_log, LOG_ERR, "Error reading header file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    fprintf(stderr, "Error reading header file, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    return EXIT_FAILURE;
  //  }
  //
  ///* Setup DADA header with given values */
  //if(ascii_header_set(hdrbuf, "UTC_START", "%s", conf->utc_start) < 0)  
  //  {
  //    multilog(runtime_log, LOG_ERR, "Error setting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    fprintf(stderr, "Error setting UTC_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    return EXIT_FAILURE;
  //  }
  //if(ascii_header_set(hdrbuf, "INSTRUMENT", "%s", conf->instrument) < 0)  
  //  {
  //    multilog(runtime_log, LOG_ERR, "Error setting INSTRUMENT, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    fprintf(stderr, "Error setting INSTRUMENT, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    return EXIT_FAILURE;
  //  }
  //if(ascii_header_set(hdrbuf, "PICOSECONDS", "%"PRIu64, conf->picoseconds) < 0)  
  //  {
  //    multilog(runtime_log, LOG_ERR, "Error setting PICOSECONDS, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    fprintf(stderr, "Error setting PICOSECONDS, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    return EXIT_FAILURE;
  //  }    
  //if(ascii_header_set(hdrbuf, "FREQ", "%.1lf", conf->center_freq) < 0)
  //  {
  //    multilog(runtime_log, LOG_ERR, "Error setting FREQ, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    fprintf(stderr, "Error setting FREQ, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    return EXIT_FAILURE;
  //  }
  //if(ascii_header_set(hdrbuf, "MJD_START", "%lf", conf->mjd_start) < 0)
  //  {
  //    multilog(runtime_log, LOG_ERR, "Error setting MJD_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    fprintf(stderr, "Error setting MJD_START, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    return EXIT_FAILURE;
  //  }
  //if(ascii_header_set(hdrbuf, "NCHAN", "%d", conf->nchan) < 0)
  //  {
  //    multilog(runtime_log, LOG_ERR, "Error setting NCHAN, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    fprintf(stderr, "Error setting NCHAN, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    return EXIT_FAILURE;
  //  }  
  //if(ascii_header_get(hdrbuf, "RESOLUTION", "%lf", &conf->resolution) < 0)
  //  {
  //    multilog(runtime_log, LOG_ERR, "Error getting RESOLUTION, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    fprintf(stderr, "Error setting RESOLUTION, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    return EXIT_FAILURE;
  //  }
  //conf->bw = conf->resolution * conf->nchan;
  //if(ascii_header_set(hdrbuf, "BW", "%.1lf", conf->bw) < 0)
  //  {
  //    multilog(runtime_log, LOG_ERR, "Error setting BW, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    fprintf(stderr, "Error setting BW, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    return EXIT_FAILURE;
  //  }
  ///* donot set header parameters anymore - acqn. doesn't start */
  //if (ipcbuf_mark_filled (conf->hdu->header_block, DADA_HDR_SIZE) < 0)
  //  {
  //    multilog(runtime_log, LOG_ERR, "Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    fprintf(stderr, "Error header_fill, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    return EXIT_FAILURE;
  //  }
  //
  return EXIT_SUCCESS;
}

void *capture_thread(void *conf)
{
  char *df = NULL;
  conf_t *captureconf = (conf_t *)conf;
  int sock, ithread, pktsz, pktoff, required_pktsz, ifreq; 
  struct sockaddr_in sa;
  struct timeval time_out={captureconf->sec_prd, 0};  // Force to timeout if we could not receive data frames for one period.
  socklen_t fromlen = sizeof(sa);
  int64_t idf;
  uint64_t tbuf_loc, cbuf_loc;
  hdr_t hdr;
  
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
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&time_out, sizeof(time_out));  
  memset(&sa, 0x00, sizeof(sa));
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(captureconf->port_active[ithread]);
  sa.sin_addr.s_addr = inet_addr(captureconf->ip_active[ithread]);
  bind(sock, (struct sockaddr *)&sa, sizeof(sa));

  while(true)
    {      
      if(recvfrom(sock, (void *)df, pktsz, 0, (struct sockaddr *)&sa, &fromlen) == -1)
	{
	  multilog(runtime_log, LOG_ERR,  "Can not receive data from %s:%d, which happens at \"%s\", line [%d].\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
	  fprintf (stderr, "Can not receive data from %s:%d, which happens at \"%s\", line [%d].\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);

	  /* Force to quit if we have time out */
	  pthread_mutex_lock(&quit_mutex);
	  quit = 1;
	  pthread_mutex_unlock(&quit_mutex);

	  conf = (void *)captureconf;
	  pthread_exit(NULL);
	  return NULL;
	}      
      hdr_keys(df, &hdr);               // Get header information, which will be used to get the location of packets
      acquire_ifreq(hdr, *captureconf, &ifreq);              
      
      pthread_mutex_lock(&hdr_ref_mutex[ithread]);
      acquire_idf(hdr, hdr_ref[ithread], *captureconf, &idf);  // How many data frames we get after the reference;
      pthread_mutex_unlock(&hdr_ref_mutex[ithread]);

      if (idf < 0 )
	// Drop data frams which are behind time;
	continue;
      else
	{
	  if(idf >= captureconf->rbuf_ndf_chk)
	    {
	      /*
		Means we can not put the data into current ring buffer block anymore and we have to use temp buffer;
		If the number of chunks we used on temp buffer is equal to active chunks, we have to move to a new ring buffer block;
		If the temp buffer is too small, we may lose packets;
		If the temp buffer is too big, we will waste time to copy the data from it to ring buffer;
	      
		The above is the original plan, but it is too strict that we may stuck on some point;
		THE NEW PLAN IS we count the data frames which are not recorded with ring buffer, later than rbuf_ndf_chk;
		If the counter is bigger than the active links, we active a new ring buffer block;
		Meanwhile, before reference hdr is updated, the data frames can still be put into temp buffer if it is not behind rbuf_ndf_chk + tbuf_ndf_chk;
		The data frames which are behind the limit will have to be dropped;
		the reference hdr update follows tightly and then sleep for about 1 millisecond to wait all capture threads move to new ring buffer block;
		at this point, no data frames will be put into temp buffer anymore and we are safe to copy data from it into the new ring buffer block and reset the temp buffer;
		If the data frames are later than rbuf_ndf_chk + tbuf_ndf_chk, we force the swtich of ring buffer blocks;
		The new plan will drop couple of data frames every ring buffer block, but it guarentee that we do not stuck on some point;
		We force to quit the capture if we do not get any data in one block;
		We also foce to quit if we have time out problem;
	      */
	      transit[ithread]++;
	     	      
	      if(idf >= (2 * captureconf->rbuf_ndf_chk)) // quit
		{
		  /* Force to quit if we do not get any data in one data block */
		  pthread_mutex_lock(&quit_mutex);
		  quit = 1;
		  pthread_mutex_unlock(&quit_mutex);
		  		  
#ifdef DEBUG
		  fprintf(stdout, "Too many temp data frames:\t%d\t%d\t%d\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRId64"\n", ithread, ntohs(sa.sin_port), ifreq, hdr_ref[ithread].sec, hdr_ref[ithread].idf, hdr.sec, hdr.idf, idf);
#endif
		  conf = (void *)captureconf;
		  pthread_exit(NULL);
		  return NULL;
		}
	      else if(((idf >= (captureconf->rbuf_ndf_chk + captureconf->tbuf_ndf_chk)) && (idf < (2 * captureconf->rbuf_ndf_chk))))   // Force to get a new ring buffer block
		{
		  /* 
		     One possibility here: if we lose more that RBUF_NDF data frames continually, we will miss one data block;
		     for rbuf_ndf_chk = 12500, that will be about 1 second data;
		     Do we need to deal with it?
		     I force the thread quit and also tell other threads quit if we loss one buffer;
		  */
#ifdef DEBUG
		  fprintf(stdout, "Forced force_switch %d\t%"PRIu64"\t%"PRIu64"\t%d\t%"PRIu64"\n", ithread, hdr.sec, hdr.idf, ifreq, idf);
#endif
		  pthread_mutex_lock(&force_switch_mutex);
		  force_switch = 1;
		  pthread_mutex_unlock(&force_switch_mutex);
		}
	      else  // Put data in to temp buffer
		{
		  tail[ithread] = (uint64_t)((idf - captureconf->rbuf_ndf_chk) * captureconf->nchunk + ifreq); // This is in TFTFP order
		  tbuf_loc      = (uint64_t)(tail[ithread] * (required_pktsz + 1));
		  tail[ithread]++;  // Otherwise we will miss the last available data frame in tbuf;
		  
		  tbuf[tbuf_loc] = 'Y';
		  memcpy(tbuf + tbuf_loc + 1, df + pktoff, required_pktsz);	  
		  captureconf->ndf_port[ithread]++;
		  captureconf->ndf_chan[ifreq]++;
		}
	    }
	  else
	    {
	      transit[ithread] = 0;
	      // Put data into current ring buffer block if it is before rbuf_ndf_chk;
	      cbuf_loc = (uint64_t)((idf * captureconf->nchunk + ifreq) * required_pktsz); // This is in TFTFP order
	      //cbuf_loc = (uint64_t)((idf + ifreq * captureconf->rbuf_ndf_chk) * required_pktsz);   // This should give us FTTFP (FTFP) order
	      memcpy(cbuf + cbuf_loc, df + pktoff, required_pktsz);	      
	      captureconf->ndf_port[ithread]++;
	      captureconf->ndf_chan[ifreq]++;
	    }
	}  
    }
  free(df);
    
  /* Exit */
  conf = (void *)captureconf;
  pthread_exit(NULL);
  return NULL;
}

int init_capture(conf_t *conf)
{
  int i;

  init_buf(conf);  // Initi ring buffer and setup DADA header

  for(i = 0; i < conf->nchan; i++) // Setup the counter for each frequency
    conf->ndf_chan[i] = 0;
  
  /* Init status */
  for(i = 0; i < conf->nport_active; i++)
    {
      transit[i] = 0;
      tail[i] = 0;

      conf->ndf_port[i] = 0;
      hdr_ref[i].sec = conf->sec_start;
      hdr_ref[i].idf = conf->idf_start;
    }
  force_switch = 0;
  quit = 0;
  
  /* Initialise mutex */
  pthread_mutexattr_init(&ithread_mutex_attr);
  pthread_mutexattr_settype(&ithread_mutex_attr, PTHREAD_PROCESS_SHARED);
  pthread_mutexattr_init(&quit_mutex_attr);
  pthread_mutexattr_settype(&quit_mutex_attr, PTHREAD_PROCESS_SHARED);
  pthread_mutex_init(&ithread_mutex, &ithread_mutex_attr);
  pthread_mutexattr_init(&force_switch_mutex_attr);
  pthread_mutexattr_settype(&force_switch_mutex_attr, PTHREAD_PROCESS_SHARED);
  for(i = 0; i < conf->nport_active; i++)
    {      
      pthread_mutexattr_init(&hdr_ref_mutex_attr[i]);
      pthread_mutexattr_settype(&hdr_ref_mutex_attr[i], PTHREAD_PROCESS_SHARED);
      pthread_mutex_init(&hdr_ref_mutex[i], &hdr_ref_mutex_attr[i]);
    }
  
  /* Get the buffer block ready */
  uint64_t block_id = 0;
  cbuf = ipcio_open_block_write(conf->hdu->data_block, &block_id);
  
  return EXIT_SUCCESS;
}

int acquire_ifreq(hdr_t hdr, conf_t conf, int *ifreq)
{  
  *ifreq = (int)((hdr.freq - conf.center_freq - 0.5)/(conf.nchan/conf.nchunk) + conf.nchunk/2);
  return EXIT_SUCCESS;
}

int acquire_idf(hdr_t hdr, hdr_t hdr_ref, conf_t conf, int64_t *idf)
{
  *idf = (int64_t)hdr.idf + (int64_t)(hdr.sec - hdr_ref.sec) / conf.sec_prd * conf.ndf_chk_prd - (int64_t)hdr_ref.idf;
  return EXIT_SUCCESS;
}
