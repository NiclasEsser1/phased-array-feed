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

int quit;
int force_next;
int ithread_extern;

uint64_t ndf_port[MPORT_CAPTURE];
uint64_t ndf_chk[MCHK_CAPTURE];

int transit[MPORT_CAPTURE];
uint64_t tail[MPORT_CAPTURE];
hdr_t hdr0[MPORT_CAPTURE]; // It is the reference for packet counter, we do not need mutex lock for this
hdr_t hdr_ref[MPORT_CAPTURE];
hdr_t hdr_current[MPORT_CAPTURE];

pthread_mutex_t quit_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t force_next_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t ithread_mutex = PTHREAD_MUTEX_INITIALIZER;

pthread_mutex_t ndf_port_mutex[MPORT_CAPTURE] = {PTHREAD_MUTEX_INITIALIZER};
pthread_mutex_t ndf_chk_mutex[MCHK_CAPTURE] = {PTHREAD_MUTEX_INITIALIZER};

pthread_mutex_t hdr_ref_mutex[MPORT_CAPTURE] = {PTHREAD_MUTEX_INITIALIZER};
pthread_mutex_t hdr_current_mutex[MPORT_CAPTURE] = {PTHREAD_MUTEX_INITIALIZER};

pthread_mutex_t transit_mutex[MPORT_CAPTURE] = {PTHREAD_MUTEX_INITIALIZER};

int init_buf(conf_t *conf)
{
  int i, nbufs;
  ipcbuf_t *db = NULL;

  /* Create HDU and check the size of buffer block */
  conf->required_pktsz = conf->pktsz - conf->pktoff;

  conf->rbufsz = conf->nchk * conf->required_pktsz * conf->rbuf_ndf_chk;  // The required buffer block size in byte;
  conf->hdu = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu, conf->key);
  if(dada_hdu_connect(conf->hdu) < 0)
  {
    multilog(runtime_log, LOG_ERR, "Can not connect to hdu, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

    pthread_mutex_destroy(&ithread_mutex);
    pthread_mutex_destroy(&quit_mutex);
    pthread_mutex_destroy(&force_next_mutex);
    for(i = 0; i < MPORT_CAPTURE; i++)
		{
		  pthread_mutex_destroy(&hdr_ref_mutex[i]);
		  pthread_mutex_destroy(&hdr_current_mutex[i]);
		  pthread_mutex_destroy(&transit_mutex[i]);
		  pthread_mutex_destroy(&ndf_port_mutex[i]);
		}

    for(i = 0; i < MCHK_CAPTURE; i++)
			pthread_mutex_destroy(&ndf_chk_mutex[i]);

    dada_hdu_destroy(conf->hdu);
    return EXIT_FAILURE;
  }

  if(dada_hdu_lock_write(conf->hdu) < 0) // make ourselves the write client
    {
      multilog(runtime_log, LOG_ERR, "Error locking HDU, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      pthread_mutex_destroy(&ithread_mutex);
      pthread_mutex_destroy(&quit_mutex);
      pthread_mutex_destroy(&force_next_mutex);
      for(i = 0; i < MPORT_CAPTURE; i++)
	{
	  pthread_mutex_destroy(&hdr_ref_mutex[i]);
	  pthread_mutex_destroy(&hdr_current_mutex[i]);
	  pthread_mutex_destroy(&transit_mutex[i]);
	  pthread_mutex_destroy(&ndf_port_mutex[i]);
	}

      for(i = 0; i < MCHK_CAPTURE; i++)
	pthread_mutex_destroy(&ndf_chk_mutex[i]);

      dada_hdu_disconnect(conf->hdu);
      return EXIT_FAILURE;
    }

  db = (ipcbuf_t *)conf->hdu->data_block;
  if(conf->rbufsz != ipcbuf_get_bufsz(db))  // Check the buffer size
    {
      multilog(runtime_log, LOG_ERR, "Buffer size mismatch, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      pthread_mutex_destroy(&ithread_mutex);
      pthread_mutex_destroy(&quit_mutex);
      pthread_mutex_destroy(&force_next_mutex);
      for(i = 0; i < MPORT_CAPTURE; i++)
	{
	  pthread_mutex_destroy(&hdr_ref_mutex[i]);
	  pthread_mutex_destroy(&hdr_current_mutex[i]);
	  pthread_mutex_destroy(&transit_mutex[i]);
	  pthread_mutex_destroy(&ndf_port_mutex[i]);
	}

      for(i = 0; i < MCHK_CAPTURE; i++)
	pthread_mutex_destroy(&ndf_chk_mutex[i]);

      dada_hdu_unlock_write(conf->hdu);
      return EXIT_FAILURE;
    }
  if(ipcbuf_disable_sod(db) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      pthread_mutex_destroy(&ithread_mutex);
      pthread_mutex_destroy(&quit_mutex);
      pthread_mutex_destroy(&force_next_mutex);
      for(i = 0; i < MPORT_CAPTURE; i++)
  	{
  	  pthread_mutex_destroy(&hdr_ref_mutex[i]);
  	  pthread_mutex_destroy(&hdr_current_mutex[i]);
  	  pthread_mutex_destroy(&transit_mutex[i]);
  	  pthread_mutex_destroy(&ndf_port_mutex[i]);
  	}

      for(i = 0; i < MCHK_CAPTURE; i++)
  	pthread_mutex_destroy(&ndf_chk_mutex[i]);

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
  conf_t *captureconf = (conf_t *)conf;	// This cast and assignment is not necessary, why not using the directly passed variable conf?
  int sock;			// Variable holding the socket fstream
	int ithread; 	// Thread ID of the calling thread
	int pktsz = captureconf->pktsz;					// Size of a UDP packet in bytes
	int pktoff = captureconf->pktoff;					// Offset, can be used to skip data in bytes; usually set to 0
	int required_pktsz = captureconf->required_pktsz;	// The total number
	int ichk;			// Channel chunk identifier
  struct sockaddr_in sa;	// POSIX socket server
	struct sockaddr_in fromsa; // Assumed to be the POSIX socket client. If so this is not necessary, since the server itself could be used
  struct timeval tout={captureconf->sec_prd, 0};  // Force to timeout if we could not receive data frames for one period.
  socklen_t fromlen;	// = sizeof(fromsa);
  int64_t idf;	// Index of dataframe
  uint64_t tbuf_loc;	// Tail buffer; Seems to be the index/location of intermediate buffer
	uint64_t cbuf_loc;	// Index pointing to the correct location of the ringbuffer
  hdr_t hdr;	// Header
  int quit_status;	// Local quit status of capturing thread. Allows to carry out the current thread task, while another thread already reached an failure point.

	// Seems variable below are just used for debbuing purpose

	// ifdef DEBUG added by Niclas Esser
#ifdef DEBUG
  double elapsed_time;
  struct timespec start, stop;
#endif

  init_hdr(&hdr);	// Initializes the hdr_t struct by setting all attributes to 0

	// Allocate memory for a dataframe.
  df = (char *)malloc(sizeof(char) * pktsz);

  /* Get right socket for current thread */
	// Thread ID assigned from global variable, Accessed by all threads but mutexed
  pthread_mutex_lock(&ithread_mutex);
  ithread = ithread_extern;
  ithread_extern++;
  pthread_mutex_unlock(&ithread_mutex);

	// Create POSIX UDP socket
  sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	// Set a receiving timeout. If timeout happens sock will return partial count or errno set to EAGAIN or EWOULDBLOCK
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tout, sizeof(tout));
  memset(&sa, 0x00, sizeof(sa));	// Guess this is not necessary
  sa.sin_family      = AF_INET;	// Assign socket family
  sa.sin_port        = htons(captureconf->port_active[ithread]); // Assign the port for the thread socket and translates it to POSIX number
  sa.sin_addr.s_addr = inet_addr(captureconf->ip_active[ithread]); // Assign IP address for the thread socket.

	// Create a server by binding the socket to the specified socket_t struct
  if(bind(sock, (struct sockaddr *)&sa, sizeof(sa)) == -1)
  {
    multilog(runtime_log, LOG_ERR,  "Can not bind to %s:%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
    fprintf(stderr, "Can not bind to %s:%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);

    /* Force to quit if we have time out */
    pthread_mutex_lock(&quit_mutex);
    quit = 1;
    pthread_mutex_unlock(&quit_mutex);

    free(df);
    conf = (void *)captureconf;
    pthread_exit(NULL);
    return NULL;
  }

	// This is the reference for the counter, which is different from the start of capture
	// The first packet receive is here applied, while all others are happening in the while loop below. Just to check if a thread socket is receiving or not. If any of the threads does not receive, capturing will fail!
  if(recvfrom(sock, (void *)df, pktsz, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
  {
    multilog(runtime_log, LOG_ERR,  "Can not receive data from %s:%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
    fprintf(stderr, "Can not receive data from %s:%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);

    /* Force to quit if we have time out */
    pthread_mutex_lock(&quit_mutex);
    quit = 1;
    pthread_mutex_unlock(&quit_mutex);

    free(df);
    conf = (void *)captureconf;
    pthread_exit(NULL);
    return NULL;
  }
	// Get header information, which will be used to get the location of packets
  hdr_keys(df, &hdr0[ithread]);

  fprintf(stdout, "%d\t%"PRIu64"\t%"PRIu64"\n", ithread, hdr0[ithread].sec, hdr0[ithread].idf);

	// State of the thread
	// If quit is set to 1 by one of any threads, the program will be aborted.
	// Assigning quit to local variable allows every thread to finish the current work, before the abort.
  pthread_mutex_lock(&quit_mutex);
  quit_status = quit;
  pthread_mutex_unlock(&quit_mutex);

	// Aslong as quit_status is 0 proceed with capturing
  while(quit_status == 0)
  {
		// Thread socket wants to receive a UDP packet
    if(recvfrom(sock, (void *)df, pktsz, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
		{
		  multilog(runtime_log, LOG_ERR,  "Can not receive data from %s:%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);
		  fprintf(stderr, "Can not receive data from %s:%d, which happens at \"%s\", line [%d], has to abort.\n", inet_ntoa(sa.sin_addr), ntohs(sa.sin_port), __FILE__, __LINE__);

		  /* Force to quit if we have time out */
		  pthread_mutex_lock(&quit_mutex);
		  quit = 1;
		  pthread_mutex_unlock(&quit_mutex);

		  free(df);
		  conf = (void *)captureconf;
		  pthread_exit(NULL);
		  return NULL;
		}

		// Get header information, which will be used to get the location of packets
    hdr_keys(df, &hdr);

// ifdef DEBUG added by Niclas Esser
#ifdef DEBUG
		// Tracking the memory copy of an header
    clock_gettime(CLOCK_REALTIME, &start);
    pthread_mutex_lock(&hdr_current_mutex[ithread]);
#endif

    hdr_current[ithread] = hdr;

#ifdef DEBUG
		// Tracking the memory copy of an header
    pthread_mutex_unlock(&hdr_current_mutex[ithread]);
    clock_gettime(CLOCK_REALTIME, &stop);
    elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
    fprintf(stdout, "%E seconds used for hdr\n", elapsed_time);
#endif

		// Calculate the dataframe index of received packet
    pthread_mutex_lock(&hdr_ref_mutex[ithread]);
    acquire_idf(hdr.idf, hdr.sec, hdr_ref[ithread].idf, hdr_ref[ithread].sec, captureconf->sec_prd, captureconf->ndf_chk_prd, &idf);
    pthread_mutex_unlock(&hdr_ref_mutex[ithread]);

		// Check if the acquired packet has the correct number of channels and get chunk index
    if(acquire_ichk(hdr.freq, captureconf->center_freq, captureconf->nchan_chk, captureconf->nchk, &ichk))
  	{
  	  multilog(runtime_log, LOG_ERR,  "Frequency chunk index < 0 || > %d, which happens at \"%s\", line [%d], has to abort.\n", MCHK_CAPTURE,__FILE__, __LINE__);
  	  fprintf(stderr, "Frequency chunk index < 0 || > %d, which happens at \"%s\", line [%d], has to abort.\n", MCHK_CAPTURE,__FILE__, __LINE__);

  	  /* Force to quit if we have time out */
  	  pthread_mutex_lock(&quit_mutex);
  	  quit = 1;
  	  pthread_mutex_unlock(&quit_mutex);

  	  free(df);
  	  conf = (void *)captureconf;
  	  pthread_exit(NULL);
  	  return NULL;
  	}

    if(idf >= 0)
  	{
		  // Drop data frams which are behind time;
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
	      pthread_mutex_lock(&transit_mutex[ithread]);
	      transit[ithread]++;
	      pthread_mutex_unlock(&transit_mutex[ithread]);

				/* Force to quit if we do not get any data in one data block */
	      if(idf >= (2 * captureconf->rbuf_ndf_chk)) // quit
				{
				  pthread_mutex_lock(&quit_mutex);
				  quit = 1;
				  pthread_mutex_unlock(&quit_mutex);

#ifdef DEBUG
				  pthread_mutex_lock(&hdr_ref_mutex[ithread]);
				  fprintf(stdout, "Too many temp data frames:\t%d\t%d\t%d\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRId64"\n", ithread, ntohs(sa.sin_port), ichk, hdr_ref[ithread].sec, hdr_ref[ithread].idf, hdr.sec, hdr.idf, idf);
#endif
				  pthread_mutex_unlock(&hdr_ref_mutex[ithread]);

				  free(df);
				  conf = (void *)captureconf;
				  close(sock);

				  pthread_exit(NULL);
				  return NULL;
				}
				// Force to get a new ring buffer block
	      else if(((idf >= (captureconf->rbuf_ndf_chk + captureconf->tbuf_ndf_chk)) && (idf < (2 * captureconf->rbuf_ndf_chk))))
				{
			  /*
			     One possibility here: if we lose more that rbuf_ndf_nchk data frames continually, we will miss one data block;
			     for rbuf_ndf_chk = 12500, that will be about 1 second data;
			     Do we need to deal with it?
			     I force the thread quit and also tell other threads quit if we loss one buffer;
			  */
		#ifdef DEBUG
				  fprintf(stdout, "Forced %d\t%"PRIu64"\t%"PRIu64"\t%d\t%"PRIu64"\n", ithread, hdr.sec, hdr.idf, ichk, idf);
		#endif
				  pthread_mutex_lock(&force_next_mutex);
				  force_next = 1;
				  pthread_mutex_unlock(&force_next_mutex);
				}
	      else  // Put data in to temp buffer
				{
				  tail[ithread] = (uint64_t)((idf - captureconf->rbuf_ndf_chk) * captureconf->nchk + ichk); // This is in TFTFP order
				  tbuf_loc      = (uint64_t)(tail[ithread] * (required_pktsz + 1));

				  tail[ithread]++;  // Otherwise we will miss the last available data frame in tbuf;

				  tbuf[tbuf_loc] = 'Y';
				  memcpy(tbuf + tbuf_loc + 1, df + pktoff, required_pktsz);

				  pthread_mutex_lock(&ndf_port_mutex[ithread]);
				  ndf_port[ithread]++;
				  pthread_mutex_unlock(&ndf_port_mutex[ithread]);

				  pthread_mutex_lock(&ndf_chk_mutex[ichk]);
				  ndf_chk[ichk]++;
				  pthread_mutex_unlock(&ndf_chk_mutex[ichk]);
				}
	    }
	  	else
	    {
	      pthread_mutex_lock(&transit_mutex[ithread]);
	      transit[ithread] = 0;
	      pthread_mutex_unlock(&transit_mutex[ithread]);

	      // Put data into current ring buffer block if it is before rbuf_ndf_chk;
	      cbuf_loc = (uint64_t)((idf * captureconf->nchk + ichk) * required_pktsz); // This is in TFTFP order
	      //cbuf_loc = (uint64_t)((idf + ichk * captureconf->rbuf_ndf_chk) * required_pktsz);   // This should give us FTTFP (FTFP) order
	      memcpy(cbuf + cbuf_loc, df + pktoff, required_pktsz);

	      pthread_mutex_lock(&ndf_port_mutex[ithread]);
	      ndf_port[ithread]++;
	      pthread_mutex_unlock(&ndf_port_mutex[ithread]);

	      pthread_mutex_lock(&ndf_chk_mutex[ichk]);
	      ndf_chk[ichk]++;
	      pthread_mutex_unlock(&ndf_chk_mutex[ichk]);
	    }
		}

    pthread_mutex_lock(&quit_mutex);
    quit_status = quit;
    pthread_mutex_unlock(&quit_mutex);
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

  ithread_extern = 0;
  for(i = 0; i < conf->nchk; i++) // Setup the counter for each frequency
    ndf_chk[i] = 0;

  /* Init status */
  for(i = 0; i < conf->nport_active; i++)
  {
    transit[i] = 0;
    tail[i] = 0;

    ndf_port[i] = 0;
    hdr_ref[i].sec = conf->sec_ref;
    hdr_ref[i].idf = conf->idf_ref;
  }
  force_next = 0;
  quit = 0;	// Used to quit the program if any thread reached a failure point

  /* Get the buffer block ready */
  uint64_t block_id = 0;
  uint64_t write_blkid;
  cbuf = ipcbuf_get_next_write((ipcbuf_t*)conf->hdu->data_block);
  //cbuf = ipcio_open_block_write(conf->hdu->data_block, &write_blkid);
  if(cbuf == NULL)
    {
      multilog(runtime_log, LOG_ERR, "open_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "open_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  conf->df_res   = (double)conf->sec_prd/(double)conf->ndf_chk_prd;
  conf->blk_res  = conf->df_res * (double)conf->rbuf_ndf_chk;
  //conf->buf_dfsz = conf->required_pktsz * (double)conf->nchk;

  conf->nchan_chk = conf->nchan/conf->nchk;

  return EXIT_SUCCESS;
}

int acquire_ichk(double freq, double center_freq, int nchan_chk, int nchk, int *ichk)
{
  *ichk = (int)((freq - center_freq + 0.5)/nchan_chk + nchk/2);

  if ((*ichk < 0) || (*ichk >= MCHK_CAPTURE))
    return EXIT_FAILURE;

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
  pthread_mutex_destroy(&quit_mutex);
  pthread_mutex_destroy(&force_next_mutex);
  for(i = 0; i < MPORT_CAPTURE; i++)
    {
      pthread_mutex_destroy(&hdr_ref_mutex[i]);
      pthread_mutex_destroy(&hdr_current_mutex[i]);
      pthread_mutex_destroy(&transit_mutex[i]);
      pthread_mutex_destroy(&ndf_port_mutex[i]);
    }

  for(i = 0; i < MCHK_CAPTURE; i++)
    pthread_mutex_destroy(&ndf_chk_mutex[i]);

  dada_hdu_unlock_write(conf.hdu);
  dada_hdu_disconnect(conf.hdu);
  dada_hdu_destroy(conf.hdu);

  return EXIT_SUCCESS;
}
