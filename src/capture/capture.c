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

extern multilog_t *runtime_log;

char *cbuf = NULL;
char *tbuf = NULL;

int quit = 0;
//double elapsed_time = 0.0;

uint64_t ndf_port[MPORT_CAPTURE] = {0};
uint64_t ndf_chk[MCHK_CAPTURE] = {0};
int64_t ndf_chk_delay[MCHK_CAPTURE] = {0};

int transit[MPORT_CAPTURE] = {0};
uint64_t tail[MPORT_CAPTURE] = {0};
hdr_t hdr_ref[MPORT_CAPTURE];

pthread_mutex_t hdr_ref_mutex[MPORT_CAPTURE]  = {PTHREAD_MUTEX_INITIALIZER};
pthread_mutex_t ndf_port_mutex[MPORT_CAPTURE] = {PTHREAD_MUTEX_INITIALIZER};

int init_buf(conf_t *conf)
{
  int i;

  /* Create HDU and check the size of buffer bolck */
  conf->required_dfsz = conf->dfsz - conf->dfoff;
  
  conf->rbufsz = conf->nchk * conf->required_dfsz * conf->rbuf_ndf_chk;  // The required buffer block size in byte;
  conf->hdu = dada_hdu_create(runtime_log);
  dada_hdu_set_key(conf->hdu, conf->key);
  if(dada_hdu_connect(conf->hdu) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Can not connect to hdu, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Can not connect to hdu, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
      for(i = 0; i < MPORT_CAPTURE; i++)
	{
	  pthread_mutex_destroy(&hdr_ref_mutex[i]);
	  pthread_mutex_destroy(&ndf_port_mutex[i]);
	}
      
      dada_hdu_destroy(conf->hdu);
      return EXIT_FAILURE;    
    }
  
  if(dada_hdu_lock_write(conf->hdu) < 0) // make ourselves the write client
    {
      multilog(runtime_log, LOG_ERR, "Error locking HDU, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error locking HDU, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
  
      for(i = 0; i < MPORT_CAPTURE; i++)
	{
	  pthread_mutex_destroy(&hdr_ref_mutex[i]);
	  pthread_mutex_destroy(&ndf_port_mutex[i]);
	}
      
      dada_hdu_disconnect(conf->hdu);
      return EXIT_FAILURE;
    }
  conf->db_data = (ipcbuf_t *)(conf->hdu->data_block);
  conf->db_hdr  = (ipcbuf_t *)(conf->hdu->header_block);
  
  if(conf->rbufsz != ipcbuf_get_bufsz(conf->db_data))  // Check the buffer size
    {
      multilog(runtime_log, LOG_ERR, "Buffer size mismatch, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Buffer size mismatch, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
        
      for(i = 0; i < MPORT_CAPTURE; i++)
	{
	  pthread_mutex_destroy(&hdr_ref_mutex[i]);
	  pthread_mutex_destroy(&ndf_port_mutex[i]);
	}
      
      dada_hdu_unlock_write(conf->hdu);
      return EXIT_FAILURE;    
    }

  if(conf->cpt_ctrl)
    {
      if(ipcbuf_disable_sod(conf->db_data) < 0)
	{
	  multilog(runtime_log, LOG_ERR, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  fprintf(stderr, "Can not write data before start, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  
	  for(i = 0; i < MPORT_CAPTURE; i++)
	    {
	      pthread_mutex_destroy(&hdr_ref_mutex[i]);
	      pthread_mutex_destroy(&ndf_port_mutex[i]);
	    }
      
	  dada_hdu_unlock_write(conf->hdu);
	  return EXIT_FAILURE;
	}
    }
  else
    {
      for(i = 0; i < MSTR_LEN; i++)
	{
	  if(conf->ra[i] == ' ')
	    conf->ra[i] = ':';
	  if(conf->dec[i] == ' ')
	    conf->dec[i] = ':';
	}
      conf->sec_int     = conf->ref.sec_int;
      conf->picoseconds = conf->ref.picoseconds;
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
      multilog(runtime_log, LOG_ERR, "open_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "open_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
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
  register int nchan_chk = captureconf->nchan_chk;
  register double ichk0 = captureconf->ichk0;
  register double df_res = captureconf->df_res;
  struct timespec start, now;
  uint64_t writebuf, *ptr = NULL;

  register int dfsz = captureconf->dfsz;
  register int dfoff = captureconf->dfoff;
  register int required_dfsz = captureconf->required_dfsz;
  register uint64_t rbuf_ndf_chk = captureconf->rbuf_ndf_chk;
  register uint64_t rbuf_tbuf_ndf_chk = captureconf->rbuf_ndf_chk + captureconf->tbuf_ndf_chk;
  register int ithread = captureconf->ithread;
  
  df = (char *)malloc(sizeof(char) * dfsz);

  multilog(runtime_log, LOG_INFO, "In funtion \nthread id = %ld, %d, %d\n", (long)pthread_self(), captureconf->ithread, ithread);
  
  sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&captureconf->tout, sizeof(captureconf->tout));  
  memset(&sa, 0x00, sizeof(sa));
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(captureconf->port_alive[ithread]);
  sa.sin_addr.s_addr = inet_addr(captureconf->ip_alive[ithread]);
  
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

  /*
    If we start the recvfrom and memcpy simultaneously at the very beginning, it will push the cpu usage to 100%;
    it does not come down sometimes will cause the capture reports significant packet loss rate;
    Here assume that the reference information is about one buffer block behind current data frame;
    the software will receive data frame for a while without writing any data into buffer;
    which will make sure that the cpu usage does not hit 100% at the beginning;

    NOTE: FOR FUTURE PIPELINE DESIGN, WE MUST ASIGN THE REFERNECE INFORMATION TO A FUTURE VALUE.
   */
  while(idf_blk<0)
    {
      if(recvfrom(sock, (void *)df, dfsz, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
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
      
      ptr = (uint64_t*)df;
      writebuf = bswap_64(*ptr);
      idf_prd = writebuf & 0x00000000ffffffff;
      df_sec = (writebuf & 0x3fffffff00000000) >> 32;
      writebuf = bswap_64(*(ptr + 2));
      freq = (double)((writebuf & 0x00000000ffff0000) >> 16);
      
      pthread_mutex_lock(&hdr_ref_mutex[ithread]);
      idf_blk = (int64_t)(idf_prd - hdr_ref[ithread].idf_prd) + ((double)df_sec - (double)hdr_ref[ithread].sec) / df_res;
      pthread_mutex_unlock(&hdr_ref_mutex[ithread]);
    }    
  ndf_chk_delay[ithread] = idf_blk;
  ichk = (int)(freq/nchan_chk + ichk0);
  
  multilog(runtime_log, LOG_INFO, "%s\t%d\t%d\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRId64"\t%"PRId64"\n\n\n", captureconf->ip_alive[ithread], captureconf->port_alive[ithread], ithread, idf_prd, hdr_ref[ithread].idf_prd, df_sec, hdr_ref[ithread].sec, idf_prd, ndf_chk_delay[ithread]);

  //clock_gettime(CLOCK_REALTIME, &start);
  while(!quit)
    {
      if(idf_blk>=0)
	{
	  if(idf_blk < rbuf_ndf_chk)  // Put data into current ring buffer block
	    {
	      transit[ithread] = 0; // The reference is already updated.
	      
	      /* Put data into current ring buffer block if it is before rbuf_ndf_chk; */
	      //cbuf_loc = (uint64_t)((idf + ichk * rbuf_ndf_chk) * required_dfsz);   // This should give us FTTFP (FTFP) order
	      cbuf_loc = (uint64_t)((idf_blk * nchk + ichk) * required_dfsz); // This is in TFTFP order
	      memcpy(cbuf + cbuf_loc, df + dfoff, required_dfsz);
	      
	      pthread_mutex_lock(&ndf_port_mutex[ithread]);
	      ndf_port[ithread]++;
	      pthread_mutex_unlock(&ndf_port_mutex[ithread]);
	    }
	  else
	    {
	      transit[ithread] = 1; // The reference should be updated very soon
	      if(idf_blk < rbuf_tbuf_ndf_chk)
		{		  
		  tail[ithread] = (uint64_t)((idf_blk - rbuf_ndf_chk) * nchk + ichk); // This is in TFTFP order
		  tbuf_loc      = (uint64_t)(tail[ithread] * (required_dfsz + 1));
		  
		  tail[ithread]++;  // Otherwise we will miss the last available data frame in tbuf;
		  
		  tbuf[tbuf_loc] = 'Y';
		  memcpy(tbuf + tbuf_loc + 1, df + dfoff, required_dfsz);
		  
		  pthread_mutex_lock(&ndf_port_mutex[ithread]);
		  ndf_port[ithread]++;
		  pthread_mutex_unlock(&ndf_port_mutex[ithread]);
		}
	      
	    }
	}
      else
	transit[ithread] = 0;
      
      //clock_gettime(CLOCK_REALTIME, &now);
      //elapsed_time += (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec)/1.0E9L;
      //start        = now;
      
      if(recvfrom(sock, (void *)df, dfsz, 0, (struct sockaddr *)&fromsa, &fromlen) == -1)
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

      //clock_gettime(CLOCK_REALTIME, &start);

      /* Get header information from bmf packet */    
      ptr = (uint64_t*)df;
      writebuf = bswap_64(*ptr);
      idf_prd = writebuf & 0x00000000ffffffff;
      df_sec = (writebuf & 0x3fffffff00000000) >> 32;
      writebuf = bswap_64(*(ptr + 2));
      freq = (double)((writebuf & 0x00000000ffff0000) >> 16);
  
      pthread_mutex_lock(&hdr_ref_mutex[ithread]);
      idf_blk = (int64_t)(idf_prd - hdr_ref[ithread].idf_prd) + ((double)df_sec - (double)hdr_ref[ithread].sec) / df_res;
      pthread_mutex_unlock(&hdr_ref_mutex[ithread]);

      ichk = (int)(freq/nchan_chk + ichk0);
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
      hdr_ref[i].sec = conf->ref.sec;
      hdr_ref[i].idf_prd = conf->ref.idf_prd + conf->rbuf_ndf_chk;
      //hdr_ref[i].idf_prd = conf->ref.idf_prd + conf->rbuf_ndf_chk; // To wait the cpu calm down before we record the data;
      conf->nchk       += conf->nchk_alive_expect[i];
      conf->nchk_alive += conf->nchk_alive_actual[i];
    }
  for(i = 0; i < conf->nport_dead; i++)
    conf->nchk       += conf->nchk_dead[i];
  
  if(conf->pad == 1)
    conf->nchk = 48;
  conf->df_res       = (double)conf->prd/(double)conf->ndf_chk_prd;
  conf->blk_res      = conf->df_res * (double)conf->rbuf_ndf_chk;
  conf->nchan        = conf->nchk * conf->nchan_chk;
  conf->ichk0        = -(conf->cfreq + 1.0)/conf->nchan_chk + 0.5 * conf->nchk;
  
  conf->ref.sec_int     = floor(conf->df_res * conf->ref.idf_prd) + conf->ref.sec + SECDAY * conf->ref.epoch;
  conf->ref.picoseconds = 1E6 * round(1.0E6 * (conf->prd - floor(conf->df_res * conf->ref.idf_prd)));

  conf->tout.tv_sec     = conf->prd;
  conf->tout.tv_usec    = 0;
  
  init_buf(conf);  // Initi ring buffer, must be here;
  
  return EXIT_SUCCESS;
}

int destroy_capture(conf_t conf)
{
  int i;
  
  for(i = 0; i < MPORT_CAPTURE; i++)
    {
      pthread_mutex_destroy(&hdr_ref_mutex[i]);
      pthread_mutex_destroy(&ndf_port_mutex[i]);
    }

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
      multilog(runtime_log, LOG_ERR, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if(!conf.hfname)
    {
      multilog(runtime_log, LOG_ERR, "Please specify header file, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Please specify header file, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }  
  if(fileread(conf.hfname, hdrbuf, DADA_HDRSZ) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Error reading header file, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error reading header file, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  /* Setup DADA header with given values */
  if(ascii_header_set(hdrbuf, "UTC_START", "%s", conf.utc_start) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "Error setting UTC_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting UTC_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_set(hdrbuf, "RA", "%s", conf.ra) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "Error setting RA, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting RA, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_set(hdrbuf, "DEC", "%s", conf.dec) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "Error setting DEC, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting DEC, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_set(hdrbuf, "SOURCE", "%s", conf.source) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "Error setting SOURCE, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting SOURCE, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_set(hdrbuf, "INSTRUMENT", "%s", conf.instrument) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "Error setting INSTRUMENT, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting INSTRUMENT, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  
  if(ascii_header_set(hdrbuf, "PICOSECONDS", "%"PRIu64, conf.picoseconds) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "Error setting PICOSECONDS, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting PICOSECONDS, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }    
  if(ascii_header_set(hdrbuf, "FREQ", "%.6lf", conf.cfreq) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Error setting FREQ, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting FREQ, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if(ascii_header_set(hdrbuf, "MJD_START", "%.10lf", conf.mjd_start) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Error setting MJD_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting MJD_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  if(ascii_header_set(hdrbuf, "NCHAN", "%d", conf.nchan) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Error setting NCHAN, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting NCHAN, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }  
  if(ascii_header_get(hdrbuf, "RESOLUTION", "%lf", &conf.chan_res) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Error getting RESOLUTION, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting RESOLUTION, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  conf.bw = conf.chan_res * conf.nchan;
  if(ascii_header_set(hdrbuf, "BW", "%.6lf", conf.bw) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Error setting BW, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting BW, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  /* donot set header parameters anymore - acqn. doesn't start */
  if(ipcbuf_mark_filled(conf.db_hdr, DADA_HDRSZ) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Error header_fill, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error header_fill, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;	      
}
