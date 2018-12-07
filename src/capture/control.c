#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdbool.h>
#include <sched.h>
#include <math.h>
#include <sys/socket.h>
#include <linux/un.h>

#include "multilog.h"
#include "ipcbuf.h"
#include "capture.h"

extern multilog_t *runtime_log;

extern char *cbuf;
extern char *tbuf;

extern int quit;

extern uint64_t ndf_port[MPORT_CAPTURE];

extern int transit[MPORT_CAPTURE];
extern uint64_t tail[MPORT_CAPTURE];

extern hdr_t hdr0[MPORT_CAPTURE];
extern hdr_t hdr_ref[MPORT_CAPTURE];
extern hdr_t hdr_current[MPORT_CAPTURE];

extern pthread_mutex_t hdr_ref_mutex[MPORT_CAPTURE];
extern pthread_mutex_t hdr_current_mutex[MPORT_CAPTURE];

extern pthread_mutex_t ndf_port_mutex[MPORT_CAPTURE];

int threads(conf_t *conf)
{
  int i, ret[MPORT_CAPTURE + 2];
  pthread_t thread[MPORT_CAPTURE + 2];
  pthread_attr_t attr;
  cpu_set_t cpus;
  int nport_active = conf->nport_active;
  
  for(i = 0; i < nport_active; i++)   // Create threads
    {
      if(!(conf->thread_bind == 0))
	{
	  pthread_attr_init(&attr);  
	  CPU_ZERO(&cpus);
	  CPU_SET(conf->port_cpu[i], &cpus);
	  pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
	  ret[i] = pthread_create(&thread[i], &attr, capture, (void *)conf);
	  pthread_attr_destroy(&attr);
	}
      else
	ret[i] = pthread_create(&thread[i], NULL, capture, (void *)conf);
    }

  if(!(conf->thread_bind == 0)) 
    {
      pthread_attr_init(&attr);
      CPU_ZERO(&cpus);
      CPU_SET(conf->buf_ctrl_cpu, &cpus);      
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);	
      ret[nport_active] = pthread_create(&thread[nport_active], &attr, buf_control, (void *)conf);
      pthread_attr_destroy(&attr);
      
      pthread_attr_init(&attr);
      CPU_ZERO(&cpus);
      CPU_SET(conf->capture_ctrl_cpu, &cpus);      
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);	
      ret[nport_active + 1] = pthread_create(&thread[nport_active + 1], &attr, capture_control, (void *)conf);
      pthread_attr_destroy(&attr);
    }
  else
    {
      ret[nport_active] = pthread_create(&thread[nport_active], NULL, buf_control, (void *)conf);
      ret[nport_active + 1] = pthread_create(&thread[nport_active + 1], NULL, capture_control, (void *)conf);
    }
  
  for(i = 0; i < nport_active + 2; i++)   // Join threads and unbind cpus
    pthread_join(thread[i], NULL);

  return EXIT_SUCCESS;
}

void *buf_control(void *conf)
{
  conf_t *captureconf = (conf_t *)conf;
  int i, nchk = captureconf->nchk, transited;
  uint64_t cbuf_loc, tbuf_loc, ntail;
  int ifreq, idf;
  ipcbuf_t *db = (ipcbuf_t *)(captureconf->hdu->data_block);
  hdr_t hdr;
  uint64_t ndf_port_expect[MPORT_CAPTURE];
  uint64_t ndf_port_actual[MPORT_CAPTURE];
  double loss_rate;
  double rbuf_time; // The duration of each ring buffer block
  uint64_t rbuf_iblk = 0;

  rbuf_time = captureconf->sec_prd * captureconf->rbuf_ndf_chk/(double)captureconf->ndf_chk_prd;
  
  while(!quit)
    {
      transited = 0;
      for(i = 0; i < captureconf->nport_active; i++)
	transited = transited || transit[i];
	      
      /* To see if we need to move to next buffer block or quit */
      if(transited)                   // Get new buffer when we see transit
	{
	  rbuf_iblk++;
	  loss_rate = 0;
	  for(i = 0; i < captureconf->nport_active; i++)
	    {
	      pthread_mutex_lock(&hdr_current_mutex[i]);
	      hdr = hdr_current[i];
	      pthread_mutex_unlock(&hdr_current_mutex[i]);
	      
	      ndf_port_expect[i] = captureconf->rbuf_ndf_chk * captureconf->nchk_active_actual[i]; // Only for current buffer
	      
	      pthread_mutex_lock(&ndf_port_mutex[i]);
	      ndf_port_actual[i] = ndf_port[i];
	      ndf_port[i] = 0;  // Only for current buffer
	      pthread_mutex_unlock(&ndf_port_mutex[i]);
	      
	      fprintf(stdout, "Thread %d:\t\t%E\t%"PRIu64"\t%"PRIu64"\t%.1E\n", i, rbuf_iblk * rbuf_time, ndf_port_actual[i], ndf_port_expect[i], 1.0 - ndf_port_actual[i]/(double)ndf_port_expect[i]);
	      loss_rate += fabs((1.0 - ndf_port_actual[i]/(double)ndf_port_expect[i])/(double)captureconf->nport_active);
	    }
	  fprintf(stdout, "\n");
	  multilog(runtime_log, LOG_INFO, "PACKET LOSS RATE ON %s: is %E\n", captureconf->ip_active[0], loss_rate);

	  /* Close current buffer */
	  if(ipcbuf_mark_filled(db, captureconf->rbufsz) < 0)
	    {
	      multilog(runtime_log, LOG_ERR, "close_buffer failed, has to abort.\n");
	      fprintf(stderr, "close_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      
	      quit = 1;
	      pthread_exit(NULL);
	      return NULL;
	    }
	  if(ipcbuf_get_nfull(db) > (ipcbuf_get_nbufs(db) - 2)) // If we have a reader, there will be at least one buffer which is not full
	    {	     
	      multilog(runtime_log, LOG_ERR, "buffers are all full, has to abort.\n");
	      fprintf(stderr, "buffers are all full, which happens at \"%s\", line [%d], has to abort..\n", __FILE__, __LINE__);
	      
	      quit = 1;
	      pthread_exit(NULL);
	      return NULL;
	    }
	  
	  if(!quit)
	    cbuf = ipcbuf_get_next_write(db);
	  else
	    {
	      quit = 1;
	      pthread_exit(NULL);
	      return NULL; 
	    }
	  
	  if(cbuf == NULL)
	    {
	      multilog(runtime_log, LOG_ERR, "open_buffer failed, has to abort.\n");
	      fprintf(stderr, "open_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      quit = 1;
	      pthread_exit(NULL);
	      return NULL; 
	    }
	  
	  for(i = 0; i < captureconf->nport_active; i++)
	    {
	      // Update the reference hdr, once capture thread get the updated reference, the data will go to the next block;
	      // We have to put a lock here as partial update of reference hdr will be a trouble to other threads;
	      
	      pthread_mutex_lock(&hdr_ref_mutex[i]);
	      hdr_ref[i].idf += captureconf->rbuf_ndf_chk;
	      if(hdr_ref[i].idf >= captureconf->ndf_chk_prd)       
		{
		  hdr_ref[i].sec += captureconf->sec_prd;
		  hdr_ref[i].idf -= captureconf->ndf_chk_prd;
		}
	      pthread_mutex_unlock(&hdr_ref_mutex[i]);
	    }

	  while(!quit) // Wait until all threads are on new buffer block
	    {
	      transited = 0;
	      for(i = 0; i < captureconf->nport_active; i++)
		transited = transited||transit[i];
	      if(transited == 0)
		break;
	    }
	  
	  /* To see if we need to copy data from temp buffer into ring buffer */
	  ntail = 0;
	  for(i = 0; i < captureconf->nport_active; i++)
	    ntail = (tail[i] > ntail) ? tail[i] : ntail;
	 	  
#ifdef DEBUG
	  fprintf(stdout, "Temp copy:\t%"PRIu64" positions need to be checked.\n", ntail);
#endif
	  
	  for(i = 0; i < ntail; i++)
	    {
	      tbuf_loc = (uint64_t)(i * (captureconf->required_pktsz + 1));	      
	      if(tbuf[tbuf_loc] == 'Y')
		{		  
		  cbuf_loc = (uint64_t)(i * captureconf->required_pktsz);  // This is for the TFTFP order temp buffer copy;
		  
		  //idf = (int)(i / nchk);
		  //ifreq = i - idf * nchk;
		  //cbuf_loc = (uint64_t)(ifreq * captureconf->rbuf_ndf_chk + idf) * captureconf->required_pktsz;  // This is for the FTFP order temp buffer copy;		

		  memcpy(cbuf + cbuf_loc, tbuf + tbuf_loc + 1, captureconf->required_pktsz);		  
		  tbuf[tbuf_loc + 1] = 'N';  // Make sure that we do not copy the data later;
		  // If we do not do that, we may have too many data frames to copy later
		}
	    }
	  for(i = 0; i < captureconf->nport_active; i++)
	    tail[i] = 0;  // Reset the location of tbuf;
	}
    }
  
  /* Exit */
  if(ipcbuf_mark_filled(db, captureconf->rbufsz) < 0)
    {
      multilog(runtime_log, LOG_ERR, "close_buffer: ipcbuf_mark_filled failed, has to abort.\n");
      fprintf(stderr, "close_buffer: ipcio_close_block failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      pthread_exit(NULL);
      return NULL;
    }

  pthread_exit(NULL);
  return NULL;
}


void *capture_control(void *conf)
{  
  int sock, i;
  struct sockaddr_un sa, fromsa;
  socklen_t fromlen;
  conf_t *captureconf = (conf_t *)conf;
  struct timeval tout={1, 0};  // Force to timeout if we could not receive data frames in 1 second;
  char command_line[MSTR_LEN], command[MSTR_LEN];
  uint64_t start_byte, start_buf;
  ipcbuf_t *db = (ipcbuf_t *)captureconf->hdu->data_block;
  uint64_t ndf_port_expect[MPORT_CAPTURE];
  uint64_t ndf_port_actual[MPORT_CAPTURE];
  hdr_t hdr;
  double chan_res, bw;
  uint64_t picoseconds;
  char utc_start[MSTR_LEN];
  double mjd_start;
  char *hdrbuf = NULL;
  time_t sec_ref; // second at start
  uint64_t picoseconds_ref; // picoseconds at start
  double sec_prd;   // seconds in one data frame period
  double sec_offset; // Offset from the reference time;
  uint64_t picoseconds_offset; // The sec_offset fraction part in picoseconds
  time_t sec;
  char source[MSTR_LEN], ra[MSTR_LEN], dec[MSTR_LEN];
  
  sec_prd = captureconf->df_res * captureconf->idf_ref;
  sec_ref = floor(sec_prd) + captureconf->sec_ref + SECDAY * captureconf->epoch_ref;
  picoseconds_ref = 1E6 * round(1.0E6 * (sec_prd - floor(sec_prd)));
  
  /* Create an unix socket for control */
  if((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1)
    {
      multilog(runtime_log, LOG_ERR, "Can not create file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Can not create file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      quit = 1;
      pthread_exit(NULL);
      return NULL;
    }  
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tout, sizeof(tout));
  memset(&sa, 0, sizeof(struct sockaddr_un));
  sa.sun_family = AF_UNIX;
  snprintf(sa.sun_path, UNIX_PATH_MAX, "%s", captureconf->ctrl_addr);
  unlink(captureconf->ctrl_addr);
  fprintf(stdout, "%s\n", captureconf->ctrl_addr);
  
  if(bind(sock, (struct sockaddr*)&sa, sizeof(sa)) == -1)
    {
      multilog(runtime_log, LOG_ERR, "Can not bind to file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Can not bind to file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      quit = 1;
      close(sock);
      pthread_exit(NULL);
      return NULL;
    }

  while(!quit)
    {
      if(recvfrom(sock, (void *)command_line, MSTR_LEN, 0, (struct sockaddr*)&fromsa, &fromlen) > 0)
	{
	  if(strstr(command_line, "END-OF-CAPTURE") != NULL)
	    {	      
	      multilog(runtime_log, LOG_INFO, "Got END-OF-CAPTURE signal, has to quit.\n");
	      fprintf(stdout, "Got END-OF-CAPTURE signal, which happens at \"%s\", line [%d], has to quit.\n", __FILE__, __LINE__);

	      for(i = 0; i < captureconf->nport_active; i++)
		{
		  pthread_mutex_lock(&hdr_current_mutex[i]);
		  hdr = hdr_current[i];
		  pthread_mutex_unlock(&hdr_current_mutex[i]);

		  ndf_port_expect[i] = (uint64_t)captureconf->nchk_active_actual[i] * (captureconf->ndf_chk_prd * (hdr.sec - hdr0[i].sec) / captureconf->sec_prd + (hdr.idf - hdr0[i].idf));
		  pthread_mutex_lock(&ndf_port_mutex[i]);
		  ndf_port_actual[i] = ndf_port[i];
		  pthread_mutex_unlock(&ndf_port_mutex[i]);
		}
	      quit = 1;
	      fprintf(stdout, "\n");
	      
	      if(ipcbuf_is_writing(db))
		ipcbuf_enable_eod(db);
	      
	      close(sock);
	      pthread_exit(NULL);
	      return NULL;
	    }
	  if(strstr(command_line, "STATUS-OF-TRAFFIC") != NULL)
	    {	      
	      for(i = 0; i < captureconf->nport_active; i++)
		{
		  pthread_mutex_lock(&hdr_current_mutex[i]);
		  hdr = hdr_current[i];
		  pthread_mutex_unlock(&hdr_current_mutex[i]);

		  ndf_port_expect[i] = (uint64_t)captureconf->nchk_active_actual[i] * (captureconf->ndf_chk_prd * (hdr.sec - hdr0[i].sec) / captureconf->sec_prd + (hdr.idf - hdr0[i].idf));
		  pthread_mutex_lock(&ndf_port_mutex[i]);
		  ndf_port_actual[i] = ndf_port[i];
		  pthread_mutex_unlock(&ndf_port_mutex[i]);
		}
	    }	  
	  if(strstr(command_line, "END-OF-DATA") != NULL)
	    {
	      multilog(runtime_log, LOG_INFO, "Got END-OF-DATA signal, has to enable eod.\n");
	      fprintf(stdout, "Got END-OF-DATA signal, which happens at \"%s\", line [%d], has to enable eod.\n", __FILE__, __LINE__);

	      ipcbuf_enable_eod(db);
	      fprintf(stdout, "IPCBUF_STATE:\t%d\n", db->state);
	    }
	  
	  if(strstr(command_line, "START-OF-DATA") != NULL)
	    {
	      //fprintf(stdout, "IPCBUF_SOD, START-OF-DATA:\t%d\n", ipcbuf_sod(db));
	      fprintf(stdout, "IPCBUF_IS_WRITING, START-OF-DATA:\t%d\n", ipcbuf_is_writing(db));
	      fprintf(stdout, "IPCBUF_IS_WRITER, START-OF-DATA:\t%d\n", ipcbuf_is_writer(db));
	      
	      multilog(runtime_log, LOG_INFO, "Got START-OF-DATA signal, has to enable sod.\n");
	      fprintf(stdout, "Got START-OF-DATA signal, which happens at \"%s\", line [%d], has to enable sod.\n", __FILE__, __LINE__);

	      //sscanf(command_line, "%[^:]:%[^:]:%[^:]:%[^:]:%"SCNu64":%"SCNu64"", command, source, ra, dec, &start_buf, &start_byte); // Read the start bytes from socket or get the minimum number from the buffer
	      sscanf(command_line, "%[^:]:%[^:]:%[^:]:%[^:]:%"SCNu64"", command, source, ra, dec, &start_buf); // Read the start bytes from socket or get the minimum number from the buffer
	      start_buf = (start_buf > ipcbuf_get_write_count(db)) ? start_buf : ipcbuf_get_write_count(db); // To make sure the start bytes is valuable, to get the most recent buffer
	      fprintf(stdout, "NUMBER OF BUF\t%"PRIu64"\n", ipcbuf_get_write_count(db));
	      
	      //fprintf(stdout, "%"PRIu64"\t%"PRIu64"\n", start_buf, start_byte);
	      fprintf(stdout, "%"PRIu64"\n", start_buf);

	      //ipcbuf_enable_sod(db, start_buf, start_byte);
	      ipcbuf_enable_sod(db, start_buf, 0);
	      
	      /* To get time stamp for current header */
	      //sec_offset = start_buf * captureconf->blk_res + round(start_byte / captureconf->buf_dfsz) * captureconf->df_res;
	      sec_offset = start_buf * captureconf->blk_res; // Only work with buffer number
	      picoseconds_offset = 1E6 * (round(1.0E6 * (sec_offset - floor(sec_offset))));
	      picoseconds = picoseconds_offset + picoseconds_ref;
	      sec = sec_ref + sec_offset;
	      if(!(picoseconds < 1E12))
	      	{
	      	  sec += 1;
	      	  picoseconds -= 1E12;
	      	}
	      strftime (utc_start, MSTR_LEN, DADA_TIMESTR, gmtime(&sec)); // String start time without fraction second 
	      mjd_start = sec / SECDAY + MJD1970;                         // Float MJD start time without fraction second
	      for(i = 0; i < MSTR_LEN; i++)
		{
		  if(ra[i] == ' ')
		    ra[i] = ':';
		  if(dec[i] == ' ')
		    dec[i] = ':';
		}
	      
	      /* Register header */
	      hdrbuf = ipcbuf_get_next_write(captureconf->hdu->header_block);
	      if(!hdrbuf)
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      if(!captureconf->hfname)
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Please specify header file, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Please specify header file, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}  
	      if(fileread(captureconf->hfname, hdrbuf, DADA_HDRSZ) < 0)
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error reading header file, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error reading header file, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  
		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      
	      /* Setup DADA header with given values */
	      if(ascii_header_set(hdrbuf, "UTC_START", "%s", utc_start) < 0)  
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting UTC_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting UTC_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      
	      if(ascii_header_set(hdrbuf, "RA", "%s", ra) < 0)  
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting RA, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting RA, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      
	      if(ascii_header_set(hdrbuf, "DEC", "%s", dec) < 0)  
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting DEC, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting DEC, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      
	      if(ascii_header_set(hdrbuf, "SOURCE", "%s", source) < 0)  
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting SOURCE, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting SOURCE, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      
	      if(ascii_header_set(hdrbuf, "INSTRUMENT", "%s", captureconf->instrument) < 0)  
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting INSTRUMENT, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting INSTRUMENT, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      
	      if(ascii_header_set(hdrbuf, "PICOSECONDS", "%"PRIu64, picoseconds) < 0)  
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting PICOSECONDS, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting PICOSECONDS, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}    
	      if(ascii_header_set(hdrbuf, "FREQ", "%.6lf", captureconf->center_freq) < 0)
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting FREQ, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting FREQ, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      if(ascii_header_set(hdrbuf, "MJD_START", "%.10lf", mjd_start) < 0)
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting MJD_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting MJD_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      if(ascii_header_set(hdrbuf, "NCHAN", "%d", captureconf->nchan) < 0)
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting NCHAN, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting NCHAN, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  
		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}  
	      if(ascii_header_get(hdrbuf, "RESOLUTION", "%lf", &chan_res) < 0)
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error getting RESOLUTION, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting RESOLUTION, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      bw = chan_res * captureconf->nchan;
	      if(ascii_header_set(hdrbuf, "BW", "%.6lf", bw) < 0)
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting BW, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting BW, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      /* donot set header parameters anymore - acqn. doesn't start */
	      if(ipcbuf_mark_filled(captureconf->hdu->header_block, DADA_HDRSZ) < 0)
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error header_fill, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error header_fill, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

		  quit = 1;
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      //ipcbuf_enable_sod(db, start_buf, start_byte);
	    }
	}
    }
  
  close(sock);
  pthread_exit(NULL);
  return NULL;
}
