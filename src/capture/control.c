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
//extern double elapsed_time;

extern int quit;

extern uint64_t ndf_port[MPORT_CAPTURE];
extern uint64_t ndf_chk[MCHK_CAPTURE];
extern int64_t ndf_chk_delay[MCHK_CAPTURE];

extern int transit[MPORT_CAPTURE];
extern uint64_t tail[MPORT_CAPTURE];

extern hdr_t hdr_ref[MPORT_CAPTURE];

extern pthread_mutex_t hdr_ref_mutex[MPORT_CAPTURE];
extern pthread_mutex_t ndf_port_mutex[MPORT_CAPTURE];

int threads(conf_t *conf)
{
  int i, ret[MPORT_CAPTURE + 2];
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
	  //ret[i] = pthread_create(&thread[i], &attr, capture, (void *)conf);
	  ret[i] = pthread_create(&thread[i], &attr, capture, (void *)&conf_thread[i]);
	  pthread_attr_destroy(&attr);
	}
      else
	//ret[i] = pthread_create(&thread[i], NULL, capture, (void *)conf);
	ret[i] = pthread_create(&thread[i], &attr, capture, (void *)&conf_thread[i]);
    }

  if(!(conf->cpu_bind == 0)) 
    {      
      conf_thread[i] = *conf;
      conf_thread[i].ithread = i;
      
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
  
  for(i = 0; i < nport_alive + 1; i++)   // Join threads and unbind cpus
    pthread_join(thread[i], NULL);
  if(conf->cpt_ctrl)
    pthread_join(thread[nport_alive + 1], NULL);
    
  return EXIT_SUCCESS;
}

void *buf_control(void *conf)
{
  /* 
     The software does not quit when sometime too many temp packets?
     Stuck at the buffer get?
   */
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
      multilog(runtime_log, LOG_INFO,  "BUF CONTROL CHANGE:\t0");
      if(quit)
	{
	  multilog(runtime_log, LOG_INFO,  "BUF CONTROL QUIT:\t0...");
	  
	  pthread_exit(NULL);
	  return NULL;
	}
      
      /* Check the traffic of previous buffer cycle */
      rbuf_nblk = ipcbuf_get_write_count(captureconf->db_data) + 1;
      ndf_blk_expect = 0;
      ndf_blk_actual = 0;
      for(i = 0; i < captureconf->nport_alive; i++)
	{
	  pthread_mutex_lock(&ndf_port_mutex[i]); 
	  ndf_blk_actual += ndf_port[i];
	  ndf_port[i] = 0; 
	  pthread_mutex_unlock(&ndf_port_mutex[i]);
	}
      ndf_actual += ndf_blk_actual;
      if(rbuf_nblk==1)
	{
	  for(i = 0; i < captureconf->nport_alive; i++)
	    ndf_blk_expect += (captureconf->rbuf_ndf_chk - ndf_chk_delay[i]) * captureconf->nchk_alive_actual[i];
	}
      else
	ndf_blk_expect += captureconf->rbuf_ndf_chk * captureconf->nchk_alive; // Only for current buffer
      ndf_expect += ndf_blk_expect;
      //multilog(runtime_log, LOG_INFO,  "%s\t%d\t%f\t%E\t%E\t%E\n", captureconf->ip_alive[0], captureconf->port_alive[0], rbuf_nblk * captureconf->blk_res, (1.0 - ndf_actual/(double)ndf_expect), (1.0 - ndf_blk_actual/(double)ndf_blk_expect), elapsed_time);
      multilog(runtime_log, LOG_INFO,  "%s\t%d\t%f\t%E\t%E\n", captureconf->ip_alive[0], captureconf->port_alive[0], rbuf_nblk * captureconf->blk_res, (1.0 - ndf_actual/(double)ndf_expect), (1.0 - ndf_blk_actual/(double)ndf_blk_expect));
      
      fprintf(stdout, "%f %f %f\n", rbuf_nblk * captureconf->blk_res, fabs(1.0 - ndf_actual/(double)ndf_expect), fabs(1.0 - ndf_blk_actual/(double)ndf_blk_expect));
      fflush(stdout);
	  
      /* Close current buffer */
      if(ipcbuf_mark_filled(captureconf->db_data, captureconf->rbufsz) < 0)
	{
	  multilog(runtime_log, LOG_ERR, "close_buffer failed, has to abort.\n");
	  fprintf(stderr, "close_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

	  multilog(runtime_log, LOG_INFO,  "BUF CONTROL QUIT:\t1...");
	  
	  quit = 1;
	  pthread_exit(NULL);
	  return NULL;
	}
      
      /*
	To see if the buffer is full, quit if yes.
	If we have a reader, there will be at least one buffer which is not full
      */
      if(ipcbuf_get_nfull(captureconf->db_data) >= (ipcbuf_get_nbufs(captureconf->db_data) - 1)) 
	{	     
	  multilog(runtime_log, LOG_ERR, "buffers are all full, has to abort.\n");
	  fprintf(stderr, "buffers are all full, which happens at \"%s\", line [%d], has to abort..\n", __FILE__, __LINE__);
	  multilog(runtime_log, LOG_INFO,  "BUF CONTROL QUIT:\t2...");
	  
	  quit = 1;
	  pthread_exit(NULL);
	  return NULL;
	}
      
      /* Get new buffer block */
      cbuf = ipcbuf_get_next_write(captureconf->db_data); 
      if(cbuf == NULL)
	{
	  multilog(runtime_log, LOG_ERR, "open_buffer failed, has to abort.\n");
	  fprintf(stderr, "open_buffer failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	  multilog(runtime_log, LOG_INFO,  "BUF CONTROL QUIT:\t3...");
	  
	  quit = 1;
	  pthread_exit(NULL);
	  return NULL; 
	}
      
      /* Update reference point */
      for(i = 0; i < captureconf->nport_alive; i++)
	{
	  // Update the reference hdr, once capture thread get the updated reference, the data will go to the next block or be dropped;
	  // We have to put a lock here as partial update of reference hdr will be a trouble to other threads;
	  
	  pthread_mutex_lock(&hdr_ref_mutex[i]);
	  hdr_ref[i].idf_prd += captureconf->rbuf_ndf_chk;
	  if(hdr_ref[i].idf_prd >= captureconf->ndf_chk_prd)       
	    {
	      hdr_ref[i].sec += captureconf->prd;
	      hdr_ref[i].idf_prd -= captureconf->ndf_chk_prd;
	    }
	  pthread_mutex_unlock(&hdr_ref_mutex[i]);
	}
      
      /* To see if we need to copy data from temp buffer into ring buffer */
      multilog(runtime_log, LOG_INFO,  "BUF CONTROL CHANGE:\t1");
      while(transited && (!quit))
	{
	  transited = transit[0];
	  for(i = 1; i < captureconf->nport_alive; i++)
	    //transited = transited || transit[i]; // all happen, take action
	    transited = transited && transit[i]; // one happens, take action
	}      
      if(quit)
	{
	  multilog(runtime_log, LOG_INFO,  "BUF CONTROL QUIT:\t4...");
	  pthread_exit(NULL);
	  return NULL;
	}
      multilog(runtime_log, LOG_INFO,  "BUF CONTROL CHANGE:\t2");

      ntail = 0;
      for(i = 0; i < captureconf->nport_alive; i++)
	ntail = (tail[i] > ntail) ? tail[i] : ntail;
      
#ifdef DEBUG
      fprintf(stdout, "Temp copy:\t%"PRIu64" positions need to be checked.\n", ntail);
#endif
      
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
      multilog(runtime_log, LOG_ERR, "close_buffer: ipcbuf_mark_filled failed, has to abort.\n");
      fprintf(stderr, "close_buffer: ipcio_close_block failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      quit = 1;
      
      multilog(runtime_log, LOG_INFO,  "BUF CONTROL QUIT:\t5...");
      pthread_exit(NULL);
      return NULL;
    }
  
  multilog(runtime_log, LOG_INFO,  "BUF CONTROL QUIT:\t6...");
  pthread_exit(NULL);
  return NULL;
}


void *capture_control(void *conf)
{  
  int sock, i;
  struct sockaddr_un sa, fromsa;
  socklen_t fromlen;
  conf_t *captureconf = (conf_t *)conf;
  char command_line[MSTR_LEN], command[MSTR_LEN];
  uint64_t start_buf;
  double sec_offset; // Offset from the reference time;
  uint64_t picoseconds_offset; // The sec_offset fraction part in picoseconds
  int msg_len;
  
  ///* Create an unix socket for control */
  //if((sock = socket(AF_UNIX, SOCK_STREAM, 0)) == -1)
  //  {
  //    multilog(runtime_log, LOG_ERR, "Can not create file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    fprintf (stderr, "Can not create file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
  //    
  //    quit = 1;
  //    pthread_exit(NULL);
  //    return NULL;
  //  }
  //
  //setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&captureconf->tout, sizeof(captureconf->tout));
  //memset(&sa, 0, sizeof(struct sockaddr_un));
  //sa.sun_family = AF_UNIX;
  //strncpy(&sa.sun_path[1], captureconf->cpt_ctrl_addr, strlen(captureconf->cpt_ctrl_addr));

  /* Create an unix socket for control */
  if((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1)
    {
      multilog(runtime_log, LOG_ERR, "Can not create file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Can not create file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      quit = 1;
      pthread_exit(NULL);
      return NULL;
    }
  
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&captureconf->tout, sizeof(captureconf->tout));
  memset(&sa, 0, sizeof(struct sockaddr_un));
  sa.sun_family = AF_UNIX;
  strncpy(sa.sun_path, captureconf->cpt_ctrl_addr, strlen(captureconf->cpt_ctrl_addr));
  
  multilog(runtime_log, LOG_INFO, "%s\n", captureconf->cpt_ctrl_addr);
  unlink(captureconf->cpt_ctrl_addr);
  
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
      msg_len = 0;
      while(!(msg_len > 0) && !quit)
	msg_len = recvfrom(sock, (void *)command_line, MSTR_LEN, 0, (struct sockaddr*)&fromsa, &fromlen);      
      if(quit)
	{	  
	  close(sock);
	  pthread_exit(NULL);
	  return NULL;
	}
      
      if(strstr(command_line, "END-OF-CAPTURE") != NULL)
	{	      
	  multilog(runtime_log, LOG_INFO, "Got END-OF-CAPTURE signal, has to quit.\n");
	  
	  quit = 1;	      
	  if(ipcbuf_is_writing(captureconf->db_data))
	    ipcbuf_enable_eod(captureconf->db_data);
	  
	  close(sock);
	  pthread_exit(NULL);
	  return NULL;
	}  
      if(strstr(command_line, "END-OF-DATA") != NULL)
	{
	  multilog(runtime_log, LOG_INFO, "Got END-OF-DATA signal, has to enable eod.\n");
	  
	  ipcbuf_enable_eod(captureconf->db_data);
	}
	  
      if(strstr(command_line, "START-OF-DATA") != NULL)
	{
	  multilog(runtime_log, LOG_INFO, "Got START-OF-DATA signal, has to enable sod.\n");
	  
	  sscanf(command_line, "%[^:]:%[^:]:%[^:]:%[^:]:%"SCNu64"", command, captureconf->source, captureconf->ra, captureconf->dec, &start_buf); // Read the start buffer from socket or get the minimum number from the buffer, we keep starting at the begining of buffer block;
	  start_buf = (start_buf > ipcbuf_get_write_count(captureconf->db_data)) ? start_buf : ipcbuf_get_write_count(captureconf->db_data); // To make sure the start buffer is valuable, to get the most recent buffer
	  
	  multilog(runtime_log, LOG_INFO, "The data is enabled at %"PRIu64" buffer block.\n", start_buf);
	  	  
	  ipcbuf_enable_sod(captureconf->db_data, start_buf, 0);
	  
	  /* To get time stamp for current header */
	  sec_offset = start_buf * captureconf->blk_res; // Only work with buffer number
	  picoseconds_offset = 1E6 * (round(1.0E6 * (sec_offset - floor(sec_offset))));
	  
	  captureconf->picoseconds = picoseconds_offset +captureconf->ref.picoseconds;
	  captureconf->sec_int = captureconf->ref.sec_int + floor(sec_offset);
	  if(!(captureconf->picoseconds < 1E12))
	    {
	      captureconf->sec_int += 1;
	      captureconf->picoseconds -= 1E12;
	    }
	  strftime (captureconf->utc_start, MSTR_LEN, DADA_TIMESTR, gmtime(&captureconf->sec_int)); // String start time without fraction second 
	  captureconf->mjd_start = captureconf->sec_int / SECDAY + MJD1970;                         // Float MJD start time without fraction second
	  for(i = 0; i < MSTR_LEN; i++)
	    {
	      if(captureconf->ra[i] == ' ')
		captureconf->ra[i] = ':';
	      if(captureconf->dec[i] == ' ')
		captureconf->dec[i] = ':';
	    }
	  
	  /*
	    To see if the buffer is full, quit if yes.
	    If we have a reader, there will be at least one buffer which is not full
	  */
	  if(ipcbuf_get_nfull(captureconf->db_hdr) >= (ipcbuf_get_nbufs(captureconf->db_hdr) - 1)) 
	    {	     
	      multilog(runtime_log, LOG_ERR, "buffers are all full, has to abort.\n");
	      fprintf(stderr, "buffers are all full, which happens at \"%s\", line [%d], has to abort..\n", __FILE__, __LINE__);
	      
	      quit = 1;
	      pthread_exit(NULL);
	      return NULL;
	    }
	  
	  /* setup dada header here */
	  if(dada_header(*captureconf))
	    {		  
	      quit = 1;
	      close(sock);
	      pthread_exit(NULL);
	      return NULL;
	    }
	}
    }
  
  quit = 1;
  close(sock);
  pthread_exit(NULL);
  return NULL;
}
