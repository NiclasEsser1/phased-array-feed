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
  
  for(i = 0; i < nport_alive; i++)
    // Create threads. Capture threads and ring buffer control thread are essential, the capture control thread is created when it is required to;
    // If we create a capture thread, we can control the start and end of data during runtime, the header of DADA buffer will be setup each time we start the data, which means without rerun the pipeline, we can get multiple capture runs;
    // If we do not create a capture thread, the data will start at the begining and we need to setup the header at that time, we can only do one capture without rerun the pipeline;
    {
      if(!(conf->cpu_bind == 0))
	{
	  pthread_attr_init(&attr);  
	  CPU_ZERO(&cpus);
	  CPU_SET(conf->cpt_cpu[i], &cpus);
	  pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
	  ret[i] = pthread_create(&thread[i], &attr, capture, (void *)conf);
	  pthread_attr_destroy(&attr);
	}
      else
	ret[i] = pthread_create(&thread[i], NULL, capture, (void *)conf);
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
  
  for(i = 0; i < nport_alive + 1; i++)   // Join threads and unbind cpus
    pthread_join(thread[i], NULL);
  if(conf->cpt_ctrl)
    pthread_join(thread[nport_alive + 1], NULL);
    
  return EXIT_SUCCESS;
}

void *buf_control(void *conf)
{
  conf_t *captureconf = (conf_t *)conf;
  int i, nchk = captureconf->nchk, transited;
  uint64_t cbuf_loc, tbuf_loc, ntail;
  int ichk, idf;
  ipcbuf_t *db = (ipcbuf_t *)(captureconf->hdu->data_block);
  uint64_t ndf_port_expect;
  uint64_t ndf_port_actual;
  double loss_rate = 0;
  uint64_t rbuf_iblk = 0;
  double sleep_time;
  int sleep_sec, sleep_usec;
  
  sleep_time   = 0.9 * captureconf->blk_res; // Sleep for part of buffer block time to save cpu source;
  sleep_sec    = (int)sleep_time;
  sleep_usec   = (int)((sleep_time - sleep_sec) * 1E6);
  
  while(!quit)
    {
      transited = 0;
      for(i = 0; i < captureconf->nport_alive; i++)
	transited = transited || transit[i];
	      
      /* To see if we need to move to next buffer block or quit */
      if(transited)                   // Get new buffer when we see transit
	{
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
	  
	  for(i = 0; i < captureconf->nport_alive; i++)
	    {
	      // Update the reference hdr, once capture thread get the updated reference, the data will go to the next block;
	      // We have to put a lock here as partial update of reference hdr will be a trouble to other threads;
	      
	      pthread_mutex_lock(&hdr_ref_mutex[i]);
	      hdr_ref[i].idf += captureconf->rbuf_ndf_chk;
	      if(hdr_ref[i].idf >= captureconf->ndf_chk_prd)       
		{
		  hdr_ref[i].sec += captureconf->prd;
		  hdr_ref[i].idf -= captureconf->ndf_chk_prd;
		}
	      pthread_mutex_unlock(&hdr_ref_mutex[i]);
	    }

	  while(!quit) // Wait until all threads are on new buffer block
	    {
	      transited = 0;
	      for(i = 0; i < captureconf->nport_alive; i++)
		transited = transited||transit[i];
	      
	      if(transited == 0) 
		{
		  /* Check the traffic of previous buffer cycle */
		  rbuf_iblk++;
		  for(i = 0; i < captureconf->nport_alive; i++)
		    {
		      pthread_mutex_lock(&ndf_port_mutex[i]);
		      ndf_port_actual = ndf_port[i];
		      ndf_port[i] = 0;  // Only for current buffer
		      pthread_mutex_unlock(&ndf_port_mutex[i]);

		      if(rbuf_iblk>1) // Need to ignore the first buffer block as a delay may exist;
			{
			  ndf_port_expect = captureconf->rbuf_ndf_chk * captureconf->nchk_alive_actual[i]; // Only for current buffer
			  
			  fprintf(stdout, "Port %d,\t\t%E\t%"PRIu64"\t%"PRIu64"\t%.1E\n", captureconf->port_alive[i], rbuf_iblk * captureconf->blk_res, ndf_port_actual, ndf_port_expect, 1.0 - ndf_port_actual/(double)ndf_port_expect);
			  loss_rate += ((1.0 - ndf_port_actual/(double)ndf_port_expect)/(double)captureconf->nport_alive);
			}
		    }
		  if(rbuf_iblk>1)    // Need to ignore the first buffer block as a delay may exist;
		    {
		      fprintf(stdout, "Packet loss rate so far is %E\n", loss_rate/(double)(rbuf_iblk - 1));
		      multilog(runtime_log, LOG_INFO, "Packet loss rate so far is %E\n", loss_rate/(double)(rbuf_iblk - 1));
		    }
		  
		  /* To see if we need to copy data from temp buffer into ring buffer */
		  ntail = 0;
		  for(i = 0; i < captureconf->nport_alive; i++)
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
			  //ichk = i - idf * nchk;
			  //cbuf_loc = (uint64_t)(ichk * captureconf->rbuf_ndf_chk + idf) * captureconf->required_pktsz;  // This is for the FTFP order temp buffer copy;		
			  
			  memcpy(cbuf + cbuf_loc, tbuf + tbuf_loc + 1, captureconf->required_pktsz);		  
			  tbuf[tbuf_loc + 1] = 'N';  // Make sure that we do not copy the data later;
			  // If we do not do that, we may have too many data frames to copy later
			}
		    }
		  for(i = 0; i < captureconf->nport_alive; i++)
		    tail[i] = 0;  // Reset the location of tbuf;

		  /* Sleep to save cpu source */
		  sleep(sleep_sec);
		  usleep(sleep_usec);
		  
		  break;
		}
	    }
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
  char command_line[MSTR_LEN], command[MSTR_LEN];
  uint64_t start_byte, start_buf;
  ipcbuf_t *db = (ipcbuf_t *)captureconf->hdu->data_block;
  double sec_offset; // Offset from the reference time;
  uint64_t picoseconds_offset; // The sec_offset fraction part in picoseconds
  
  /* Create an unix socket for control */
  if((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1)
    {
      multilog(runtime_log, LOG_ERR, "Can not create file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Can not create file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      quit = 1;
      pthread_exit(NULL);
      return NULL;
    }  
  //setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tout, sizeof(tout));
  memset(&sa, 0, sizeof(struct sockaddr_un));
  sa.sun_family = AF_UNIX;
  snprintf(sa.sun_path, UNIX_PATH_MAX, "%s", captureconf->cpt_ctrl_addr);
  unlink(captureconf->cpt_ctrl_addr);
  fprintf(stdout, "%s\n", captureconf->cpt_ctrl_addr);
  
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

	      quit = 1;	      
	      if(ipcbuf_is_writing(db))
		ipcbuf_enable_eod(db);
	      
	      close(sock);
	      pthread_exit(NULL);
	      return NULL;
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
	      fprintf(stdout, "IPCBUF_IS_WRITING, START-OF-DATA:\t%d\n", ipcbuf_is_writing(db));
	      fprintf(stdout, "IPCBUF_IS_WRITER, START-OF-DATA:\t%d\n", ipcbuf_is_writer(db));
	      
	      multilog(runtime_log, LOG_INFO, "Got START-OF-DATA signal, has to enable sod.\n");
	      fprintf(stdout, "Got START-OF-DATA signal, which happens at \"%s\", line [%d], has to enable sod.\n", __FILE__, __LINE__);

	      sscanf(command_line, "%[^:]:%[^:]:%[^:]:%[^:]:%"SCNu64"", command, captureconf->source, captureconf->ra, captureconf->dec, &start_buf); // Read the start bytes from socket or get the minimum number from the buffer, we keep starting at the begining of buffer block;
	      start_buf = (start_buf > ipcbuf_get_write_count(db)) ? start_buf : ipcbuf_get_write_count(db); // To make sure the start bytes is valuable, to get the most recent buffer
	      fprintf(stdout, "NUMBER OF BUF\t%"PRIu64"\n", ipcbuf_get_write_count(db));

	      multilog(runtime_log, LOG_INFO, "The data is enabled at %"PRIu64" buffer block.\n", start_buf);
	      fprintf(stdout, "%"PRIu64"\n", start_buf);

	      ipcbuf_enable_sod(db, start_buf, 0);
	      
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
    }
  
  close(sock);
  pthread_exit(NULL);
  return NULL;
}
