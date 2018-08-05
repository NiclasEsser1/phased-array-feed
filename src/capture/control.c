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

#include "ipcbuf.h"
#include "capture.h"

extern multilog_t *runtime_log;

extern char *cbuf;
extern char *tbuf;

extern int quit;
extern int force_next;

extern uint64_t ndf_port[MPORT_CAPTURE];
extern uint64_t ndf_chk[MCHK_CAPTURE];

extern int transit[MPORT_CAPTURE];
extern uint64_t tail[MPORT_CAPTURE];
extern hdr_t hdr_ref[MPORT_CAPTURE];
extern hdr_t hdr_current[MPORT_CAPTURE];

extern pthread_mutex_t quit_mutex;
extern pthread_mutex_t force_next_mutex;

extern pthread_mutex_t hdr_ref_mutex[MPORT_CAPTURE];
extern pthread_mutex_t hdr_current_mutex[MPORT_CAPTURE];

extern pthread_mutex_t ndf_port_mutex[MPORT_CAPTURE];
extern pthread_mutex_t ndf_chk_mutex[MCHK_CAPTURE];

extern pthread_mutex_t transit_mutex[MPORT_CAPTURE];

int threads(conf_t *conf)
{
  int i, ret[MPORT_CAPTURE + 2], node;
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
      CPU_SET(conf->sync_cpu, &cpus);      
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);	
      ret[nport_active] = pthread_create(&thread[nport_active], &attr, buf_control, (void *)conf);
      pthread_attr_destroy(&attr);
      
      pthread_attr_init(&attr);
      CPU_ZERO(&cpus);
      CPU_SET(conf->monitor_cpu, &cpus);      
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
  int i, nchunk = captureconf->nchunk, ntransit;
  uint64_t cbuf_loc, tbuf_loc, ntail;
  int ifreq, idf;
  uint64_t block_id = 0;
  int force_next_status, quit_status;
  
  pthread_mutex_lock(&quit_mutex);
  quit_status = quit;
  pthread_mutex_unlock(&quit_mutex);

  while(quit_status == 0)
    {   	
      ntransit = 0; 
      for(i = 0; i < captureconf->nport_active; i++)
	{
	  pthread_mutex_lock(&transit_mutex[i]);
	  ntransit += transit[i];
	  pthread_mutex_unlock(&transit_mutex[i]);
	}
      
      /* To see if we need to move to next buffer block or quit */
      pthread_mutex_lock(&force_next_mutex);
      force_next_status = force_next;
      pthread_mutex_unlock(&force_next_mutex);
      if((ntransit > nchunk) || force_next_status)                   // Once we have more than active_links data frames on temp buffer, we will move to new ring buffer block
	{
	  /* Close current buffer */
	  if(ipcbuf_mark_filled((ipcbuf_t*)captureconf->hdu->data_block, captureconf->rbufsz) < 0) 
	    {
	      multilog(runtime_log, LOG_ERR, "close_buffer: ipcbuf_mark_filled failed, has to abort.\n");
	      fprintf(stderr, "close_buffer: ipcbuf_mark_filled failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      
	      pthread_mutex_lock(&quit_mutex);
	      quit = 1;
	      pthread_mutex_unlock(&quit_mutex);

	      pthread_exit(NULL);
	      return NULL;
	    }

	  if(ipcbuf_get_nfull((ipcbuf_t*)captureconf->hdu->data_block) == ipcbuf_get_nbufs((ipcbuf_t*)captureconf->hdu->data_block))
	    {	     
	      multilog(runtime_log, LOG_ERR, "buffers are all full, has to abort.\n");
	      fprintf(stderr, "buffers are all full, which happens at \"%s\", line [%d], has to abort..\n", __FILE__, __LINE__);
	       
	      pthread_mutex_lock(&quit_mutex);
	      quit = 1;
	      pthread_mutex_unlock(&quit_mutex);

	      pthread_exit(NULL);
	      return NULL;
	    }
	  cbuf = ipcbuf_get_next_write((ipcbuf_t*)captureconf->hdu->data_block);
	  if(cbuf == NULL)
	    {
	      multilog(runtime_log, LOG_ERR, "open_buffer: ipcbuf_get_next_write failed, has to abort.\n");
	      fprintf(stderr, "open_buffer: ipcbuf_get_next_write failed, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      
	      pthread_mutex_lock(&quit_mutex);
	      quit = 1;
	      pthread_mutex_unlock(&quit_mutex);

	      pthread_exit(NULL);
	      return NULL; 
	    }
	  
	  for(i = 0; i < captureconf->nport_active; i++)
	    {
	      // Update the reference hdr, once capture thread get the updated reference, the data will go to the next block;
	      // We have to put a lock here as partial update of reference hdr will be a trouble to other threads;
	      
	      pthread_mutex_lock(&hdr_ref_mutex[i]);
	      hdr_ref[i].idf += captureconf->rbuf_ndf_chk;
	      if(hdr_ref[i].idf >= captureconf->ndf_chk_prd)                       // Here I assume that we could not lose one period;
		{
		  hdr_ref[i].sec += captureconf->sec_prd;
		  hdr_ref[i].idf -= captureconf->ndf_chk_prd;
		}
	      pthread_mutex_unlock(&hdr_ref_mutex[i]);
	    }

	  pthread_mutex_lock(&quit_mutex);   // Need to check quit starus in while loop, otherwise it will stuck here
	  quit_status = quit;
	  pthread_mutex_unlock(&quit_mutex);
	  while(quit_status == 0) // Wait until all threads are on new buffer block
	    {
	      ntransit = 0;
	      for(i = 0; i < captureconf->nport_active; i++)
		{
		  pthread_mutex_lock(&transit_mutex[i]);
		  ntransit += transit[i];
		  pthread_mutex_unlock(&transit_mutex[i]);
		}
	      if(ntransit == 0)
		break;
	      
	      pthread_mutex_lock(&quit_mutex);   // Need to check quit starus in while loop, otherwise it will stuck here
	      quit_status = quit;
	      pthread_mutex_unlock(&quit_mutex);
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
		  //idf = (int)(i / nchunk);
		  //ifreq = i - idf * nchunk;
		  cbuf_loc = (uint64_t)(i * captureconf->required_pktsz);  // This is for the TFTFP order temp buffer copy;
		  //cbuf_loc = (uint64_t)(ifreq * captureconf-> captureconf->rbuf_ndf_chk + idf) * captureconf->required_pktsz;  // This is for the FTFP order temp buffer copy;		

		  memcpy(cbuf + cbuf_loc, tbuf + tbuf_loc + 1, captureconf->required_pktsz);		  
		  tbuf[tbuf_loc + 1] = 'N';  // Make sure that we do not copy the data later;
		  // If we do not do that, we may have too many data frames to copy later
		}
	    }
	  for(i = 0; i < captureconf->nport_active; i++)
	    tail[i] = 0;  // Reset the location of tbuf;
	  
	  pthread_mutex_lock(&force_next_mutex);
	  force_next = 0;
	  pthread_mutex_unlock(&force_next_mutex);
	}

      pthread_mutex_lock(&quit_mutex);
      quit_status = quit;
      pthread_mutex_unlock(&quit_mutex);
    }
  
  /* Exit */
  if(ipcbuf_mark_filled((ipcbuf_t*)captureconf->hdu->data_block, captureconf->rbufsz) < 0) 
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
  char command[MSTR_LEN];
  int quit_status;
  uint64_t start_byte, start_buf;
  ipcbuf_t *db = NULL;
  uint64_t ndf_port_expect[MPORT_CAPTURE];
  uint64_t ndf_port_actual[MPORT_CAPTURE];
  uint64_t ndf_chk_actual[MCHK_CAPTURE];
  uint64_t ndf_chk_expect[MCHK_CAPTURE];
  hdr_t hdr;
  
  /* Create an unix socket for control */
  if((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1)
    {
      multilog(runtime_log, LOG_ERR, "Can not create file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Can not create file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      pthread_mutex_lock(&quit_mutex);
      quit = 1;
      pthread_mutex_unlock(&quit_mutex);

      pthread_exit(NULL);
      return NULL;
    }  
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tout, sizeof(tout));
  memset(&sa, 0, sizeof(struct sockaddr_un));
  sa.sun_family = AF_UNIX;
  snprintf(sa.sun_path, UNIX_PATH_MAX, "/tmp/capture_socket");
  unlink("/tmp/capture_socket");
  
  if(bind(sock, (struct sockaddr*)&sa, sizeof(sa)) == -1)
    {
      multilog(runtime_log, LOG_ERR, "Can not bind to file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Can not bind to file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      pthread_mutex_lock(&quit_mutex);
      quit = 1;
      pthread_mutex_unlock(&quit_mutex);

      close(sock);
      pthread_exit(NULL);
      return NULL;
    }
  
  pthread_mutex_lock(&quit_mutex);
  quit_status = quit;
  pthread_mutex_unlock(&quit_mutex);
  while(quit_status == 0)
    {
      if(recvfrom(sock, (void *)command, MSTR_LEN, 0, (struct sockaddr*)&fromsa, &fromlen) > 0)
	{
	  db = (ipcbuf_t *)captureconf->hdu->data_block;
	  if(strstr(command, "END-OF-CAPTURE") != NULL)
	    {	      
	      multilog(runtime_log, LOG_INFO, "Got END-OF-CAPTURE signal, has to quit.\n");
	      fprintf(stdout, "Got END-OF-CAPTURE signal, which happens at \"%s\", line [%d], has to quit.\n", __FILE__, __LINE__);

	      for(i = 0; i < captureconf->nport_active; i++)
		{
		  pthread_mutex_lock(&hdr_current_mutex[i]);
		  hdr = hdr_current[i];
		  pthread_mutex_unlock(&hdr_current_mutex[i]);

		  ndf_port_expect[i] = (uint64_t)captureconf->nchunk_active_actual[i] * (captureconf->ndf_chk_prd * (hdr.sec - captureconf->sec_start) / captureconf->sec_prd + (hdr.idf - captureconf->idf_start));
		  pthread_mutex_lock(&ndf_port_mutex[i]);
		  ndf_port_actual[i] = ndf_port[i];
		  pthread_mutex_unlock(&ndf_port_mutex[i]);
		  
		  ndf_chk_expect[i] = (uint64_t)(captureconf->ndf_chk_prd * (hdr.sec - captureconf->sec_start) / captureconf->sec_prd + (hdr.idf - captureconf->idf_start));
		  pthread_mutex_lock(&ndf_chk_mutex[i]);
		  ndf_chk_actual[i] = ndf_chk[i];
		  pthread_mutex_unlock(&ndf_chk_mutex[i]);
		  
		  fprintf(stdout, "HERE\t%"PRIu64"\t%"PRIu64"\t%.1E\t%"PRIu64"\t%"PRIu64"\t%.1E\n", ndf_chk_actual[i], ndf_chk_expect[i], (double)(ndf_chk_expect[i])/(double)(ndf_chk_actual[i]) - 1.0, ndf_port_actual[i], ndf_port_expect[i], (double)(ndf_port_expect[i])/(double)(ndf_port_actual[i]) - 1.0);
		}
	      fprintf(stdout, "\n");
	      
	      pthread_mutex_lock(&quit_mutex);
	      quit = 1;
	      pthread_mutex_unlock(&quit_mutex);
	      
	      close(sock);
	      pthread_exit(NULL);
	      return NULL;
	    }
	  if(strstr(command, "STATUS-OF-TRAFFIC") != NULL)
	    {	      
	      for(i = 0; i < captureconf->nport_active; i++)
		{
		  pthread_mutex_lock(&hdr_current_mutex[i]);
		  hdr = hdr_current[i];
		  pthread_mutex_unlock(&hdr_current_mutex[i]);

		  ndf_port_expect[i] = (uint64_t)captureconf->nchunk_active_actual[i] * (captureconf->ndf_chk_prd * (hdr.sec - captureconf->sec_start) / captureconf->sec_prd + (hdr.idf - captureconf->idf_start));
		  pthread_mutex_lock(&ndf_port_mutex[i]);
		  ndf_port_actual[i] = ndf_port[i];
		  pthread_mutex_unlock(&ndf_port_mutex[i]);
		  
		  ndf_chk_expect[i] = (uint64_t)(captureconf->ndf_chk_prd * (hdr.sec - captureconf->sec_start) / captureconf->sec_prd + (hdr.idf - captureconf->idf_start));
		  pthread_mutex_lock(&ndf_chk_mutex[i]);
		  ndf_chk_actual[i] = ndf_chk[i];
		  pthread_mutex_unlock(&ndf_chk_mutex[i]);
		  
		  fprintf(stdout, "HERE\t%"PRIu64"\t%"PRIu64"\t%.1E\t%"PRIu64"\t%"PRIu64"\t%.1E\n", ndf_chk_actual[i], ndf_chk_expect[i], (double)(ndf_chk_expect[i])/(double)(ndf_chk_actual[i]) - 1.0, ndf_port_actual[i], ndf_port_expect[i], (double)(ndf_port_expect[i])/(double)(ndf_port_actual[i]) - 1.0);
		}
	      fprintf(stdout, "\n");
	    }	  
	  if(strstr(command, "END-OF-DATA") != NULL)
	    {
	      multilog(runtime_log, LOG_INFO, "Got END-OF-DATA signal, has to disable sod.\n");
	      fprintf(stdout, "Got END-OF-DATA signal, which happens at \"%s\", line [%d], has to disable sod.\n", __FILE__, __LINE__);

	      ipcbuf_disable_sod(db);
	    }
	  
	  if(strstr(command, "START-OF-DATA") != NULL)
	    {
	      multilog(runtime_log, LOG_INFO, "Got START-OF-DATA signal, has to enable sod.\n");
	      fprintf(stdout, "Got START-OF-DATA signal, which happens at \"%s\", line [%d], has to enable sod.\n", __FILE__, __LINE__);

	      sscanf(command, "%*s:%"SCNu64":%"SCNu64"", &start_byte, &start_buf); // Read the start bytes from socket or get the minimum number from the buffer
	      if(start_buf == 0)
		start_buf = ipcbuf_get_sod_minbuf(db);
	      else
		start_buf = (start_buf > ipcbuf_get_sod_minbuf(db)) ? start_byte :  ipcbuf_get_sod_minbuf(db); // To make sure the start bytes is valuable
	      ipcbuf_enable_sod(db, start_buf, start_byte);
	    }
	}

      pthread_mutex_lock(&quit_mutex);
      quit_status = quit;
      pthread_mutex_unlock(&quit_mutex);
    }
  
  close(sock);
  pthread_exit(NULL);
  return NULL;
}
