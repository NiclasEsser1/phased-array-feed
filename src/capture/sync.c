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
#include "sync.h"
#include "capture.h"
#include "monitor.h"

extern multilog_t *runtime_log;

extern char *cbuf;
extern char *tbuf;

extern int quit;
extern int force_switch;

extern uint64_t ndf_port[MPORT_CAPTURE];
extern uint64_t ndf_chk[MCHK_CAPTURE];

extern int transit[MPORT_CAPTURE];
extern uint64_t tail[MPORT_CAPTURE];
extern hdr_t hdr_ref[MPORT_CAPTURE];
extern hdr_t hdr_current[MPORT_CAPTURE];

extern pthread_mutex_t quit_mutex;
extern pthread_mutex_t force_switch_mutex;

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
	  ret[i] = pthread_create(&thread[i], &attr, capture_thread, (void *)conf);
	  pthread_attr_destroy(&attr);
	}
      else
	ret[i] = pthread_create(&thread[i], NULL, capture_thread, (void *)conf);
    }

  if(!(conf->thread_bind == 0))
    {
      pthread_attr_init(&attr);
      CPU_ZERO(&cpus);
      CPU_SET(conf->sync_cpu, &cpus);      
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);	
      ret[nport_active] = pthread_create(&thread[nport_active], &attr, sync_thread, (void *)conf);
      pthread_attr_destroy(&attr);
      
      pthread_attr_init(&attr);
      CPU_ZERO(&cpus);
      CPU_SET(conf->monitor_cpu, &cpus);      
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);	
      ret[nport_active + 1] = pthread_create(&thread[nport_active + 1], &attr, monitor_thread, (void *)conf);
      pthread_attr_destroy(&attr);
    }
  else
    {
      ret[nport_active] = pthread_create(&thread[nport_active], NULL, sync_thread, (void *)conf);
      ret[nport_active + 1] = pthread_create(&thread[nport_active + 1], NULL, monitor_thread, (void *)conf);
    }
  
  //if(!(conf->thread_bind == 0))
  //  {
  //    pthread_attr_init(&attr);
  //    CPU_ZERO(&cpus);
  //    CPU_SET(conf->monitor_cpu, &cpus);      
  //    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);	
  //    ret[nport_active + 1] = pthread_create(&thread[nport_active + 1], &attr, monitor_thread, (void *)conf);
  //    pthread_attr_destroy(&attr);
  //  }
  //else
  //  ret[nport_active + 1] = pthread_create(&thread[nport_active + 1], NULL, monitor_thread, (void *)conf);
  
  for(i = 0; i < nport_active + 2; i++)   // Join threads and unbind cpus
    pthread_join(thread[i], NULL);

  return EXIT_SUCCESS;
}

void *sync_thread(void *conf)
{
  conf_t *captureconf = (conf_t *)conf;
  int i, nchunk = captureconf->nchunk, ntransit;
  uint64_t cbuf_loc, tbuf_loc, ntail;
  int ifreq, idf;
  uint64_t block_id = 0;
  struct timespec ref_time, current_time;
  hdr_t hdr;
  uint64_t ndf_port_expect[MPORT_CAPTURE];
  uint64_t ndf_port_actual[MPORT_CAPTURE];
  uint64_t ndf_chk_actual[MCHK_CAPTURE];
  uint64_t ndf_chk_expect[MCHK_CAPTURE];
  int force_switch_status, quit_status;
  
  pthread_mutex_lock(&quit_mutex);
  quit_status = quit;
  pthread_mutex_unlock(&quit_mutex);

  clock_gettime(CLOCK_REALTIME, &ref_time);
  while(quit_status == 0)
    {      
      clock_gettime(CLOCK_REALTIME, &current_time);
      if((current_time.tv_sec - ref_time.tv_sec) > captureconf->monitor_sec) // Check the traffic status every monitor_sec;
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
	  ref_time = current_time;
	}
	
      ntransit = 0; 
      for(i = 0; i < captureconf->nport_active; i++)
	{
	  pthread_mutex_lock(&transit_mutex[i]);
	  ntransit += transit[i];
	  pthread_mutex_unlock(&transit_mutex[i]);
	}
      
      /* To see if we need to move to next buffer block */
      pthread_mutex_lock(&force_switch_mutex);
      force_switch_status = force_switch;
      pthread_mutex_unlock(&force_switch_mutex);      
      if((ntransit > nchunk) || force_switch_status)                   // Once we have more than active_links data frames on temp buffer, we will move to new ring buffer block
	{
	  /* Close current buffer */
	  if(ipcio_close_block_write(captureconf->hdu->data_block, captureconf->rbufsz) < 0)
	    {
	      multilog (runtime_log, LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
	      fprintf(stderr, "close_buffer: ipcio_close_block_write failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      
	      pthread_mutex_lock(&quit_mutex);
	      quit = 1;
	      pthread_mutex_unlock(&quit_mutex);

	      pthread_exit(NULL);
	      return NULL;
	    }

	  cbuf = ipcio_open_block_write(captureconf->hdu->data_block, &block_id);
	  
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
	  
	  while(true) // Wait until all threads are on new buffer block
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
	  
	  pthread_mutex_lock(&force_switch_mutex);
	  force_switch = 0;
	  pthread_mutex_unlock(&force_switch_mutex);
	}
      
      pthread_mutex_lock(&quit_mutex);
      quit_status = quit;
      pthread_mutex_unlock(&quit_mutex);
    }
  
  /* Exit */
  if (ipcio_close_block_write (captureconf->hdu->data_block, captureconf->rbufsz) < 0)  // This should enable eod at current buffer
    {
      multilog (runtime_log, LOG_ERR, "close_buffer: ipcio_close_block_write failed\n");
      fprintf(stderr, "close_buffer: ipcio_close_block_write failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);

      pthread_exit(NULL);
      return NULL;
    }
  
  pthread_exit(NULL);
  return NULL;
}
