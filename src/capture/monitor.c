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
extern int quit;
extern pthread_mutex_t quit_mutex;
extern hdr_t hdr_current[MPORT_CAPTURE];
extern uint64_t ndf_port[MPORT_CAPTURE];
extern uint64_t ndf_chk[MCHK_CAPTURE];

extern pthread_mutex_t hdr_current_mutex[MPORT_CAPTURE];
extern pthread_mutex_t ndf_port_mutex[MPORT_CAPTURE];
extern pthread_mutex_t ndf_chk_mutex[MCHK_CAPTURE];

void *monitor_thread(void *conf)
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
