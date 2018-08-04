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

extern multilog_t *runtime_log;
extern int quit;
extern pthread_mutex_t quit_mutex;

void *monitor_thread(void *conf)
{  
  int sock;
  struct sockaddr_un sa, fromsa;
  socklen_t fromlen;
  conf_t *captureconf = (conf_t *)conf;
  struct timeval tout={1, 0};  // Force to timeout if we could not receive data frames in 1 second;
  char command[MSTR_LEN];
  int quit_status;
  uint64_t start_byte, start_buf;
  ipcbuf_t *db = NULL;
    
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
	      pthread_mutex_lock(&quit_mutex);
	      quit = 1;
	      pthread_mutex_unlock(&quit_mutex);
	      
	      close(sock);
	      pthread_exit(NULL);
	      return NULL;
	    }	  
	  if(strstr(command, "END-OF-DATA") != NULL)
	    //ipcio_stop(db);
	    //ipcbuf_enable_eod(db);
	    ipcbuf_disable_sod(db);
	  
	  if(strstr(command, "START-OF-DATA") != NULL)
	    {
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
