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
  struct timeval tout={1, 0};  // Force to timeout if we could not receive data frames in 1 microsecond
  char command[MSTR_LEN];
  int quit_status;
  
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
      //unlink("/tmp/capture_socket");
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
	  fprintf(stdout, "%s\n", command);
	  pthread_mutex_lock(&quit_mutex);
	  quit = 1;
	  pthread_mutex_unlock(&quit_mutex);

	  close(sock);
	  //unlink("/tmp/capture_socket");
	  pthread_exit(NULL);
	  return NULL;
	}

      pthread_mutex_lock(&quit_mutex);
      quit_status = quit;
      pthread_mutex_unlock(&quit_mutex);
    }

  close(sock);
  //unlink("/tmp/capture_socket");
  pthread_exit(NULL);
  return NULL;
}
