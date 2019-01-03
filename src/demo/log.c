#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include "multilog.h"

#define MSTR_LEN 1024

multilog_t *runtime_log;

// gcc -o log log.c -I/usr/local/include -lpsrdada
// gcc -o log log.c -I/home/pulsar/.local/include -L/home/pulsar/.local/lib -lpsrdada 

int main(int argc, char **argv)
{
  /* Setup log interface */
  char fname_log[MSTR_LEN];
  FILE *fp_log = NULL;
  double elapsed_time;
  struct timespec start, now;
  
  clock_gettime(CLOCK_REALTIME, &start); 
  sprintf(fname_log, "log.log");
  fp_log = fopen(fname_log, "ab+"); 
  if(fp_log == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", fname_log);
      return EXIT_FAILURE;
    }
  runtime_log = multilog_open("capture", 1);
  multilog_add(runtime_log, fp_log);
  multilog(runtime_log, LOG_INFO, "CAPTURE START\n");
  while (1)
    {
      clock_gettime(CLOCK_REALTIME, &now); 
      multilog(runtime_log, LOG_INFO, "Try to crash a server, now it is %f seconds ...\n", (double)(now.tv_sec - start.tv_sec) + (double)(now.tv_nsec - start.tv_nsec)/1.0E9);
    }
  /* Destory log interface */
  multilog(runtime_log, LOG_INFO, "CAPTURE END\n\n");
  multilog_close(runtime_log);
  fclose(fp_log);

  return EXIT_SUCCESS;
}
