#ifdef __cplusplus
extern "C" {
#endif
  
#ifndef __LOG_H
#define __LOG_H

#include <pthread.h>
  
#define MSTR_LEN 1024

  FILE *log_open(char *fname, const char *mode);
  int log_add(FILE *fp, const char *type, int flush, pthread_mutex_t mutex, const char *format, ...);
  int log_close(FILE *fp);
  
#endif
  
#ifdef __cplusplus
} 
#endif
