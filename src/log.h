#ifdef __cplusplus
extern "C" {
#endif
  
#ifndef __LOG_H
#define __LOG_H

#include <pthread.h>
  
#define MSTR_LEN 1024

  FILE *paf_log_open(char *fname, const char *mode);
  int paf_log_add(FILE *fp, const char *type, int flush, pthread_mutex_t mutex, const char *format, ...);
  int paf_log_close(FILE *fp);
  
#endif
  
#ifdef __cplusplus
} 
#endif
