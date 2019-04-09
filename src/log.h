#ifdef __cplusplus
extern "C" {
#endif

#ifndef _LOG_H
#define _LOG_H
  
#include <pthread.h>
  
  FILE *log_open(char *fname, const char *mode);
  int log_add(FILE *fp, const char *type, int flush, const char *format, va_list args);
  int log_close(FILE *fp);
  
#endif
  
#ifdef __cplusplus
} 
#endif
