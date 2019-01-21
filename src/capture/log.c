#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "log.h"

FILE *paf_log_open(char *fname, char *mode)
{  
  FILE *fp = fopen(fname, mode);
  if(fp == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", fname);
      exit(EXIT_FAILURE);
    }
  return fp;
}

int paf_log_add(FILE *fp, char *type, int flush, pthread_mutex_t mutex, const char *format, ...)
{
  struct tm *local;
  time_t rawtime;
  char buffer[MSTR_LEN];
  va_list args;
  
  /* Get real message */
  va_start(args, format);
  vsprintf(buffer, format, args);
  va_end (args);
  
  /* Write to log file */
  pthread_mutex_lock(&mutex);
  time(&rawtime);
  local = localtime(&rawtime);
  fprintf(fp, "[%s] %s\t%s\n", strtok(asctime(local), "\n"), type, buffer);
  pthread_mutex_unlock(&mutex);
  
  /* Flush it if required */
  if(flush)
    fflush(fp);
  
  return EXIT_SUCCESS;
}

int paf_log_close(FILE *fp)
{
  if(fp!=NULL)
    fclose(fp);
  
  return EXIT_SUCCESS;
}
