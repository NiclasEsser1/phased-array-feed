#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "constants.h"
#include "log.h"

pthread_mutex_t paf_log_mutex = PTHREAD_MUTEX_INITIALIZER;

FILE *log_open(char *fname, const char *mode)
{  
  FILE *fp = fopen(fname, mode);
  if(fp == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", fname);
      return NULL;
    }
  return fp;
}

int log_add(FILE *fp, const char *type, int flush, const char *format, va_list args)
{
  struct tm *local = NULL;
  time_t rawtime;
  char buffer[MSTR_LEN] = {'\0'};

  if(fp == NULL)
    {      
      fprintf(stderr, "LOG_ERROR: log file is not open to write, ");
      fprintf(stderr, "which happens at which happens at \"%s\", line [%d], has to abort\n\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  pthread_mutex_lock(&paf_log_mutex);
  
  /* Get current time */
  time(&rawtime);
  local = localtime(&rawtime);

  /* Get real message */
  vsprintf(buffer, format, args);
  
  /* Write to log file */
  fprintf(fp, "[%s] %s\t%s", strtok(asctime(local), "\n"), type, buffer);
  
  /* Flush it if required */
  if(flush)
    fflush(fp);

  pthread_mutex_unlock(&paf_log_mutex);
  
  return EXIT_SUCCESS;
}

int log_close(FILE *fp)
{
  if(fp!=NULL)
    fclose(fp);
  
  return EXIT_SUCCESS;
}
