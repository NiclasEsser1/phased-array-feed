#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

#include "constants.h"

int main(int argc, char **argv)
{
  
  struct tm *local = NULL;
  time_t rawtime;
  char buffer[MSTR_LEN] = {'\0'};
  va_list args;

  /* Get current time */
  time(&rawtime);
  local = localtime(&rawtime);
  fprintf(stdout, "%s\n", strtok(asctime(local), "\n"));
  fflush(stdout);

  while(strtok(asctime(local), "\n"))
    {
      time(&rawtime);
      local = localtime(&rawtime);
      fprintf(stdout, "%s\n", strtok(asctime(local), "\n"));
      fflush(stdout);
      sleep(1);
    }	
  
  return EXIT_SUCCESS;
}
