#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "dada_def.h"
#include "capture.h"
#include "log.h"

int main(int argc, char **argv)
{
  /* Initial part */
  conf_t conf;

  /* Initialization */
  initialize_capture(&conf, argc, argv);
  
  /* Do the job */
  do_capture(conf);
  
  /* Destroy log interface */
  destroy_capture(conf);
  
  return EXIT_SUCCESS;
}
