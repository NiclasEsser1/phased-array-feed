#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include "multilog.h"
#include "dada_def.h"
#include "capture.h"

int main(int argc, char **argv)
{
  configuration_t configuration;
  
  /* Initialize the capture */
  initialize_capture(argc, argv, &configuration);

  /* Do the capture */
  do_capture(configuration);
    
  /* close the capture */
  destroy_capture(configuration);    
  
  return EXIT_SUCCESS;
}
