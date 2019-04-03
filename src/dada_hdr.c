#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include "ipcbuf.h"

int read_dada_header(ipcbuf_t *hdr)
{
  char *hdrbuf = NULL;
  uint64_t bufsz;

  hdrbuf = ipcbuf_get_next_read(hdr, &bufsz);

  
  
  ipcbuf_mark_cleared(hdr);
  
  return EXIT_SUCCESS;
}
