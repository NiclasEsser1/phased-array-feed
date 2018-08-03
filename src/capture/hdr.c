#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>
#include <byteswap.h>

#include "hdr.h"

int hdr_keys(char *df, hdr_t *hdr)
{
  uint64_t *ptr, writebuf;
  ptr = (uint64_t*)df;
    
  writebuf = bswap_64(*ptr);
  hdr->idf = writebuf & 0x00000000ffffffff;
  hdr->sec = (writebuf & 0x3fffffff00000000) >> 32;
  hdr->valid = (writebuf & 0x8000000000000000) >> 63;
    
  writebuf = bswap_64(*(ptr + 1));
  hdr->epoch = (writebuf & 0x00000000fc000000) >> 26;
    
  writebuf = bswap_64(*(ptr + 2));
  hdr->freq = (double)((writebuf & 0x00000000ffff0000) >> 16);
  hdr->beam = writebuf & 0x000000000000ffff;
    
  return EXIT_SUCCESS;
}

uint64_t hdr_idf(char *df)
{ 
  uint64_t *ptr, writebuf;
  ptr = (uint64_t*)df;
  
  writebuf = bswap_64( *ptr );
  return writebuf & 0x00000000ffffffff;
}

double hdr_freq(char *df)
{ 
  uint64_t *ptr, writebuf;
  ptr = (uint64_t*)df;
  
  writebuf = bswap_64(*(ptr + 2));
  return (double)((writebuf & 0x00000000ffff0000) >> 16);
}

uint64_t hdr_sec(char *df)
{ 
  uint64_t *ptr, writebuf;
  ptr = (uint64_t*)df;

  writebuf = bswap_64( *ptr );
  return (writebuf & 0x3fffffff00000000) >> 32;
}

int init_hdr(hdr_t *hdr)
{
  hdr->valid = 0;
  hdr->idf = 0;
  hdr->sec = 0;
  hdr->epoch = 0;
  hdr->beam = 0;
  hdr->freq = 0;
  return EXIT_SUCCESS;
}
