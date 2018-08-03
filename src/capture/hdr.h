#ifndef _HDR_H
#define _HDR_H

#include <stdint.h>

typedef struct hdr_t
{
  int      valid;   // 0 the data frame is not valied, 1 the data frame is valied;
  uint64_t idf;     // data frame number in one period;
  uint64_t sec;     // Secs from reference epochch at start of period;
  int      epoch;   // Number of half a year from 1st of January, 2000 for the reference epochch;
  int      beam;    // The id of beam, counting from 0;
  double   freq;    // Frequency of the first chunnal in each block (integer MHz);
}hdr_t;

int hdr_keys(char *df, hdr_t *hdr);
uint64_t hdr_idf(char *df);
uint64_t hdr_sec(char *df);
double hdr_freq(char *df);
int init_hdr(hdr_t *hdr);

#endif
