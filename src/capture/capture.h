#ifndef __CAPTURE_H
#define __CAPTURE_H

#include <netinet/in.h>
#include <stdint.h>
#include <inttypes.h>
#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"

#define MAX_STRLEN 1024
#define SECDAY     86400.0

void usage();
  
typedef struct configuration_t
{
  key_t rbuf_key;
  double freq;

  uint64_t npkt_per_chunk_period;
  uint64_t npkt_per_chunk_rbuf, npkt_per_chunk_tbuf;

  int port;
  int pkt_period_secs;
  int nchan_per_chunk;
  int pktsize_bytes, offset_pktsize_bytes, remind_pktsize_bytes;
  int nchunk_actual, nchunk_expect, nchan_actual, nchan_expect;

  char ip[MAX_STRLEN];
  char instrument_name[MAX_STRLEN];
  
  char dada_hdr_fname[MAX_STRLEN];
  char runtime_dir[MAX_STRLEN];

  uint64_t refpkt_secs, refpkt_idx_period; 
  int refpkt_epoch;
  
  time_t int_reftime_secs;
  uint64_t frac_reftime_psecs;

  multilog_t *runtime_log;
  FILE *fp_log;

  double pkt_tres_secs, rbuf_tres_secs;
  double refchunk_idx;

  uint64_t rbufsize_bytes, tbufsize_bytes;
  uint64_t tbuf_thred_pkts;
  dada_hdu_t *hdu;
  char *tbuf, *rbuf;
  ipcbuf_t *db_data, *db_hdr;
  
  char *pkt;
  int sock;
  
  struct timeval timeout;
  
}configuration_t;

int parse_arguments(int argc, char **argv, configuration_t *configuration);
int initialize_socket(configuration_t *configuration);
int initialize_buffer(configuration_t *configuration);
int initialize_capture(int argc, char **argv, configuration_t *configuration);
int do_capture(configuration_t configuration);
int destroy_capture(configuration_t configuration);

#endif
