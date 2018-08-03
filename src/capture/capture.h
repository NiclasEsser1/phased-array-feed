#ifndef CAPTURE_H
#define CAPTURE_H

#include <netinet/in.h>

#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"
#include "hdr.h"

#define MSTR_LEN      1024
#define MPORT_CAPTURE 16
#define NCHK_PORT     16
#define DADA_HDR_SIZE 4096
#define MCHAN_CAPTURE 336

typedef struct conf_t
{ 
  key_t key;
  int hdr;
  dada_hdu_t *hdu;
  
  int sock[MPORT_CAPTURE];
  uint64_t ndf_port[MPORT_CAPTURE];
  uint64_t ndf_chan[MCHAN_CAPTURE];
  uint64_t rbuf_ndf;
  uint64_t tbuf_ndf;
  
  char ip_active[MPORT_CAPTURE][MSTR_LEN];
  int port_active[MPORT_CAPTURE];
  int nport_active;
  int nchunk_active_expect[MPORT_CAPTURE];  
  int nchunk_active_actual[MPORT_CAPTURE];  
  int port_cpu[MPORT_CAPTURE];
  int sync_cpu, monitor_cpu;
  int thread_bind;

  int pktsz, pktoff, required_pktsz;
  char ip_dead[MPORT_CAPTURE][MSTR_LEN];
  int port_dead[MPORT_CAPTURE];
  int nport_dead;
  int nchunk_dead[MPORT_CAPTURE];

  char hdr_fname[MSTR_LEN];
  double center_freq;
  int nchan;
  
  char utc_start[MSTR_LEN];
  double mjd_start;
  uint64_t picoseconds;
  uint64_t df_sec;
  uint64_t df_idf;

  int nchunk;
  double bw, resolution;
  char instrument[MSTR_LEN];
  int df_prd;
  char dir[MSTR_LEN];

  uint64_t rbufsz;
  uint64_t ndf_prd;
}conf_t;

int init_capture(conf_t *conf);
void *capture_thread(void *conf);
int acquire_idf(hdr_t hdr, hdr_t hdr_ref, conf_t conf, int64_t *idf);
int acquire_ifreq(hdr_t hdr, conf_t conf, int *ifreq);
int init_buf(conf_t *conf);

#endif
