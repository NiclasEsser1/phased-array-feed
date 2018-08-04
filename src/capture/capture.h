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
#define DADA_HDR_SIZE 4096
#define MCHK_CAPTURE  48

typedef struct conf_t
{ 
  key_t key;
  int hdr;
  dada_hdu_t *hdu;
  
  uint64_t rbuf_ndf_chk, tbuf_ndf_chk;

  int pktsz, pktoff, required_pktsz;
  int port_cpu[MPORT_CAPTURE];
  int sync_cpu, monitor_cpu;
  int thread_bind;
  
  char ip_active[MPORT_CAPTURE][MSTR_LEN];
  int port_active[MPORT_CAPTURE];
  int nport_active;
  int nchunk_active_expect[MPORT_CAPTURE];  
  int nchunk_active_actual[MPORT_CAPTURE];  

  char ip_dead[MPORT_CAPTURE][MSTR_LEN];
  int port_dead[MPORT_CAPTURE];
  int nport_dead;
  int nchunk_dead[MPORT_CAPTURE];

  double center_freq;
  int nchan;
  
  uint64_t sec_start, idf_start;

  int nchunk;
  int sec_prd;
  int monitor_sec;
  char dir[MSTR_LEN];

  uint64_t rbufsz, tbufsz;
  
  uint64_t ndf_chk_prd;
}conf_t;

int init_capture(conf_t *conf);
void *capture_thread(void *conf);
int acquire_idf(hdr_t hdr, hdr_t hdr_ref, conf_t conf, int64_t *idf);
int acquire_ichk(hdr_t hdr, conf_t conf, int *ifreq);
int init_buf(conf_t *conf);

#endif
