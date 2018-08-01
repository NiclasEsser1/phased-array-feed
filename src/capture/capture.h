#ifndef CAPTURE_H
#define CAPTURE_H

#include <netinet/in.h>

#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"

#define MSTR_LEN      1024
#define MPORT_CAPTURE 16
#define NCHK_PORT     16

typedef struct conf_t
{
  key_t key;
  int hdr;
  
  char ip_active[MPORT_CAPTURE][MSTR_LEN];
  int port_active[MPORT_CAPTURE];
  int nport_active;
  int nchunk_active_expect[MPORT_CAPTURE];  
  int nchunk_active_actual[MPORT_CAPTURE];  
  
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

  char dir[MSTR_LEN];
}conf_t;

#endif
