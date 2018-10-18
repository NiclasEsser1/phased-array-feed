#ifndef _POWER2UDP_H
#define _POWER2UDP_H

#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"

#define MSTR_LEN    512
#define DADA_HDRSZ 4096
#define NCHAN_CHK     7
#define NCHK_NIC      48
#define NBYTE_BUF     4
#define META_UP       2
#define DADA_TIMESTR  "%Y-%m-%d-%H:%M:%S"
#define FITS_TIMESTR  "%Y-%m-%dT%H:%M:%S"
#define NBYTE_UTC     28  // Bytes of UTC time stamp
#define NBYTE_BIN     4   // Bytes for other number in the output binary
#define NTOKEN_META   237 // Number of token in json metadata

typedef struct conf_t
{
  key_t key;
  char ip_udp[MSTR_LEN];
  int port_udp;
  char ip_meta[MSTR_LEN];
  int port_meta;
  int sock_meta, sock_udp;
  int nrun;
  
  char dir[MSTR_LEN];

  dada_hdu_t *hdu;
  char *hdrbuf;
  uint64_t buf_size;
  int32_t beam;
  double tsamp;
  char utc_start[MSTR_LEN];
  double picoseconds;
  float freq;
  float chan_width;
  int32_t nchan;
  int leap;
}conf_t;

int init_power2udp(conf_t *conf);
int power2udp(conf_t conf);
int destroy_power2udp(conf_t conf);
int read_header(conf_t *conf);
int init_sock(conf_t *conf);
int bat2mjd(char bat[MSTR_LEN], int leap, double *mjd);
int json2info(char *buf_meta, float *ra_f, float *dec_f, float *az_f, float *el_f, char *bat);
#endif
