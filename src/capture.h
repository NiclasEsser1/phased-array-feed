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

#define MSTR_LEN       1024
#define MPORT_CAPTURE  10
#define DADA_HDRSZ     4096
#define NCHK_FULL_BAND 48
#define SECDAY         86400.0
#define MJD1970        40587.0
#define DFSZ           7232
#define NCHAN_CHK      7
#define PRD            27
#define NDF_CHK_PRD    250000

typedef struct conf_t
{
  int ithread;
  FILE *logfile;
  
  key_t key;
  dada_hdu_t *hdu;
  ipcbuf_t *db_data, *db_hdr;
  
  uint64_t rbuf_ndf_chk, tbuf_ndf_chk, rbufsz, tbufsz;
  
  int pad;
  int dfoff, required_dfsz;
  int cpt_cpu[MPORT_CAPTURE], rbuf_ctrl_cpu, cpt_ctrl_cpu, cpt_ctrl, cpu_bind;
  char cpt_ctrl_addr[MSTR_LEN];
  
  char ip_alive[MPORT_CAPTURE][MSTR_LEN], ip_dead[MPORT_CAPTURE][MSTR_LEN];;
  int port_alive[MPORT_CAPTURE], port_dead[MPORT_CAPTURE];
  int nport_alive, nport_dead;
  int nchk_alive_expect[MPORT_CAPTURE], nchk_alive_actual[MPORT_CAPTURE], nchk_dead[MPORT_CAPTURE];;  // For each port;
  int nchan, nchk, nchk_alive;
  
  char dir[MSTR_LEN], hfname[MSTR_LEN], instrument[MSTR_LEN];
  char source[MSTR_LEN], ra[MSTR_LEN], dec[MSTR_LEN];
  
  double cfreq, chan_res, bw;
  double df_res, blk_res;  // time resolution of each data frame and ring buffer block, for start time determination;
  double ichk0;

  int epoch0;                       // Number of days from 1970
  time_t sec_int, sec_int_ref;      // int seconds from 1970
  uint64_t df_sec0, idf_prd0; // Seconds from epoch time of BMF and the index of data frame in BMF stream period
  uint64_t picoseconds, picoseconds_ref;
  
  char utc_start[MSTR_LEN];
  double mjd_start;

  struct timeval tout;
}conf_t;

int init_capture(conf_t *conf);
void *capture(void *conf);
int init_buf(conf_t *conf);
int destroy_capture(conf_t conf);

void *capture_control(void *conf);

int threads(conf_t *conf);
void *buf_control(void *conf);

int dada_header(conf_t conf);

#endif
