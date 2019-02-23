#ifdef __cplusplus
extern "C" {
#endif
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
#include "constants.h"

typedef struct conf_t
{
  FILE *log_file;
  int process_index, thread_index, beam_index;
  
  key_t key;
  dada_hdu_t *hdu;
  ipcbuf_t *data_block, *header_block;
  
  uint64_t ndf_per_chunk_rbuf, ndf_per_chunk_tbuf, blksz_rbuf, tbufsz;
  
  int pad;
  int dfsz_seek, dfsz_keep;
  int capture_cpu[NPORT_MAX], rbuf_ctrl_cpu, capture_ctrl_cpu, capture_ctrl, cpu_bind;
  char capture_ctrl_addr[MSTR_LEN];
  
  char ip_alive[NPORT_MAX][MSTR_LEN], ip_dead[NPORT_MAX][MSTR_LEN];;
  int port_alive[NPORT_MAX], port_dead[NPORT_MAX];
  int nport_alive, nport_dead;
  int nchunk_alive_expect_on_port[NPORT_MAX], nchunk_alive_actual_on_port[NPORT_MAX], nchunk_dead_on_port[NPORT_MAX];;  // For each port;
  int nchan, nchunk, nchunk_alive;
  
  char dir[MSTR_LEN], dada_header_template[MSTR_LEN];
  char source[MSTR_LEN], ra[MSTR_LEN], dec[MSTR_LEN];

  double center_freq, freq_res, bandwidth;
  double time_res_df, time_res_blk;  // time resolution of each data frame and ring buffer block, for start time determination;
  double chunk_index0;

  int days_from_1970;   // Number of days from 1970
  time_t seconds_from_1970;
  uint64_t seconds_from_epoch, df_in_period; // Seconds from epoch time of BMF and the index of data frame in BMF stream period
  uint64_t picoseconds, picoseconds_ref;
  char utc_start[MSTR_LEN];
  double mjd_start;

  struct timeval tout;
}conf_t;

int initialize_capture(conf_t *conf);
void *do_capture(void *conf);
int destroy_capture(conf_t conf);

void *capture_control(void *conf);
void *buf_control(void *conf);

int default_arguments(conf_t *conf);
int threads(conf_t *conf);
int register_dada_header(conf_t conf);
int examine_record_arguments(conf_t conf, char **argv, int argc);
#endif

#ifdef __cplusplus
} 
#endif
