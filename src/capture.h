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
#include "psrdada.h"

#define acquire_chunk_index(freq, cfreq, nchunk) (int((freq - cfreq + 0.5)/NCHAN_PER_CHUNK + 0.5 * nchunk))

  typedef struct conf_t
  {
    int debug; // If debug, disable_sod
    FILE *log_file;  
    key_t key;
    
    uint64_t ndf_per_chunk_rbuf, ndf_per_chunk_tbuf;

    double time_res_blk;
    int dfsz_seek, dfsz_keep;
    int nchunk_expect, nchunk_actual;  
    
    char ip[MSTR_LEN];
    int port;
    char *tbuf, *dbuf;
    uint64_t tbufsz, blksz_rbuf;
        
    char dir[MSTR_LEN], dada_header_template[MSTR_LEN];
    
    int days_from_1970;   // Number days of epoch from 1970 
    uint64_t seconds_from_epoch, df_in_period; // Seconds from epoch time of BMF and the index of data frame in BMF stream period
    
    dada_header_t *dada_header;
    dada_hdu_t *hdu;
    ipcbuf_t *data_block, *header_block;

    char receiver[MSTR_LEN];
    double freq;
    
    struct timeval tout;
  }conf_t;

  void usage();
  int default_arguments(conf_t *conf);
  int parse_arguments(conf_t *conf, int argc, char **argv);
  int verify_arguments(conf_t conf, int argc, char **argv);
  int create_buffer(conf_t *conf);
  int decode_df_header(char *dbuf, uint64_t *df_in_period, uint64_t *seconds_from_epoch, double *freq);
  int update_dada_header(conf_t *conf);
  int log_add_wrap(conf_t conf, const char *type, int flush, const char *format, ...);
  int initialize_capture(conf_t *conf, int argc, char **argv);

  int destroy_buffer(conf_t conf);
  int destroy_capture(conf_t conf);
    
#endif

#ifdef __cplusplus
} 
#endif
