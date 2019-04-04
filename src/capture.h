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
    FILE *log_file;  
    key_t key;
    
    uint64_t ndf_per_chunk_rbuf, ndf_per_chunk_tbuf;
    
    int dfsz_seek;
    int nchunk_expect, nchan_expect, nchunk_actual;  
    
    char ip[MSTR_LEN];
    int port;
    char *tbuf;
    uint64_t tbufsz, blksz_rbuf;
    
    char dir[MSTR_LEN], dada_header_template[MSTR_LEN];
    
    int days_from_1970;   // Number days of epoch from 1970 
    uint64_t seconds_from_epoch, df_in_period; // Seconds from epoch time of BMF and the index of data frame in BMF stream period
    
    dada_header_t dada_header;
    dada_hdu_t *hdu;
    ipcbuf_t *data_block, *header_block;
    
    struct timeval tout;
  }conf_t;
  
  int default_arguments(conf_t *conf);
  int examine_record_arguments(conf_t conf, char **argv, int argc);
  int create_buffer(conf_t *conf);
  int destory_buffer(conf_t conf);
  
#endif

#ifdef __cplusplus
} 
#endif
