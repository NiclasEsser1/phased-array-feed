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

#define MSTR_LEN      1024
#define MPORT_CAPTURE 16
#define DADA_HDRSZ    4096
#define MCHK_CAPTURE  48
#define SECDAY        86400.0
#define MJD1970       40587.0

/** Structure that is used to store the entire configuration for capturing **/
typedef struct conf_t
{
  /** Key for referencing the shared memory ringbuffer **/
  key_t key;

  /** Header and data unit (HDU) provided by psrdada
      contains following attribtues
       - log (multilog_t*),
       - data_block (ipcio_t*),
       - header_block (ipcbuf_t*),
       - header (char*),
       - header_size (size_t),
       - data_block_key in hex (key_t),
       - header_block_key in hex (key_t)
  **/
  dada_hdu_t *hdu;

  /** Number of data frame in each buffer block of each chunk **/
  uint64_t rbuf_ndf_chk;

  /** The number of data frames in each temp buffer of each chunk **/
  uint64_t tbuf_ndf_chk;

  /** Size of one UDP packet, usually should be 7168 (raw) + 64 (header)**/
  int pktsz;

  /** Start point or offset of the packet, seems to allow skipping unnecessaey header data **/
  int pktoff:

  /** Actual size that is required for one UDP packet (pktsz - pktoff) **/
  int required_pktsz;

  /** Array holding CPU ID's used to run several threads concurrently**/
  int port_cpu[MPORT_CAPTURE];

  /** ID of CPU controlling the psrdada ringbuffer **/
  int buf_ctrl_cpu;

  /** ID of CPU controlling the capturing process **/
  int capture_ctrl_cpu;

  /** 0 = threads are not binded; 1 = each thread is assign to a dedicated CPU **/
  int thread_bind;

  /** 2D Array holding the ip address of acrive ports **/
  char ip_active[MPORT_CAPTURE][MSTR_LEN];

  /** Array holding the port number (e.g. 17100)**/
  int port_active[MPORT_CAPTURE];

  /** Number of active ports **/
  int nport_active;

  /** Number actually active channel chunks per port **/
  int nchk_active_actual[MPORT_CAPTURE];

  /** Name of the used instrument **/
  char instrument[MSTR_LEN];

  /** Overall center frequency of received channel chunks **/
  double center_freq;

  /** Number of channels received by the capture program (multiple of 7?)**/
  int nchan;

  /** Assumed: Channels per chunk. If so, has to be 7 in each case **/
  int nchan_chk;

  /** String containing the name of header template/file for PSRDADA - required (What is it)**/
  char hfname[MSTR_LEN];

  /** Reference in seconds from BMF on start of capturing **/
  uint64_t sec_ref;

  /** initial or index data frame (IDF) reference from BMF on start of capturing **/
  uint64_t idf_ref;

  /** Epoch reference, contained in each packet header **/
  double epoch_ref;

  /** Frequency chunks of current capture, including all alive chunks and dead chunks **/
  int nchk;

  /** Assumed to be the period between 2 packets in seconds (?) **/
  int sec_prd;

  char ctrl_addr[MSTR_LEN];
  char dir[MSTR_LEN];

  double df_res;  // time resolution of each data frame, for start time determination;
  double blk_res; // time resolution of each buffer block, for start time determination;
  //uint64_t buf_dfsz; // data fram size in buf, TFTFP order, and here is the size of each T, which is the size of each FTP. It is for the time determination with start_byte, it should be multiple of buf_size;

  uint64_t rbufsz, tbufsz;

  /** Number of dataframe in each period or each chunk (???)**/
  uint64_t ndf_chk_prd;

  /************************************/
  /** Are these attributes necessary? These are just assigned and never used in the whole program **/
  /** Is it necessary? Assigned by command line, but never used... **/
  int nchk_active_expect[MPORT_CAPTURE];
  /** Is it necessary? Assigned by command line, but never used... **/
  char ip_dead[MPORT_CAPTURE][MSTR_LEN];
  /** Is it necessary? Assigned by command line, but never used... **/
  int port_dead[MPORT_CAPTURE];
  /** Is it necessary? Assigned by command line, but never used... **/
  int nport_dead;
  /** Is it necessary? Assigned by command line, but never used... **/
  int nchk_dead[MPORT_CAPTURE];
  /************************************/

}conf_t;

/** Structure used for parsing each UDP packet header **/
typedef struct hdr_t
{
  /** 0 the data frame is not valied, 1 the data frame is valied; **/
  int valid;

  /** data frame number in one period; **/
  uint64_t idf;

  /** Secs from reference epoch at start of period; **/
  uint64_t sec;

  /** Number of half a year from 1st of January, 2000 for the reference epochch; **/
  int epoch;

  /** The id of beam, counting from 0; **/
  int beam;

  /** Frequency of the first chunnal in each block (integer MHz); **/
  double freq;
}hdr_t;


/**
* @brief    Initilizes the capturing enviroment
*
* @param    conf  A pointer to the actual used configuration struct
*
* @detail   Creates a ringbuffer based on psrdada library.
*/
int init_capture(conf_t *conf);


/**
* @brief    Captures received UDP packets in psrdada ringbuffer
*
* @param    conf  A pointer to the actual used configuration struct
*
*
* @return   Always returns NULL (??)
*
* @ detail  This function parses received data frames. Each active threads has to call this function concurrently.
**/
void *capture(void *conf);


/**
* @brief    Calculates the index value of a dataframe and thus the position of it
*
* @param    idf           Index of received packet, contained in packet header
* @param    sec           Seconds from reference epoch at start period, contained in packet header
* @param    idf_ref       Reference index from initialization process
* @param    sec_ref       Reference seconds from initialization process
* @param    sec_prd       Assumed to be the period between 2 packets in seconds
* @param    ndf_chk_prd   Number of dataframe in each period or each frequency chunk
* @param    idf_buf       Pointer to the dataframe index
*
* @return   Always returns EXIT_SUCCESS (??)
**/
int acquire_idf(uint64_t idf, uint64_t sec, uint64_t idf_ref, uint64_t sec_ref, int sec_prd, uint64_t ndf_chk_prd, int64_t *idf_buf);


/**
* @brief    Calculates the index value of a dataframe and thus the position of it
*
* @param    freq        Frequency of the first channel within a chunk. Contained in packet header
* @param    center_freq Center frequency of the current capture. Contained in configuration struct
* @param    nchan_chk   Number of channels per Packet, should be 7 in each case. Contained in configuration struct.
* @param    nchk       Number of chunks
* @param    ichk       Pointer to the chunk index
*
* @return   Returns EXIT_SUCCESS on success and EXIT_FAILURE onf failure
**/
int acquire_ichk(double freq, double center_freq, int nchan_chk, int nchk, int *ichk);



int init_buf(conf_t *conf);



int destroy_capture(conf_t conf);



void *capture_control(void *conf);



int threads(conf_t *conf);



void *buf_control(void *conf);

int hdr_keys(char *df, hdr_t *hdr);
uint64_t hdr_idf(char *df);
uint64_t hdr_sec(char *df);
double hdr_freq(char *df);
int init_hdr(hdr_t *hdr);

#endif
