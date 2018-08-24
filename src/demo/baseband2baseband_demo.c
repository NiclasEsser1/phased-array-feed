#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <sys/socket.h>
#include <linux/un.h>
#include <pthread.h>

#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"

#define MSTR_LEN     1024
#define DADA_HDRSZ   4096
#define SECDAY       86400.0
#define MJD1970      40587.0

// It is a demo to check the sod, eod and start time of baseband2baseband
// It also helps to understand the control of baseband2baseband
// gcc -o baseband2baseband_demo baseband2baseband_demo.c -L/usr/local/cuda/lib64 -I/usr/local/include -lpsrdada -lcudart -lm -pthread

multilog_t *runtime_log;
int quit;
pthread_mutex_t quit_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct conf_t
{
  dada_hdu_t *hdu_in, *hdu_out;
  key_t key_in, key_out;
  char *curbuf_in, *curbuf_out;
  int pktsz;
  char ctrl_addr[MSTR_LEN];
  uint64_t curbufsz;
  double blk_res;
  int thread_bind;
  time_t sec_ref;
  uint64_t picoseconds_ref;
  int cpus[2];
  char hdr[DADA_HDRSZ];
  
}conf_t;

int threads(conf_t *conf);
void *baseband2baseband(void *conf);
void *control(void *conf);

void usage ()
{
  fprintf (stdout,
	   "baseband2baseband_demo - A demo to pass baseband data from a ring buffer to another ring buffer \n"
	   "\n"
	   "Usage: baseband2baseband_main [options]\n"
	   " -a  Hexacdecimal shared memory key for incoming ring buffer\n"
	   " -b  Hexacdecimal shared memory key for outcoming ring buffer\n"
	   " -c  The packet size\n"
	   " -d  The address for control socket\n"
	   " -e  The resolution of output ring buffer block\n"
	   " -f  Bind thread or not, 1:cpu:cpu, or 0:0:0\n"
	   " -h  show help\n");
}

int main(int argc, char **argv)
{
  conf_t conf;
  int i, arg, pktsz;
  uint64_t curbufsz;
  
  /* Init */
  while((arg=getopt(argc,argv,"a:b:hc:d:e:f:")) != -1)
    {
      
      switch(arg)
	{
	case 'h':
	  usage();
	  return EXIT_FAILURE;
	  
	case 'a':	  	  
	  if(sscanf(optarg, "%x", &conf.key_in) != 1)
	    {
	      fprintf(stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;

	case 'b':
	  if (sscanf (optarg, "%x", &conf.key_out) != 1)
	    {
	      fprintf (stderr, "Could not parse key from %s, which happens at \"%s\", line [%d].\n", optarg, __FILE__, __LINE__);
	      return EXIT_FAILURE;
	    }
	  break;

	case 'c':
	  sscanf(optarg, "%d", &conf.pktsz);
	  break;
	  
	case 'd':
	  sscanf(optarg, "%s", conf.ctrl_addr);
	  break;

	case 'e':
	  sscanf(optarg, "%lf", &conf.blk_res);
	  break;
	  
	case 'f':
	  sscanf(optarg, "%d:%d:%d", &conf.thread_bind, &conf.cpus[0], &conf.cpus[1]);
	  break;
	}
    }
  char fname_log[MSTR_LEN];
  FILE *fp_log = NULL;
  sprintf(fname_log, "/beegfs/DENG/docker/baseband2baseband_demo.log");
  fp_log = fopen(fname_log, "ab+"); 
  if(fp_log == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", fname_log);
      return EXIT_FAILURE;
    }
  runtime_log = multilog_open("baseband2baseband_demo", 1);
  multilog_add(runtime_log, fp_log);
  multilog(runtime_log, LOG_INFO, "BASEBAND2BASEBAND_DEMO START\n");
  
  /* attach to input ring buffer */
  conf.hdu_in = dada_hdu_create(runtime_log);
  if(conf.hdu_in == NULL)
    {
      fprintf(stdout, "HERE DADA_HDU_CREATE\n");
      exit(1);
    }
  dada_hdu_set_key(conf.hdu_in, conf.key_in);
  if(dada_hdu_connect(conf.hdu_in))    
    {
      fprintf(stdout, "HERE DADA_HDU_CONNECT\n");
      exit(1);
    }
  if(dada_hdu_lock_read(conf.hdu_in))
    {      
      fprintf(stdout, "HERE DADA_HDU_LOCK_READ\n");
      exit(1);
    }
  
  /* Prepare output ring buffer */
  conf.hdu_out = dada_hdu_create(runtime_log);
  if(conf.hdu_out == NULL)    
    {
      fprintf(stdout, "HERE DADA_HDU_CREATE\n");
      exit(1);
    }
  dada_hdu_set_key(conf.hdu_out, conf.key_out);
  if(dada_hdu_connect(conf.hdu_out))      
    {
      fprintf(stdout, "HERE DADA_HDU_CONNECT\n");
      exit(1);
    }
  if(dada_hdu_lock_write(conf.hdu_out))
    {      
      fprintf(stdout, "HERE DADA_HDU_LOCK_READ\n");
      exit(1);
    }
  
  /* Do the real job */
  threads(&conf);
  
  /* Destroy */
  dada_hdu_unlock_write(conf.hdu_out);    
  dada_hdu_disconnect(conf.hdu_out);
  dada_hdu_destroy(conf.hdu_out);
  
  dada_hdu_unlock_read(conf.hdu_in);
  dada_hdu_disconnect(conf.hdu_in);
  dada_hdu_destroy(conf.hdu_in);

  multilog(runtime_log, LOG_INFO, "BASEBAND2BASEBAND_DEMO END\n");
  fclose(fp_log);
  
  return EXIT_SUCCESS;
}

void *baseband2baseband(void *conf)
{
  conf_t *b2bconf = (conf_t *)conf;
  ipcbuf_t *db_in = NULL, *db_out = NULL;
  ipcbuf_t *hdr_in = NULL;
  int quit_status;
  uint64_t hdrsz;
  double mjdstart_ref;
  char *hdrbuf_in = NULL;
  
  /* To see if these two buffers are the same size */
  db_in  = (ipcbuf_t *)b2bconf->hdu_in->data_block;
  db_out = (ipcbuf_t *)b2bconf->hdu_out->data_block;
  hdr_in  = (ipcbuf_t *)b2bconf->hdu_in->header_block;

  ipcbuf_disable_sod(db_out);
  
  quit = 0;
  pthread_mutex_lock(&quit_mutex);
  quit_status = quit;
  pthread_mutex_unlock(&quit_mutex);
  
  hdrbuf_in = ipcbuf_get_next_read(hdr_in, &hdrsz);
  if(!hdrbuf_in)
    {
      multilog(runtime_log, LOG_ERR, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      pthread_mutex_lock(&quit_mutex);
      quit = 1;
      pthread_mutex_unlock(&quit_mutex);
      
      pthread_exit(NULL);
      return NULL;
    }
  memcpy(b2bconf->hdr, hdrbuf_in, DADA_HDRSZ);  // Get a copy of the header
  
  if(ascii_header_get(hdrbuf_in, "MJD_START", "%lf", &mjdstart_ref) < 0)
    {
      multilog(runtime_log, LOG_ERR, "Error getting MJD_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error setting MJD_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      pthread_mutex_lock(&quit_mutex);
      quit = 1;
      pthread_mutex_unlock(&quit_mutex);
      
      pthread_exit(NULL);
      return NULL;
    }
  b2bconf->sec_ref = (time_t)round(SECDAY * (mjdstart_ref - MJD1970));
  if(ascii_header_get(hdrbuf_in, "PICOSECONDS", "%"PRIu64, &(b2bconf->picoseconds_ref)) < 0)  
    {
      multilog(runtime_log, LOG_ERR, "Error getting PICOSECONDS, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      fprintf(stderr, "Error getting PICOSECONDS, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
      
      pthread_mutex_lock(&quit_mutex);
      quit = 1;
      pthread_mutex_unlock(&quit_mutex);
      
      pthread_exit(NULL);
      return NULL;
    }    
  ipcbuf_mark_cleared(hdr_in); // Clear the header block for later use

  b2bconf->curbuf_in  = ipcbuf_get_next_read(db_in, &b2bconf->curbufsz);
  b2bconf->curbuf_out = ipcbuf_get_next_write(db_out);

  while(!ipcbuf_eod(db_in) && (quit_status == 0))
    {
      memcpy(b2bconf->curbuf_out, b2bconf->curbuf_in, b2bconf->pktsz);
      
      ipcbuf_mark_filled(db_out, b2bconf->pktsz);
      ipcbuf_mark_cleared(db_in);
      b2bconf->curbuf_in  = ipcbuf_get_next_read(db_in, &b2bconf->curbufsz);
      b2bconf->curbuf_out = ipcbuf_get_next_write(db_out);
      fprintf(stdout, "HERE EOD\t%d\t", ipcbuf_eod(db_in));
      fprintf(stdout, "HERE SOD\t%d\n", ipcbuf_sod(db_in));
      fprintf(stdout, "HERE EOD\t%d\t", ipcbuf_eod(db_in));
      fprintf(stdout, "HERE SOD\t%d\n\n", ipcbuf_sod(db_in));

      pthread_mutex_lock(&quit_mutex);
      quit_status = quit;
      pthread_mutex_unlock(&quit_mutex);
    }
  
  pthread_mutex_lock(&quit_mutex);
  quit = 1;
  pthread_mutex_unlock(&quit_mutex);
  
  pthread_exit(NULL);
  return NULL;
}

void *control(void *conf)
{
  struct timeval tout={1, 0};  // Force to timeout if we could not receive data frames in 1 second;
  conf_t *b2bconf = (conf_t *)conf;
  int quit_status;
  int sock, i;
  struct sockaddr_un sa, fromsa;
  uint64_t start_buf;
  ipcbuf_t *db_out = NULL, *hdr_out = NULL;
  char command_line[MSTR_LEN], command[MSTR_LEN];
  socklen_t fromlen;
  char source[MSTR_LEN], ra[MSTR_LEN], dec[MSTR_LEN];
  time_t sec;
  double sec_offset;
  uint64_t picoseconds, picoseconds_offset; 
  char utc_start[MSTR_LEN];
  char *hdrbuf_out = NULL;
  uint64_t hdrsz;
  char hdr[DADA_HDRSZ];
  double mjd_start;
  
  db_out = (ipcbuf_t *)b2bconf->hdu_out->data_block;
  hdr_out = (ipcbuf_t *)b2bconf->hdu_out->header_block;
  
  /* Do the real job */
  pthread_mutex_lock(&quit_mutex);
  quit_status = quit;
  pthread_mutex_unlock(&quit_mutex);

  /* Create an unix socket for control */
  if((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) == -1)
    {
      multilog(runtime_log, LOG_ERR, "Can not create file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Can not create file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      pthread_mutex_lock(&quit_mutex);
      quit = 1;
      pthread_mutex_unlock(&quit_mutex);

      pthread_exit(NULL);
      return NULL;
    }  
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tout, sizeof(tout));
  memset(&sa, 0, sizeof(struct sockaddr_un));
  sa.sun_family = AF_UNIX;
  snprintf(sa.sun_path, UNIX_PATH_MAX, "%s", b2bconf->ctrl_addr);
  unlink(b2bconf->ctrl_addr);
  
  if(bind(sock, (struct sockaddr*)&sa, sizeof(sa)) == -1)
    {
      multilog(runtime_log, LOG_ERR, "Can not bind to file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      fprintf (stderr, "Can not bind to file socket, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      pthread_mutex_lock(&quit_mutex);
      quit = 1;
      pthread_mutex_unlock(&quit_mutex);

      close(sock);
      pthread_exit(NULL);
      return NULL;
    }
  
  while(quit_status == 0)
    {     
      if(recvfrom(sock, (void *)command_line, MSTR_LEN, 0, (struct sockaddr*)&fromsa, &fromlen) > 0)
	{
	  if(strstr(command_line, "END-OF-DATA") != NULL)
	    {
	      multilog(runtime_log, LOG_INFO, "Got END-OF-DATA signal, has to enable eod.\n");
	      fprintf(stdout, "Got END-OF-DATA signal, which happens at \"%s\", line [%d], has to enable eod.\n", __FILE__, __LINE__);

	      ipcbuf_enable_eod(db_out);
	      ipcbuf_disable_sod(db_out);

	      ipcbuf_enable_eod(hdr_out);
	      ipcbuf_disable_sod(hdr_out);

	      //ipcbuf_hard_reset(db_out);
	      //ipcbuf_hard_reset(hdr_out);
	      
	      hdrbuf_out = ipcbuf_get_next_write(hdr_out);
	      ipcbuf_mark_filled(hdr_out, 0);
	    }
	  
	  if(strstr(command_line, "START-OF-DATA") != NULL)
	    {
	      multilog(runtime_log, LOG_INFO, "Got START-OF-DATA signal, has to enable sod.\n");
	      fprintf(stdout, "Got START-OF-DATA signal, which happens at \"%s\", line [%d], has to enable sod.\n", __FILE__, __LINE__);

	      sscanf(command_line, "%[^:]:%[^:]:%[^:]:%[^:]:%"SCNu64"", command, source, ra, dec, &start_buf); 
	      start_buf = (start_buf > ipcbuf_get_write_count(db_out)) ? start_buf : ipcbuf_get_write_count(db_out);
	      fprintf(stdout, "NUMBER OF BUF\t%"PRIu64"\n", ipcbuf_get_write_count(db_out));
	      fprintf(stdout, "%"PRIu64"\n", start_buf);

	      ipcbuf_enable_sod(db_out, start_buf, 0);
	      ipcbuf_enable_sod(hdr_out, ipcbuf_get_write_count(hdr_out), 0);
	      
	      /* To get time stamp for current header */
	      sec_offset = start_buf * b2bconf->blk_res;
	      picoseconds_offset = 1E6 * (round(1.0E6 * (sec_offset - floor(sec_offset))));
	      picoseconds = picoseconds_offset + b2bconf->picoseconds_ref;
	      sec = b2bconf->sec_ref + sec_offset;
	      if(!(picoseconds < 1E12))
	      	{
	      	  sec += 1;
	      	  picoseconds -= 1E12;
	      	}
	      strftime (utc_start, MSTR_LEN, DADA_TIMESTR, gmtime(&sec)); // String start time without fraction second 
	      mjd_start = sec / SECDAY + MJD1970;                         // Float MJD start time without fraction second
	      for(i = 0; i < MSTR_LEN; i++)
		{
		  if(ra[i] == ' ')
		    ra[i] = ':';
		  if(dec[i] == ' ')
		    dec[i] = ':';
		}
	      
	      /* Register header */
	      hdrbuf_out = ipcbuf_get_next_write(hdr_out);
	      memcpy(hdrbuf_out, b2bconf->hdr, DADA_HDRSZ);  // Get a copy of the header
	      
	      if(!hdrbuf_out)
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error getting header_buf, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  
	      	  pthread_mutex_lock(&quit_mutex);
	      	  quit = 1;
	      	  pthread_mutex_unlock(&quit_mutex);
	      	  
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      
	      /* Setup DADA header with given values */
	      if(ascii_header_set(hdrbuf_out, "UTC_START", "%s", utc_start) < 0)  
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting UTC_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting UTC_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  
	      	  pthread_mutex_lock(&quit_mutex);
	      	  quit = 1;
	      	  pthread_mutex_unlock(&quit_mutex);
	      	  
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      
	      if(ascii_header_set(hdrbuf_out, "RA", "%s", ra) < 0)  
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting RA, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting RA, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  
	      	  pthread_mutex_lock(&quit_mutex);
	      	  quit = 1;
	      	  pthread_mutex_unlock(&quit_mutex);
	      	  
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      
	      if(ascii_header_set(hdrbuf_out, "DEC", "%s", dec) < 0)  
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting DEC, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting DEC, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  
	      	  pthread_mutex_lock(&quit_mutex);
	      	  quit = 1;
	      	  pthread_mutex_unlock(&quit_mutex);
	      	  
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      
	      if(ascii_header_set(hdrbuf_out, "SOURCE", "%s", source) < 0)  
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting SOURCE, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting SOURCE, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  
	      	  pthread_mutex_lock(&quit_mutex);
	      	  quit = 1;
	      	  pthread_mutex_unlock(&quit_mutex);
	      	  
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      	      
	      if(ascii_header_set(hdrbuf_out, "PICOSECONDS", "%"PRIu64, picoseconds) < 0)  
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting PICOSECONDS, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting PICOSECONDS, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  
	      	  pthread_mutex_lock(&quit_mutex);
	      	  quit = 1;
	      	  pthread_mutex_unlock(&quit_mutex);
	      	  
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}    
	      if(ascii_header_set(hdrbuf_out, "MJD_START", "%.10lf", mjd_start) < 0)
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error setting MJD_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error setting MJD_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  
	      	  pthread_mutex_lock(&quit_mutex);
	      	  quit = 1;
	      	  pthread_mutex_unlock(&quit_mutex);
	      	  
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      /* donot set header parameters anymore - acqn. doesn't start */
	      if(ipcbuf_mark_filled(hdr_out, DADA_HDRSZ) < 0)
	      	{
	      	  multilog(runtime_log, LOG_ERR, "Error header_fill, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  fprintf(stderr, "Error header_fill, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	      	  
	      	  pthread_mutex_lock(&quit_mutex);
	      	  quit = 1;
	      	  pthread_mutex_unlock(&quit_mutex);
	      	  
	      	  close(sock);
	      	  pthread_exit(NULL);
	      	  return NULL;
	      	}
	      fprintf(stdout, "HERE START-OF-DATA\n");
	    }
	}
      pthread_mutex_lock(&quit_mutex);
      quit_status = quit;
      pthread_mutex_unlock(&quit_mutex);
    }
  pthread_exit(NULL);
  return NULL;
}

int threads(conf_t *conf)
{
  int i, ret[2];
  pthread_t thread[2];
  pthread_attr_t attr;
  cpu_set_t cpus;
  
  if(!(conf->thread_bind == 0))
    {
      pthread_attr_init(&attr);  
      CPU_ZERO(&cpus);      
      CPU_SET(conf->cpus[0], &cpus);
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
      ret[0] = pthread_create(&thread[0], &attr, baseband2baseband, (void *)conf);
      pthread_attr_destroy(&attr);
      
      pthread_attr_init(&attr);  
      CPU_ZERO(&cpus);
      CPU_SET(conf->cpus[1], &cpus);
      pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
      ret[1] = pthread_create(&thread[1], &attr, control, (void *)conf);
      pthread_attr_destroy(&attr);
    }
  
  for(i = 0; i < 2; i++)   // Join threads and unbind cpus
    pthread_join(thread[i], NULL);
  
  return EXIT_SUCCESS;
}
