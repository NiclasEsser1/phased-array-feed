#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <arpa/inet.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#include "constants.h"

void usage ()
{
  fprintf (stdout,
	   "fits_test - Test the fits kernel \n"
	   "\n"
	   "Usage: fits_test [options]\n"
	   " -a  Network interface\n"
	   " -b  Number of channels\n"
	   " -c  Type of pol\n"
	   " -d  Number of frequency chunks\n"
	   " -e  File name to record the data, for compare\n"
	   " -f  beam index\n"
	   " -g  center frequency in MHz\n"
	   " -h  show help\n"
	   " -i  channel width in MHz\n"
	   );
}

typedef struct fits_t
{
  int beam_index;
  char time_stamp[FITS_TIME_STAMP_LEN];
  float tsamp;
  int nchan;
  float cfreq;
  float chan_wdith;
  int pol_type;
  int pol_index;
  int nchunk;
  int chunk_index;
  float data[UDP_PAYLOAD_SIZE_MAX]; // Can not alloc dynamic
}fits_t;

// ./fitswriter_interface_test -a n_134.104.70.90_17106 -b 199584 -c 4 -d 231 -e spectral.txt -f 12 -g 1340.5 -i 0.001157407407 -j 1.769472

int main(int argc, char *argv[])
{
  int arg;
  int i, j;
  int enable = 1;
  int port_udp, nchan_per_chunk;
  char ip_udp[MSTR_LEN] = {'\0'};
  char fname[MSTR_LEN] = {'\0'};
  float *spectral_buffer;
  int spectral_buffer_size;
  FILE *fp = NULL;
  int sock_udp;
  struct sockaddr_in sa_udp;
  socklen_t tolen = sizeof(sa_udp);
  int data_size, net_pktsz;
  fits_t fits;
  time_t now;
  int index;
  struct timespec start, stop;
  double elapsed_time;

  /* Initializeial part */  
  while((arg=getopt(argc,argv,"a:b:c:d:he:f:g:i:j:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);
	  	  
	case 'a':
	  if(sscanf(optarg, "%*[^_]_%[^_]_%d", ip_udp, &port_udp) != 2)
	    {		  
	      fprintf (stderr, "BASEBAND2SPECTRAL_ERROR:Can not get output network configuration\t, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
	      exit(EXIT_FAILURE);
	    }
	  break;
	  
	case 'b':
	  sscanf(optarg, "%d", &fits.nchan);
	  break;
	  
	case 'c':
	  sscanf(optarg, "%d", &fits.pol_type);
	  break;

	case 'd':
	  sscanf(optarg, "%d", &fits.nchunk);
	  break;
	  
	case 'e':
	  sscanf(optarg, "%s", fname);
	  break;
	  
	case 'f':
	  sscanf(optarg, "%d", &fits.beam_index);
	  break;
	  
	case 'g':
	  sscanf(optarg, "%f", &fits.cfreq);
	  break;
	  
	case 'i':
	  sscanf(optarg, "%f", &fits.chan_wdith);
	  break;
	  
	case 'j':
	  sscanf(optarg, "%f", &fits.tsamp);
	  break;
      	}
    }
  nchan_per_chunk = fits.nchan/fits.nchunk;
  data_size = NBYTE_FLOAT * nchan_per_chunk;
  net_pktsz = data_size + FITS_TIME_STAMP_LEN + 6 * NBYTE_INT + 3 * NBYTE_FLOAT;  // be careful here    
  fprintf(stdout, "The configuration is: ip %s, port %d, nchan %d, pol_type %d, nchunk %d, nchan_per_chunk is %d, data_size is %d, net_pktsz is %d, file name %s, beam index is %d, center frequency is %f MHz and channel width is %f MHz\n", ip_udp, port_udp, fits.nchan, fits.pol_type, fits.nchunk, nchan_per_chunk, data_size, net_pktsz, fname, fits.beam_index, fits.cfreq, fits.chan_wdith);

  /* 
     Create spectral_buffer
     fill it with random numbers
     record the number to the file
     create network interface
  */
  srand(time(NULL));
  spectral_buffer_size = NBYTE_FLOAT * fits.nchan * NDATA_PER_SAMP_FULL;
  //spectral_buffer = (float *)malloc(spectral_buffer_size);  // Always create a spectral_buffer for 4 pols
  spectral_buffer = (float *)calloc(fits.nchan * NDATA_PER_SAMP_FULL, NBYTE_FLOAT);  // Always create a spectral_buffer for 4 pols
  fp = fopen(fname, "w");
  for(i = 0; i < fits.pol_type; i++)
    {
      for(j = 0; j < fits.nchan; j++)
	{
	  spectral_buffer[i * fits.nchan + j] = (float)rand()/(float)(RAND_MAX/(float)MAX_RAND);
	  fprintf(fp, "%f\n", spectral_buffer[i * fits.nchan + j]);
	}
    }
  if((sock_udp = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
    {
      fprintf(stderr, "socket creation failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      return EXIT_FAILURE;
    }
  memset((char *) &sa_udp, 0, sizeof(sa_udp));
  sa_udp.sin_family      = AF_INET;
  sa_udp.sin_port        = htons(port_udp);
  sa_udp.sin_addr.s_addr = inet_addr(ip_udp);
  setsockopt(sock_udp, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
  
  /* 
     send the data
  */
  index = 0;
  while(1)
    {
      clock_gettime(CLOCK_REALTIME, &start);
      fprintf(stdout, "sending %d spectral sample ...\n", index);
      
      time(&now);
      strftime(fits.time_stamp, FITS_TIME_STAMP_LEN, FITS_TIMESTR, gmtime(&now));
      sprintf(fits.time_stamp, "%s.0000UTC ", fits.time_stamp);
      for(i = 0; i < NDATA_PER_SAMP_FULL; i++)
	{
	  fits.pol_index = i;
	  for(j = 0; j < fits.nchunk; j++)
	    {
	      fits.chunk_index = j;
	      memcpy(fits.data, &spectral_buffer[i * fits.nchan + j * nchan_per_chunk], data_size);
	      
	      if(sendto(sock_udp, (void *)&fits, net_pktsz, 0, (struct sockaddr *)&sa_udp, tolen) == -1)
		{
		  fprintf(stderr, "sendto() failed, which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
		  return EXIT_FAILURE;
		}
	    }
	}
      clock_gettime(CLOCK_REALTIME, &stop);
      elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
      fprintf(stdout, "elapse_time for spectral sending is %f\n", elapsed_time);
      fflush(stdout);
      sleep(fits.tsamp);
      index ++;
    }
  
  /* 
     Free spectral_buffer 
     close socket 
  */
  fclose(fp);
  free(spectral_buffer);
  close(sock_udp);
  
  return EXIT_SUCCESS;
}
