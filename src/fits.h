#ifdef __cplusplus
extern "C" {
#endif
#ifndef _FITS_H
#define _FITS_H

#include "constants.h"
  
  typedef struct fits_t
  {
    int beam_index;
    char time_stamp[FITS_TIME_STAMP_LEN];
    float tsamp;
    int nchan;
    float center_freq;
    float chan_width;
    int pol_type;
    int pol_index;
    int nchunk;
    int chunk_index;
    float data[UDP_PAYLOAD_SIZE_MAX]; // Can not alloc dynamic
  }fits_t;
  
#endif
  
#ifdef __cplusplus
} 
#endif
