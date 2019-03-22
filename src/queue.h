#ifdef __cplusplus
extern "C" {
#endif
#ifndef _QUEUE_H
#define _QUEUE_H
  
#include "fits.h"
  
  // A structure to represent a queue 
  typedef struct queue_t
  { 
    int front, rear, size; 
    unsigned capacity; 
    fits_t* fits; 
  }queue_t; 
  
  queue_t* create_queue(unsigned capacity);
  int is_full(queue_t* queue);
  int is_empty(queue_t* queue);
  int enqueue(queue_t* queue, fits_t fits);
  int dequeue(queue_t* queue, fits_t *fits);
  int front(queue_t* queue, fits_t *fits);
  int rear(queue_t* queue, fits_t *fits);
  int destroy_queue(queue_t queue) ;
  
#endif

#ifdef __cplusplus
} 
#endif
