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
    int* array; 
  }queue_t; 
  
  queue_t* create_queue(unsigned capacity);
  int is_full(queue_t* queue);
  int is_empty(queue_t* queue);
  void enqueue(queue_t* queue, int item);
  int dequeue(queue_t* queue);
  int front(queue_t* queue);
  int rear(queue_t* queue);
  
#endif

#ifdef __cplusplus
} 
#endif
