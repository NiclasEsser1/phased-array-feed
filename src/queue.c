// C program for fits implementation of queue
// Ref: https://www.geeksforgeeks.org/queue-set-1introduction-and-fits-implementation/
#include <stdio.h> 
#include <stdlib.h> 
#include <limits.h>
#include "queue.h"

// function to create a queue of given capacity.  
// It initializes size of queue as 0 
queue_t* create_queue(unsigned capacity) 
{ 
  queue_t* queue = (queue_t*) malloc(sizeof(queue_t)); 
  queue->capacity = capacity; 
  queue->front = queue->size = 0;  
  queue->rear = capacity - 1;  // This is important, see the enqueue 
  queue->fits = (fits_t*) malloc(queue->capacity * sizeof(fits_t)); 
  return queue; 
} 

// function to destroy a queue of given capacity.  
int destroy_queue(queue_t queue) 
{
  if(queue.fits)
    free(queue.fits);
  
  return EXIT_SUCCESS;
} 


// Queue is full when size becomes equal to the capacity  
int is_full(queue_t* queue) 
{
  return (queue->size == queue->capacity);
}

// Queue is empty when size is 0 
int is_empty(queue_t* queue) 
{
  return (queue->size == 0);
} 

// Function to add an fits to the queue.   
// It changes rear and size 
void enqueue(queue_t* queue, fits_t fits) 
{ 
  if (is_full(queue)) 
    return; 
  queue->rear = (queue->rear + 1)%queue->capacity; 
  queue->fits[queue->rear] = fits; 
  queue->size = queue->size + 1; 
} 

// Function to remove an fits from queue.  
// It changes front and size 
fits_t dequeue(queue_t* queue) 
{ 
  if (is_empty(queue))
    {
      fprintf(stdout, "The queue is EMPTY!\n");
      exit(EXIT_FAILURE);
    }
  fits_t fits = queue->fits[queue->front]; 
  queue->front = (queue->front + 1)%queue->capacity; 
  queue->size = queue->size - 1; 
  return fits; 
} 

// Function to get front of queue 
fits_t front(queue_t* queue) 
{ 
  if (is_empty(queue))
    {
      fprintf(stdout, "The queue is EMPTY!\n");
      exit(EXIT_FAILURE);
    }
  return queue->fits[queue->front]; 
} 
  
// Function to get rear of queue 
fits_t rear(queue_t* queue) 
{ 
  if (is_empty(queue))
    {
      fprintf(stdout, "The queue is EMPTY!\n");
      exit(EXIT_FAILURE);
    }
  return queue->fits[queue->rear]; 
} 
