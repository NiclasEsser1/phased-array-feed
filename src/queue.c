// C program for fits implementation of queue
// Ref: https://www.geeksforgeeks.org/queue-set-1introduction-and-fits-implementation/
#include <stdio.h> 
#include <stdlib.h> 
#include <limits.h>
#include "queue.h"

extern int quit;

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
int enqueue(queue_t* queue, fits_t fits) 
{ 
  if (is_full(queue))
    exit(EXIT_FAILURE);
  
  queue->rear = (queue->rear + 1)%queue->capacity; 
  queue->fits[queue->rear] = fits; 
  queue->size = queue->size + 1; 
} 

// Function to remove an fits from queue.  
// It changes front and size 
int dequeue(queue_t* queue, fits_t *fits) 
{ 
  if (is_empty(queue))
    {
      fprintf(stdout, "The queue is EMPTY!\n");
      exit(EXIT_FAILURE);
    }
  *fits = queue->fits[queue->front]; 
  queue->front = (queue->front + 1)%queue->capacity; 
  queue->size = queue->size - 1;
  
  return EXIT_SUCCESS;
} 

// Function to get front of queue 
int front(queue_t* queue, fits_t *fits) 
{ 
  if (is_empty(queue))
    {
      fprintf(stdout, "The queue is EMPTY!\n");
      exit(EXIT_FAILURE);
    }
  *fits = queue->fits[queue->front];
  return EXIT_SUCCESS;
} 
  
// Function to get rear of queue 
int rear(queue_t* queue, fits_t *fits) 
{ 
  if (is_empty(queue))
    {
      fprintf(stdout, "The queue is EMPTY!\n");
      exit(EXIT_FAILURE);
    }
  *fits = queue->fits[queue->rear];
  
  return EXIT_SUCCESS;
} 
