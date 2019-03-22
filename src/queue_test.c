#ifndef _GNU_SOURCE 
#define _GNU_SOURCE 
#endif 

// C program for array implementation of queue 
// Ref: https://www.geeksforgeeks.org/queue-set-1introduction-and-array-implementation/ 
#include <stdio.h>  
#include <stdlib.h> 
#include <limits.h> 
#include "queue.h" 

// Driver program to test above functions./  */
int main()  
{ 
  fits_t fits; 
  queue_t* queue = create_queue(100);
   
  fits.nchan = 10; 
  enqueue(queue, fits);
  fits.nchan = 100;
  enqueue(queue, fits);
  fits.nchan = 1000;
  enqueue(queue, fits);
  fits.nchan = 10000;
  enqueue(queue, fits); 
  
  fits = dequeue(queue);
  printf("%d \n", fits.nchan); 
  
  fits = front(queue);
  printf("%d\n", fits.nchan);
  
  fits = rear(queue);
  printf("%d\n", fits.nchan);
  
  destroy_queue(*queue);
  
  return 0; 
} 
