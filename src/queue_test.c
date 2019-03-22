// C program for array implementation of queue
// Ref: https://www.geeksforgeeks.org/queue-set-1introduction-and-array-implementation/
#include <stdio.h> 
#include <stdlib.h> 
#include <limits.h> 
#include "queue.h"

// Driver program to test above functions./ 
int main() 
{ 
  queue_t* queue = create_queue(1000); 
  
  enqueue(queue, 10); 
  enqueue(queue, 20); 
  enqueue(queue, 30); 
  enqueue(queue, 40); 
  
  printf("%d dequeued from queue\n\n", dequeue(queue)); 
  
  printf("Front item is %d\n", front(queue)); 
  printf("Rear item is %d\n", rear(queue)); 
  
  return 0; 
} 
