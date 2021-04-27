DEBUG ?= 1
ifeq ($(DEBUG), 1)
    C_FLAGS=-DDEBUG
else
    C_FLAGS=-DNDEBUG
endif

C_FLAGS+=-g

CU_FLAGS=-Wno-deprecated-gpu-targets --default-stream per-thread

NVCC=nvcc
CC=gcc
GXX=g++
TARGET=capture_main
SRC_DIR=.
OBJ_DIR=.

LIBS= -lpsrdada -lcudart -lm -lrt -lcufft -lpthread
LIB_DIRS= -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -L/usr/local/lib
INCLUDE_DIRS= -I/usr/local/include -I/usr/local/cuda/include -I/usr/local/include/psrdada

paf_capture_cli:capture_main.o capture.o log.o
	$(CC) -o paf_capture_cli capture_main.o capture.o log.o $(LIB_DIRS) $(LIBS) -fopenmp

%.o: %.c
	$(CC) $(INCLUDE_DIRS) -c $< -o $@
install:
	cp paf_capture_cli /tmp/
	sudo cp /tmp/paf_capture_cli /usr/local/bin
	rm /tmp/paf_capture_cli
clean:
	rm *.o paf_capture_cli
	sudo rm /usr/local/bin/paf_capture_cli
