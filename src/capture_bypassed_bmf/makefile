DEBUG ?= 1
ifeq ($(DEBUG), 1)
    C_FLAGS = -DDEBUG
else
    C_FLAGS = -DNDEBUG
endif

C_FLAGS      += -g
#CU_FLAGS     = -rdc=true -Wno-deprecated-gpu-targets -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52
#CU_FLAGS     = -rdc=true -Wno-deprecated-gpu-targets --default-stream per-thread
CU_FLAGS     = -Wno-deprecated-gpu-targets --default-stream per-thread #-arch=sm_30 \
 -gencode=arch=compute_20,code=sm_20 \
 -gencode=arch=compute_30,code=sm_30 \
 -gencode=arch=compute_50,code=sm_50 \
 -gencode=arch=compute_52,code=sm_52 \
 -gencode=arch=compute_60,code=sm_60 \
 -gencode=arch=compute_61,code=sm_61 \
 -gencode=arch=compute_61,code=compute_61 \
--ptxas-options=-v

NVCC         = nvcc
CC	     = gcc
GXX          = g++
TARGET       = capture_main
SRC_DIR      = .
OBJ_DIR      = .

LIBS         = -lpsrdada -lcudart -lm -lrt -lcufft -lpthread
LIB_DIRS     = -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu
INCLUDE_DIRS = -I/usr/local/include

capture_main:capture_main.o capture.o hdr.o control.o
	$(CC) -o capture_main capture_main.o capture.o log.o $(LIB_DIRS) $(LIBS)

capture_main.o:capture_main.c
	$(CC) -c capture_main.c $(INCLUDE_DIRS) ${C_FLAGS}

capture.o:capture.c
	$(CC) -c capture.c $(INCLUDE_DIRS) ${C_FLAGS}

log.o:log.c
	$(CC) -c log.c $(INCLUDE_DIRS) ${C_FLAGS}

clean:
	rm -f *.o *~
