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
SRC_DIR      = .
OBJ_DIR      = .

LIBS         = -lpsrdada -lcudart -lcuda -lm -lrt -lcufft -lpthread
LIB_DIRS     = -L/usr/local/cuda/lib64 -L/usr/local/lib 
INCLUDE_DIRS = -I/usr/local/include 

all: capture_main baseband2filterbank_main dada2filterbank_main clean

capture_main:capture_main.o capture.o log.o
	$(CC) -o capture_main capture_main.o capture.o log.o $(LIB_DIRS) $(LIBS)

capture_main.o:capture_main.c
	$(CC) -c capture_main.c $(INCLUDE_DIRS) ${C_FLAGS}

capture.o:capture.c
	$(CC) -c capture.c $(INCLUDE_DIRS) ${C_FLAGS}

dada2filterbank_main:dada2filterbank_main.o dada2filterbank.o log.o
	$(CC) -o dada2filterbank_main dada2filterbank_main.o dada2filterbank.o log.o $(LIB_DIRS) $(LIBS)

dada2filterbank_main.o:dada2filterbank_main.c
	$(CC) -c dada2filterbank_main.c $(INCLUDE_DIRS) ${C_FLAGS}

dada2filterbank.o:dada2filterbank.c
	$(CC) -c dada2filterbank.c $(INCLUDE_DIRS) ${C_FLAGS}

baseband2filterbank_main:cudautil.o kernel.o baseband2filterbank_main.o baseband2filterbank.o log.o
	$(NVCC) -o baseband2filterbank_main cudautil.o kernel.o baseband2filterbank_main.o baseband2filterbank.o log.o $(LIB_DIRS) $(LIBS) 

baseband2filterbank_main.o:baseband2filterbank_main.cu
	$(NVCC) -c baseband2filterbank_main.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

kernel.o:kernel.cu
	$(NVCC) -c kernel.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

cudautil.o:cudautil.cu
	$(NVCC) -c cudautil.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

baseband2filterbank.o:baseband2filterbank.cu
	$(NVCC) -c baseband2filterbank.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

log.o:log.c
	$(CC) -c log.c $(INCLUDE_DIRS) ${C_FLAGS}

clean:
	rm -f *.o *~

