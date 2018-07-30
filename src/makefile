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
TARGET       = paf_capture, 
SRC_DIR      = .
OBJ_DIR      = .

LIBS         = -lpsrdada -lcudart -lcuda -lm -lrt -lcufft -lpthread
LIB_DIRS     = -L/usr/local/cuda/lib64 -L/home/pulsar/.local/lib
INCLUDE_DIRS = -I/home/pulsar/.local/include

all:paf_capture #paf_process  paf_diskdb

paf_capture:paf_capture.o capture.o hdr.o sync.o
	$(CC) -o paf_capture paf_capture.o capture.o hdr.o sync.o $(LIB_DIRS) $(LIBS) 

paf_capture.o:paf_capture.c
	$(CC) -c paf_capture.c $(INCLUDE_DIRS) ${C_FLAGS}

capture.o:capture.c
	$(CC) -c capture.c $(INCLUDE_DIRS) ${C_FLAGS}

hdr.o:hdr.c
	$(CC) -c hdr.c $(INCLUDE_DIRS) ${C_FLAGS}

sync.o:sync.c
	$(CC) -c sync.c $(INCLUDE_DIRS) ${C_FLAGS}

#paf_process:cudautil.o kernel.o paf_process.o process.o
#	$(NVCC) -o paf_process cudautil.o kernel.o paf_process.o process.o $(LIB_DIRS) $(LIBS) 
#
#paf_process.o:paf_process.cu
#	$(NVCC) -c paf_process.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}
#
#kernel.o:kernel.cu
#	$(NVCC) -c kernel.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}
#
#cudautil.o:cudautil.cu
#	$(NVCC) -c cudautil.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}
#
#process.o:process.cu
#	$(NVCC) -c process.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}
#
#paf_diskdb:paf_diskdb.o diskdb.o cudautil.o
#	$(NVCC) -o paf_diskdb paf_diskdb.o diskdb.o cudautil.o $(LIB_DIRS) $(LIBS) 
#
#paf_diskdb.o:paf_diskdb.cu
#	$(NVCC) -c paf_diskdb.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}
#
#diskdb.o:diskdb.cu
#	$(NVCC) -c diskdb.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}
#
#cudautil.o:cudautil.cu
#	$(NVCC) -c cudautil.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

clean:
	rm -f *.o *~
