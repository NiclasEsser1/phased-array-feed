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

power2udp_main:power2udp_main.o power2udp.o jsmn.o
	$(CC) -o power2udp_main power2udp_main.o power2udp.o jsmn.o $(LIB_DIRS) $(LIBS) 

power2udp_main.o:power2udp_main.c
	$(CC) -c power2udp_main.c $(INCLUDE_DIRS) ${C_FLAGS}

power2udp.o:power2udp.c
	$(CC) -c power2udp.c $(INCLUDE_DIRS) ${C_FLAGS}

jsmn.o:jsmn.c
	$(CC) -c jsmn.c $(INCLUDE_DIRS) ${C_FLAGS}

clean:
	rm -f *.o *~
