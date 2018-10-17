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
LIB_DIRS     = -L/usr/local/cuda/lib64 -L/home/pulsar/.local/lib
INCLUDE_DIRS = -I/home/pulsar/.local/include

baseband2spectral_main:cudautil.o kernel.o baseband2spectral_main.o baseband2spectral.o
	$(NVCC) -o baseband2spectral_main cudautil.o kernel.o baseband2spectral_main.o baseband2spectral.o $(LIB_DIRS) $(LIBS) 

baseband2spectral_main.o:baseband2spectral_main.cu
	$(NVCC) -c baseband2spectral_main.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

kernel.o:kernel.cu
	$(NVCC) -c kernel.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

cudautil.o:cudautil.cu
	$(NVCC) -c cudautil.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

baseband2spectral.o:baseband2spectral.cu
	$(NVCC) -c baseband2spectral.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

cudautil.o:cudautil.cu
	$(NVCC) -c cudautil.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

clean:
	rm -f *.o *~

