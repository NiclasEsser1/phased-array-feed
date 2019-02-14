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

all: capture_main baseband2filterbank_main baseband2spectral_main dada2filterbank_main reduce6_test reduce7_test reduce8_test reduce9_test scale_test scale1_test scale2_test accumulate_test unpack_test swap_select_transpose_ptf_test swap_select_transpose_pft_test swap_select_transpose_pft1_test spectrum_taccumulate_test detect_faccumulate_scale2_test detect_faccumulate_scale1_test detect_faccumulate_scale_test detect_faccumulate_pad_transpose1_test detect_faccumulate_pad_transpose_test clean

capture_main:capture_main.o capture.o log.o
	$(CC) -o capture_main capture_main.o capture.o log.o $(LIB_DIRS) $(LIBS)

dada2filterbank_main:dada2filterbank_main.o dada2filterbank.o log.o
	$(CC) -o dada2filterbank_main dada2filterbank_main.o dada2filterbank.o log.o $(LIB_DIRS) $(LIBS)

baseband2filterbank_main:baseband2filterbank_main.o baseband2filterbank.o kernel.o log.o
	$(NVCC) -o baseband2filterbank_main baseband2filterbank_main.o baseband2filterbank.o kernel.o log.o $(LIB_DIRS) $(LIBS)

baseband2spectral_main:baseband2spectral_main.o baseband2spectral.o kernel.o log.o
	$(NVCC) -o baseband2spectral_main baseband2spectral_main.o baseband2spectral.o kernel.o log.o $(LIB_DIRS) $(LIBS)

reduce6_test:reduce6_test.o kernel.o 
	$(NVCC) -o reduce6_test reduce6_test.o kernel.o $(LIB_DIRS) $(LIBS)

reduce7_test:reduce7_test.o kernel.o 
	$(NVCC) -o reduce7_test reduce7_test.o kernel.o $(LIB_DIRS) $(LIBS)

reduce8_test:reduce8_test.o kernel.o 
	$(NVCC) -o reduce8_test reduce8_test.o kernel.o $(LIB_DIRS) $(LIBS)

reduce9_test:reduce9_test.o kernel.o 
	$(NVCC) -o reduce9_test reduce9_test.o kernel.o $(LIB_DIRS) $(LIBS)

scale_test:scale_test.o kernel.o 
	$(NVCC) -o scale_test scale_test.o kernel.o $(LIB_DIRS) $(LIBS)

scale1_test:scale1_test.o kernel.o 
	$(NVCC) -o scale1_test scale1_test.o kernel.o $(LIB_DIRS) $(LIBS)

scale2_test:scale2_test.o kernel.o 
	$(NVCC) -o scale2_test scale2_test.o kernel.o $(LIB_DIRS) $(LIBS)

accumulate_test:accumulate_test.o kernel.o 
	$(NVCC) -o accumulate_test accumulate_test.o kernel.o $(LIB_DIRS) $(LIBS)

unpack_test:unpack_test.o kernel.o 
	$(NVCC) -o unpack_test unpack_test.o kernel.o $(LIB_DIRS) $(LIBS)

swap_select_transpose_ptf_test:swap_select_transpose_ptf_test.o kernel.o 
	$(NVCC) -o swap_select_transpose_ptf_test swap_select_transpose_ptf_test.o kernel.o $(LIB_DIRS) $(LIBS)

spectrum_taccumulate_test:spectrum_taccumulate_test.o kernel.o 
	$(NVCC) -o spectrum_taccumulate_test spectrum_taccumulate_test.o kernel.o $(LIB_DIRS) $(LIBS)

swap_select_transpose_pft_test:swap_select_transpose_pft_test.o kernel.o 
	$(NVCC) -o swap_select_transpose_pft_test swap_select_transpose_pft_test.o kernel.o $(LIB_DIRS) $(LIBS)

swap_select_transpose_pft1_test:swap_select_transpose_pft1_test.o kernel.o 
	$(NVCC) -o swap_select_transpose_pft1_test swap_select_transpose_pft1_test.o kernel.o $(LIB_DIRS) $(LIBS)

detect_faccumulate_scale2_test:detect_faccumulate_scale2_test.o kernel.o 
	$(NVCC) -o detect_faccumulate_scale2_test detect_faccumulate_scale2_test.o kernel.o $(LIB_DIRS) $(LIBS)

detect_faccumulate_scale_test:detect_faccumulate_scale_test.o kernel.o 
	$(NVCC) -o detect_faccumulate_scale_test detect_faccumulate_scale_test.o kernel.o $(LIB_DIRS) $(LIBS)

detect_faccumulate_scale1_test:detect_faccumulate_scale1_test.o kernel.o 
	$(NVCC) -o detect_faccumulate_scale1_test detect_faccumulate_scale1_test.o kernel.o $(LIB_DIRS) $(LIBS)

detect_faccumulate_pad_transpose1_test:detect_faccumulate_pad_transpose1_test.o kernel.o 
	$(NVCC) -o detect_faccumulate_pad_transpose1_test detect_faccumulate_pad_transpose1_test.o kernel.o $(LIB_DIRS) $(LIBS)

detect_faccumulate_pad_transpose_test:detect_faccumulate_pad_transpose_test.o kernel.o 
	$(NVCC) -o detect_faccumulate_pad_transpose_test detect_faccumulate_pad_transpose_test.o kernel.o $(LIB_DIRS) $(LIBS)

capture_main.o:capture_main.c
	$(CC) -c capture_main.c $(INCLUDE_DIRS) ${C_FLAGS}

capture.o:capture.c
	$(CC) -c capture.c $(INCLUDE_DIRS) ${C_FLAGS}

baseband2filterbank_main.o:baseband2filterbank_main.cu
	$(NVCC) -c baseband2filterbank_main.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

baseband2filterbank.o:baseband2filterbank.cu
	$(NVCC) -c baseband2filterbank.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

baseband2spectral_main.o:baseband2spectral_main.cu
	$(NVCC) -c baseband2spectral_main.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

baseband2spectral.o:baseband2spectral.cu
	$(NVCC) -c baseband2spectral.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

detect_faccumulate_scale2_test.o:detect_faccumulate_scale2_test.cu
	$(NVCC) -c detect_faccumulate_scale2_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

detect_faccumulate_scale_test.o:detect_faccumulate_scale_test.cu
	$(NVCC) -c detect_faccumulate_scale_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

detect_faccumulate_scale1_test.o:detect_faccumulate_scale1_test.cu
	$(NVCC) -c detect_faccumulate_scale1_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

detect_faccumulate_pad_transpose1_test.o:detect_faccumulate_pad_transpose1_test.cu
	$(NVCC) -c detect_faccumulate_pad_transpose1_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

detect_faccumulate_pad_transpose_test.o:detect_faccumulate_pad_transpose_test.cu
	$(NVCC) -c detect_faccumulate_pad_transpose_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

reduce6_test.o:reduce6_test.cu
	$(NVCC) -c reduce6_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

reduce7_test.o:reduce7_test.cu
	$(NVCC) -c reduce7_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

reduce8_test.o:reduce8_test.cu
	$(NVCC) -c reduce8_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

reduce9_test.o:reduce9_test.cu
	$(NVCC) -c reduce9_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

scale_test.o:scale_test.cu
	$(NVCC) -c scale_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

scale1_test.o:scale1_test.cu
	$(NVCC) -c scale1_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

scale2_test.o:scale2_test.cu
	$(NVCC) -c scale2_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

swap_select_transpose_ptf_test.o:swap_select_transpose_ptf_test.cu
	$(NVCC) -c swap_select_transpose_ptf_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

swap_select_transpose_pft_test.o:swap_select_transpose_pft_test.cu
	$(NVCC) -c swap_select_transpose_pft_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

swap_select_transpose_pft1_test.o:swap_select_transpose_pft1_test.cu
	$(NVCC) -c swap_select_transpose_pft1_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

spectrum_taccumulate_test.o:spectrum_taccumulate_test.cu
	$(NVCC) -c spectrum_taccumulate_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

accumulate_test.o:accumulate_test.cu
	$(NVCC) -c accumulate_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

unpack_test.o:unpack_test.cu
	$(NVCC) -c unpack_test.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

kernel.o:kernel.cu
	$(NVCC) -c kernel.cu $(INCLUDE_DIRS) ${C_FLAGS} ${CU_FLAGS}

dada2filterbank_main.o:dada2filterbank_main.c
	$(CC) -c dada2filterbank_main.c $(INCLUDE_DIRS) ${C_FLAGS}

dada2filterbank.o:dada2filterbank.c
	$(CC) -c dada2filterbank.c $(INCLUDE_DIRS) ${C_FLAGS}

log.o:log.c
	$(CC) -c log.c $(INCLUDE_DIRS) ${C_FLAGS}

clean:
	rm -f *.o *~
