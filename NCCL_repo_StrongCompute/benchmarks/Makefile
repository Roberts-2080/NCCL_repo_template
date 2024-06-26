
CUDA_HOME ?= /usr/local/cuda
PREFIX ?= /usr/local
VERBOSE ?= 0
DEBUG ?= 0

CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_INC ?= $(CUDA_HOME)/include
NVCC = $(CUDA_HOME)/bin/nvcc
CUDARTLIB ?= cudart 
CUDA_VERSION = $(strip $(shell which $(NVCC) >/dev/null && $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/\,.*//'))
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)

# Better define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 11; echo $$?),0)
NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
                -gencode=arch=compute_80,code=sm_80 \
                -gencode=arch=compute_80,code=compute_80
else
NVCC_GENCODE ?= -gencode=arch=compute_35,code=sm_35 \
                -gencode=arch=compute_50,code=sm_50 \
                -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
                -gencode=arch=compute_70,code=compute_70
endif

NVCUFLAGS  := -ccbin $(CXX) $(NVCC_GENCODE) -std=c++11

LDFLAGS    := -L${CUDA_LIB} -lcudart -lrt
NVLDFLAGS  := -L${CUDA_LIB} -l${CUDARTLIB} -lrt

ifeq ($(DEBUG), 0)
NVCUFLAGS += -O3 -g
CXXFLAGS  += -O3 -g
else
NVCUFLAGS += -O0 -G -g
CXXFLAGS  += -O0 -g -ggdb3
endif

ifneq ($(VERBOSE), 0)
NVCUFLAGS += -Xcompiler -Wall,-Wextra,-Wno-unused-parameter
else
.SILENT:
endif

.PHONY: build clean

BUILDDIR ?= ./build
ifneq ($(NCCL_HOME), "")
NVCUFLAGS += -I$(NCCL_HOME)/include/
NVLDFLAGS += -L$(NCCL_HOME)/lib
endif

ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -L$(MPI_HOME)/lib64 -lmpi
endif
ifeq ($(MPI_IBM),1)
NVCUFLAGS += -DMPI_SUPPORT
NVLDFLAGS += -lmpi_ibm
endif
LIBRARIES += nccl
NVLDFLAGS += $(LIBRARIES:%=-l%)

DST_DIR := $(BUILDDIR)
SRC := allreduce.cu cudamemcpy.cu pinnedtransfer.cu naivereduce.cu ringreduce.cu unifiedreduce.cu

SRC_FILES := $(wildcard ./src/*.cu)
OBJ_FILES := $(SRC:%.cu=${DST_DIR}/%.o)
BIN_FILES := $(SRC:%.cu=${DST_DIR}/%)

build: ${BIN_FILES}

run: 
	@echo Running tests:
	./$(BUILDDIR)/cudamemcpy
	./$(BUILDDIR)/pinnedtransfer
	./$(BUILDDIR)/allreduce
	./$(BUILDDIR)/naivereduce
	./$(BUILDDIR)/ringreduce
	./$(BUILDDIR)/unifiedreduce

cuda: ${DST_DIR}/cudamemcpy
	./$(BUILDDIR)/cudamemcpy
pinned: ${DST_DIR}/pinnedtransfer
	./$(BUILDDIR)/pinnedtransfer
reduce: ${DST_DIR}/allreduce
	./$(BUILDDIR)/allreduce
naive: ${DST_DIR}/naivereduce
	./$(BUILDDIR)/naivereduce
ring: ${DST_DIR}/ringreduce
	./$(BUILDDIR)/ringreduce
unified: ${DST_DIR}/unifiedreduce
	./$(BUILDDIR)/unifiedreduce
clean:
	rm -rf ${DST_DIR}


${DST_DIR}/%.o: ./src/%.cu ./src/common.h
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/%: ${DST_DIR}/%.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}

