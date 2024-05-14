#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdio.h>
#include <pthread.h>
#include <nccl.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>

#include "cuda_runtime.h"

#define CUDACHECK(cmd) do {                         \
	cudaError_t err = cmd;                            \
	if (err != cudaSuccess) {                         \
		printf("Failed: Cuda error %s:%d '%s'\n",       \
				__FILE__,__LINE__,cudaGetErrorString(err)); \
		exit(EXIT_FAILURE);                             \
	}                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
	ncclResult_t res = cmd;                           \
	if (res != ncclSuccess) {                         \
		printf("Failed, NCCL error %s:%d '%s'\n",       \
				__FILE__,__LINE__,ncclGetErrorString(res)); \
		exit(EXIT_FAILURE);                             \
	}                                                 \
} while(0)

#define MAX_GPUS 8


inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
	  if (result != cudaSuccess) {
		    fprintf(stderr, "CUDA Runtime Error: %s\n", 
			    cudaGetErrorString(result));
		    assert(result == cudaSuccess);
	  }
#endif
  	  return result;
}

typedef struct benchmark_t {
		int nGpus;
		size_t size;
		int devs[MAX_GPUS];
		ncclComm_t comms[MAX_GPUS];
		cudaStream_t stream[MAX_GPUS];
} benchmark_t;

typedef struct metric_t {
		float time;
		float bandwidth;
} metric_t;

#endif
