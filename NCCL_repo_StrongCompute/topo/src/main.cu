#include "common.h"

// Taken from: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html

int main(int argc, char* argv[]) {

	ncclComm_t comms[2];

	//managing 2 device
	int nDev = 2;
	int size = 32 * 1024 * 1024;
	int devs[2] = {0,1};

	//allocating and initializing device buffers
	float** sendbuff = (float**)malloc(nDev * sizeof(float*));
	float** recvbuff = (float**)malloc(nDev * sizeof(float*));
	cudaStream_t* s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * nDev);

	for (int i = 0; i < nDev; ++i) {
		CUDACHECK(cudaSetDevice(i));
		CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
		CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
		CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
		CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
		CUDACHECK(cudaStreamCreate(s+i));
	}

	printf("Initialize NCCL\n");
	//initializing NCCL
	NCCLCHECK(ncclCommInitAll(comms, nDev, devs));


	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	checkCuda(cudaEventRecord(startEvent, 0));


	// calling NCCL communication API. Group API is required hwne using
	// multiple devices per thread
	printf("Calling NCCL All Reduce\n");
	NCCLCHECK(ncclGroupStart());
	for (int i = 0; i < nDev; ++i) {
		NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]));
	}
	NCCLCHECK(ncclGroupEnd());



	checkCuda( cudaEventRecord(stopEvent, 0));
	checkCuda( cudaEventSynchronize(stopEvent));

	//synchronizing on CUDA streams to wait for completion of NCCL operation
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	printf("Synchronize\n");
	for (int i = 0; i < nDev; ++i) {
		printf("\tset device %d\n", i);
		CUDACHECK(cudaSetDevice(i));
		printf("\tstream synchronize %d\n", i);
		CUDACHECK(cudaStreamSynchronize(s[i]));
	}

	printf("Free\n");
	//free device buffers
	for (int i = 0; i < nDev; ++i) {
		CUDACHECK(cudaSetDevice(i));
		CUDACHECK(cudaFree(sendbuff[i]));
		CUDACHECK(cudaFree(recvbuff[i]));
	}


	//finalizing NCCL
	printf("destroying nccl communicators\n");
	for(int i = 0; i < nDev; ++i)
		ncclCommDestroy(comms[i]);

	printf("Success!\n");

	return 0;
}
