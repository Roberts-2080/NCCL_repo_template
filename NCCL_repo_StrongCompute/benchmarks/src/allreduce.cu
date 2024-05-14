#include "common.h" 


void nccl_allreduce(metric_t* m, benchmark_t *args, float** sendbuff, float** recvbuff, size_t size) 
{

 	float _time_;

	cudaEvent_t startEvent, stopEvent; 
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));
	checkCuda(cudaEventRecord(startEvent, 0));

	NCCLCHECK(ncclGroupStart());
	for (int i = 0; i < args->nGpus; ++i) {
		NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], 
					(void*)recvbuff[i], 
					size, 
					ncclFloat, 
					ncclSum,
					args->comms[i], 
					args->stream[i]));
	}
  	NCCLCHECK(ncclGroupEnd());


	for (int i = 0; i < args->nGpus; ++i) {
		CUDACHECK(cudaSetDevice(i));
		CUDACHECK(cudaStreamSynchronize(args->stream[i]));
	}

	checkCuda( cudaEventRecord(stopEvent, 0));
	checkCuda( cudaEventSynchronize(stopEvent));

	checkCuda( cudaEventElapsedTime(&_time_, startEvent, stopEvent)); 
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));

	m->time += _time_;
	m->bandwidth += size * sizeof(float)*1e-6/_time_;

}


int main(int argc, char* argv[]) {
	benchmark_t *args = (benchmark_t*)malloc(sizeof(benchmark_t));
	args->nGpus = 2;
	size_t nElements = 1024 * 1024 * 32;
	int bytes = nElements * sizeof(float);
	int devs[2] = {0,1};

	float** sendbuff = (float**)malloc(args->nGpus * sizeof(float*));
	float** recvbuff = (float**)malloc(args->nGpus * sizeof(float*));

	for (int i = 0; i < args->nGpus; ++i) {
		CUDACHECK(cudaSetDevice(i));
		CUDACHECK(cudaMalloc(sendbuff + i, bytes));
		CUDACHECK(cudaMalloc(recvbuff + i, bytes));
		CUDACHECK(cudaMemset(sendbuff[i], 1, bytes));
		CUDACHECK(cudaMemset(recvbuff[i], 0, bytes));
		CUDACHECK(cudaStreamCreate(args->stream+i));
	}

	NCCLCHECK(ncclCommInitAll(args->comms, args->nGpus, devs));

	metric_t* metrics = (metric_t*)calloc(sizeof(metric_t), 128);
	const int iterations = 100;

	for (int k = 0; k < iterations; k++) {
		int j = 0;
		for (size_t i = 32; i <= nElements; i*=2) {
			nccl_allreduce(metrics+j, args, sendbuff, recvbuff, i);
			j++;
		}
	}

	printf("\n===============AVG NCCL ALLREDUCE BENCHMARK==================\n\n");

	int j = 0;
	for (size_t i = 32; i <= nElements; i*=2) {
		(metrics+j)->time /= iterations;
		(metrics+j)->bandwidth /= iterations;
		printf("\t[AVG Bandwidth (GB/s): %f] took %f", (metrics+j)->bandwidth, (metrics+j)->time);
		printf(" >> %ld bytes\n", i*sizeof(float));
		j++;
	}
	



	for (int i = 0; i < args->nGpus; ++i) {
		CUDACHECK(cudaSetDevice(i));
		CUDACHECK(cudaFree(sendbuff[i]));
		CUDACHECK(cudaFree(recvbuff[i]));
	}

	for(int i = 0; i < args->nGpus; ++i)
		ncclCommDestroy(args->comms[i]);

	free(sendbuff);
	free(recvbuff);
	free(metrics);
	
	return 0;
}
