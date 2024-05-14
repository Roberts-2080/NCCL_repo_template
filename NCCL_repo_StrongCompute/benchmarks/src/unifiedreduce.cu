#include "common.h"

void inspect(float* buff, size_t nElements, int dev) {
	printf("%ld elements on gpu_%d unified\n\t", nElements, dev);
	cudaSetDevice(dev);
	for (int i = 0; i < nElements; ++i)  {
		if (buff[i] < 0) printf("%f\t",buff[i]); 
		else printf(" %f\t",buff[i]);
		if (i % 8 == 7) printf("\n\t");
	}
	
	printf("\n");
	cudaSetDevice(0);
}


__global__ void kernel(float* result, float* A, float *B, size_t N) {
	int i = threadIdx.x;
	if (N < 1024) 
		result[i] = A[i] + B[i];
	int steps = N / 1024;
	int idx = steps * i; 
	for (int k = 0; k < steps; k++) {
		result[idx] = A[idx] + B[idx];
		idx += 1;
	}
}

void unified_all_reduce(metric_t* m, float* A, float* B, float* result, size_t n) 
{
	float _time_;
	int half = n / 2;
	cudaEvent_t startEvent, stopEvent; 

	checkCuda( cudaEventCreate(&startEvent) );
	checkCuda( cudaEventCreate(&stopEvent) );
	checkCuda( cudaEventRecord(startEvent, 0) );

	cudaSetDevice(0);
	kernel<<<1, 1024>>>(result, A, B, half);

	cudaSetDevice(1);
	kernel<<<1, 1024>>>(result+half, A+half, B+half, half);

	cudaSetDevice(0);

	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&_time_, startEvent, stopEvent) );

	checkCuda( cudaEventDestroy(startEvent) );
	checkCuda( cudaEventDestroy(stopEvent) );

	m->time += _time_;
	m->bandwidth += n * sizeof(float)*1e-6/_time_;
}

int main()
{
	size_t nElements = 1024 * 1024 * 32;
	size_t bytes = nElements * sizeof(float);

	float *A, *B;
	float *result;

	//cudaSetDevice(0);
	cudaMallocManaged((void**)&A, bytes);

	//cudaSetDevice(1);
	cudaMallocManaged((void**)&B, bytes);

	//cudaSetDevice(0);
	cudaMallocManaged((void**)&result, bytes);


	//for (int i = 0; i < nElements; ++i) A[i] = 1.0f; 
	//for (int i = 0; i < nElements; ++i) B[i] = 2.0f; 

	//printf("\n================UNIFIED ALLREDUCE BENCHMARK=================\n\n");
	//metric_t* metrics = (metric_t*)calloc(sizeof(metric_t), 128);
	//unified_all_reduce(metrics, A, B, result, nElements);
	//printf("RESULT\n");
	//inspect(result, nElements, 0);


	const int start = 32;
	const int iterations = 100;
	metric_t* metrics = (metric_t*)calloc(sizeof(metric_t), 128);

	// iterate and avg all
	for (int k = 0; k < iterations; k++) {
		int j = 0;
		for (size_t i = start; i <= nElements; i*=2) {
			unified_all_reduce(metrics+j, A, B, result, i); 
			j++;
		}
	}


	printf("\n==============AVG UNIFIED ALLREDUCE BENCHMARK===============\n\n");

	int j = 0;
	for (size_t i = start; i <= nElements; i*=2) {
		(metrics+j)->time /= iterations;
		(metrics+j)->bandwidth /= iterations;
		printf("\t[AVG Bandwidth (GB/s): %f] took %f", (metrics+j)->bandwidth, (metrics+j)->time);
		printf(" >> %ld bytes\n",i*sizeof(float));
		j++;
	}
	printf("\n");

	//inspect(result, nElements, 0);

	cudaFree(A);
	cudaFree(B);
	cudaFree(result);
	free(metrics);

	return 0;
};
