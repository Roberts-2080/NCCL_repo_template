#include "common.h" 

void run(float* h_a, float* d_a, size_t bytes) {
 	float _time_;
	cudaEvent_t startEvent, stopEvent; 
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));

	checkCuda(cudaEventRecord(startEvent, 0));
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	checkCuda( cudaEventRecord(stopEvent, 0));
	checkCuda( cudaEventSynchronize(stopEvent));
	checkCuda( cudaEventElapsedTime(&_time_, startEvent, stopEvent)); 
	printf("\t\tHost to Device >> ");
	printf("[Bandwidth (GB/s): %f]\n", bytes * 1e-6 / _time_);

	checkCuda(cudaEventRecord(startEvent, 0));
	cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
	checkCuda( cudaEventRecord(stopEvent, 0));
	checkCuda( cudaEventSynchronize(stopEvent));
	checkCuda( cudaEventElapsedTime(&_time_, startEvent, stopEvent)); 
	printf("\t\tDevice to Host >> ");
	printf("[Bandwidth (GB/s): %f]\n", bytes * 1e-6 / _time_);

	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
}

int main()
{
	size_t N = 1024*1024*32;
	size_t bytes = N * sizeof(float);
	float *h_a = (float*)malloc(bytes);
	float *d_a;
	cudaMalloc((float**)&d_a, bytes);
	printf("\n====================CUDA MEMCPY BENCHMARK====================\n\n");

	memset(h_a, 420.69, bytes);

	for (size_t i = 4; i <= bytes; i*=2) {
		printf("\tcudaMemcpy on %ld bytes\n", i);
		run(h_a, d_a, i);
	}

	return 0;
}
