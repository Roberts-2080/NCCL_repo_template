#include "common.h"

__global__ void sum(float* A, float *B) {
	// parallel cuda sum
	int i = threadIdx.x;
	A[i] += B[i];
}

void GPU0toGPU1(float* d0, float *d1, float* h,
   	      size_t bytes, size_t nElements) 
{
	// transfer from dev0 to dev1 of n bytes
	cudaSetDevice(0);
	checkCuda( cudaMemcpy(h, d0, bytes, cudaMemcpyDeviceToHost) );
	cudaSetDevice(1);
	checkCuda( cudaMemcpy(d1, h, bytes, cudaMemcpyHostToDevice) );
	cudaSetDevice(0);

}

void GPU1toGPU0(float* d0, float *d1, float* h,
   	      size_t bytes, size_t nElements) 
{
	// transfer from dev1 to dev0 of n bytes
	cudaSetDevice(1);
	checkCuda( cudaMemcpy(h, d0, bytes, cudaMemcpyDeviceToHost) );
	cudaSetDevice(0);
	checkCuda( cudaMemcpy(d1, h, bytes, cudaMemcpyHostToDevice) );

}


void inspect(float* d, float* h, size_t nElements, int dev) {
	// print floats of dev r of n elements
	printf("%ld elements on dev_%d\n\t", nElements, dev);
	cudaSetDevice(dev);
	for (int i = 0; i < nElements; ++i)  {
		if (h[i] < 0) printf("%f\t",h[i]); 
		else printf(" %f\t",h[i]);
		if (i % 8 == 7) printf("\n\t");
	}
	
	printf("\n");
	cudaSetDevice(0);
}

void naive_allreduce(metric_t* m, float* A, float* B, float* A_temp, float* B_temp, float* host, size_t nElements) {

	// reduce all naive

	float _time_;
	cudaEvent_t startEvent, stopEvent; 

	checkCuda( cudaEventCreate(&startEvent) );
	checkCuda( cudaEventCreate(&stopEvent) );
	checkCuda( cudaEventRecord(startEvent, 0) );

	// load array B from GPU1 to A_temp on GPU0 (GPU0 -> GPU1)
	GPU1toGPU0(B, A_temp, host, nElements*sizeof(float), nElements);
	// kernal parallel sum 
	sum<<<1, nElements>>>(A, A_temp);
	// load back summed values of B to A (GPU1 -> GPU0)
	GPU0toGPU1(A, B, host, nElements*sizeof(float), nElements);

	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&_time_, startEvent, stopEvent) );


	checkCuda( cudaEventDestroy(startEvent) );
	checkCuda( cudaEventDestroy(stopEvent) );
	m->time += _time_;
	m->bandwidth += nElements * sizeof(float)*1e-6/_time_;
}

int main()
{
	size_t nElements = 1025 * 1024 * 32;
	size_t bytes = nElements * sizeof(float);

	// host arrays
	float *h;

	float *dev0_a; // ARRAY A on GPU 0
	float *dev0_b; // temp array on GPU 0

	float *dev1_a; // ARRAY B on GPU 1
	float *dev1_b; // temp array on GPU 1

	checkCuda( cudaMallocHost((void**)&h, bytes) ); // host pinned

	// alloc 2 arrays A and B on both GPUs
	checkCuda( cudaMalloc((void**)&dev0_a, bytes) );           // device
	checkCuda( cudaMalloc((void**)&dev0_b, bytes) );           // device
	cudaSetDevice(1);
	checkCuda( cudaMalloc((void**)&dev1_a, bytes) );           // device
	checkCuda( cudaMalloc((void**)&dev1_b, bytes) );           // device
	cudaSetDevice(0);

	//printf("\n==================NAIVE REDUCE BENCHMARK=====================\n\n");

	//Set values and load into both GPUs 
	//Dont know why but setting values seg faults doesnt matter any way
	//for (int i = 1; i < nElements + 1; ++i) h[i] = i; 
	//cudaMemcpy(dev0_a, h, bytes, cudaMemcpyDeviceToHost);
	//inspect(dev0_a, h, nElements, 0);

	//for (int i = 1; i < nElements + 1; ++i) h[i] = i; 
	//cudaMemcpy(dev1_a, h, bytes, cudaMemcpyDeviceToHost);
	//inspect(dev1_a, h, nElements, 1);	
	//return;

	metric_t* metrics = (metric_t*)calloc(sizeof(metric_t), 128);
	const int iterations = 100;

	for (int k = 0; k < iterations; k++) {
		int j = 0;
		for (size_t i = 32; i <= nElements; i*=2) {
			naive_allreduce(metrics+j, dev0_a, dev1_a, dev0_b, dev1_b, h, i); 
			j++;
		}
	}


	printf("\n===============AVG NAIVE ALLREDUCE BENCHMARK=================\n\n");

	int j = 0;
	for (size_t i = 32; i <= nElements; i*=2) {
		(metrics+j)->time /= iterations;
		(metrics+j)->bandwidth /= iterations;
		printf("\t[AVG Bandwidth (GB/s): %f] took %f", (metrics+j)->bandwidth, (metrics+j)->time);
		printf(" >> %ld bytes\n",i*sizeof(float));
		j++;
	}

	cudaFree(dev0_a);
	cudaFree(dev0_b);
	cudaFree(dev1_a);
	cudaFree(dev1_b);
	cudaFreeHost(h);
	free(metrics);

	return 0;
};
