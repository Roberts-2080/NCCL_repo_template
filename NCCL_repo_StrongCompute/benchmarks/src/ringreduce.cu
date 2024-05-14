#include "common.h"

#define MAX_THREADS 2

// ring all reduce on two GPUs

typedef struct thread_t {
	int id;
	int dev;
	int step;
	float* host;	
	float* A;	
	float* B;	
	float* A_temp;	
	float* B_temp;	
	size_t nElements;	
	cudaStream_t *stream;
} thread_t;

thread_t* init_thread_t(int id,
		    int dev,
		    cudaStream_t *stream,
		    float* host, 
		    float* A, 
	 	    float* B,
		    float* A_temp,
                    float* B_temp,
		    size_t nElements,
		    int step) 
{
	thread_t* t = (thread_t*)malloc(sizeof(thread_t));
	t->id = id;
	t->dev = dev;
	t->host = host;
	t->A = A;
	t->B = B;
	t->A_temp = A_temp;
	t->B_temp = B_temp;
	t->nElements = nElements;
	t->step = step;
	t->stream = stream;
	return t;
}

__global__ void kernel(float* A, float *B) {
	int i = threadIdx.x;
	A[i] += B[i];
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

void* __allReduce__(void* args) 
{
	thread_t* t = (thread_t*)args;
	float* H = t->host;
	float* A = t->A;
	float* B = t->B;
	float* A_temp = t->A_temp;
	float* B_temp = t->B_temp;
	int step = t->step;
	int dev = t->dev;
	cudaStream_t stream = *t->stream;
	cudaSetDevice(dev);
	
	size_t bytes = step * sizeof(float);

	if (t->dev) 
	{
		// on GPU1
		cudaMemcpy(H, B, bytes, cudaMemcpyDeviceToHost);
		cudaSetDevice(0);
		// on GPU0
		cudaMemcpy(A_temp, H, bytes, cudaMemcpyHostToDevice);
		kernel<<<1, step, 0, stream>>>(A, A_temp);

		cudaMemcpy(H, A, bytes, cudaMemcpyDeviceToHost);
		cudaSetDevice(1);
		// on GPU1
		cudaMemcpy(B, H, bytes, cudaMemcpyHostToDevice);
	}
	else 
	{
		// on GPU0
		cudaMemcpy(H, A, bytes, cudaMemcpyDeviceToHost);
		cudaSetDevice(1);
		// on GPU1
		cudaMemcpy(B_temp, H, bytes, cudaMemcpyHostToDevice);
		kernel<<<1, step, 0, stream>>>(B, B_temp);

		cudaMemcpy(H, B, bytes, cudaMemcpyDeviceToHost);
		cudaSetDevice(0);
		// on GPU0
		cudaMemcpy(A, H, bytes, cudaMemcpyHostToDevice);
	}
	pthread_exit(args);
}


void ring_all_reduce(metric_t *m, float* host, float* A, float* B, float* A_temp, float* B_temp, int nGpus, size_t nElements) 
{
	pthread_t id[MAX_THREADS];
	thread_t* threads[MAX_THREADS];
	void* returned[MAX_THREADS];
	int step = nElements / MAX_THREADS;

	cudaStream_t streams[MAX_THREADS];


	// init threads
	for (int i = 0; i < MAX_THREADS; i++) {
		int dev = 0;
		int off = step * i;
		if (i % 2 == 0) dev = 1;
		cudaStreamCreate(&streams[i]);
		threads[i] = init_thread_t(i, 
					   dev, 
					   &streams[i],
					   host+off, 
					   A+off, 
					   B+off, 
					   A_temp+off, 
					   B_temp+off, 
					   nElements,
					   step); 
	}

	float _time_;
	cudaEvent_t startEvent, stopEvent; 

	checkCuda( cudaEventCreate(&startEvent) );
	checkCuda( cudaEventCreate(&stopEvent) );
	checkCuda( cudaEventRecord(startEvent, 0) );

	// main all reduce call
	for (int i = 0; i < MAX_THREADS; i++) {
		pthread_create(&id[i], NULL, __allReduce__, threads[i]);
	}
	// join all
	for (int i = 0; i < MAX_THREADS; i++) {
		pthread_join(id[i], (void**)&returned);
	}

	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&_time_, startEvent, stopEvent) );

	checkCuda( cudaEventDestroy(startEvent) );
	checkCuda( cudaEventDestroy(stopEvent) );

	for (int i = 0; i < MAX_THREADS; i++)
		free(threads[i]);

	m->time += _time_;
	m->bandwidth += nElements * sizeof(float)*1e-6/_time_;
}

int main()
{
	size_t nElements = 1024*1024*48;
	size_t bytes = nElements * sizeof(float);
	const int nGpus = 2;

	// host arrays
	float *h;

	float *A; // ARRAY A on GPU 0
	float *A_temp; // temp array on GPU 0

	float *B; // ARRAY B on GPU 1
	float *B_temp; // temp array on GPU 1


	checkCuda( cudaMallocHost((void**)&h, bytes) ); // host pinned

	// alloc 2 arrays A and B on both GPUs
	checkCuda( cudaMalloc((void**)&A, bytes) );           // device
	checkCuda( cudaMalloc((void**)&A_temp, bytes) );           // device
	cudaSetDevice(1);
	checkCuda( cudaMalloc((void**)&B, bytes) );           // device
	checkCuda( cudaMalloc((void**)&B_temp, bytes) );           // device
	cudaSetDevice(0);

	//printf("\n==================RING REDUCE BENCHMARK=====================\n\n");

	//printf("\nSETTING VALUES AS\n");
	//for (int i = 0; i < nElements; ++i) h[i] = i; 
	//cudaMemcpy(A, h, bytes, cudaMemcpyDeviceToHost);
	//inspect(A, h, nElements, 0);

	//for (int i = 0; i < nElements; ++i) h[i] = i; 
	//cudaMemcpy(B, h, bytes, cudaMemcpyDeviceToHost);
	//inspect(B, h, nElements, 1);	

	//printf("\nALL REDUCING\n");
	//ring_all_reduce(h, A, B, A_temp, B_temp, nGpus, nElements);
	//printf("\nDONE\n");
	//printf("NEW VALUES\n");
	//inspect(A, h, nElements, 0);	
	//inspect(B, h, nElements, 1);	


	metric_t* metrics = (metric_t*)calloc(sizeof(metric_t), 128);
	const int iterations = 100;

	for (int k = 0; k < iterations; k++) {
		int j = 0;
		for (size_t i = 32; i <= nElements; i*=2) {
			ring_all_reduce(metrics+j, h, A, B, A_temp, B_temp, nGpus, i); 
			j++;
		}
	}


	printf("\n===============AVG RING ALLREDUCE BENCHMARK=================\n\n");

	int j = 0;
	for (size_t i = 32; i <= nElements; i*=2) {
		(metrics+j)->time /= iterations;
		(metrics+j)->bandwidth /= iterations;
		printf("\t[AVG Bandwidth (GB/s): %f] took %f", (metrics+j)->bandwidth, (metrics+j)->time);
		printf(" >> %ld bytes\n",i*sizeof(float));
		j++;
	}

	cudaFree(A);
	cudaFree(B);
	cudaFree(A_temp);
	cudaFree(B_temp);
	cudaFreeHost(h);
	free(metrics);

	return 0;
};
