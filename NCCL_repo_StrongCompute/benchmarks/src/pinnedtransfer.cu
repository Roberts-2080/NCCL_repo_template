#include "common.h"


void profileCopies(float        *h_a, 
                   float        *h_b, 
                   float        *d, 
                   unsigned int n,
                   const char   *desc) {

	  printf("\n%s transfers\n", desc);

	  unsigned int bytes = n * sizeof(float);

	  // events for timing
	  cudaEvent_t startEvent, stopEvent; 

	  checkCuda( cudaEventCreate(&startEvent) );
	  checkCuda( cudaEventCreate(&stopEvent) );

	  checkCuda( cudaEventRecord(startEvent, 0) );
	  checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
	  checkCuda( cudaEventRecord(stopEvent, 0) );
	  checkCuda( cudaEventSynchronize(stopEvent) );

	  float time;
	  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	  printf("\t\t\tHost to Device >> [Bandwidth (GB/s): %f]\n", bytes * 1e-6 / time);

	  checkCuda( cudaEventRecord(startEvent, 0) );
	  checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
	  checkCuda( cudaEventRecord(stopEvent, 0) );
	  checkCuda( cudaEventSynchronize(stopEvent) );

	  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	  printf("\t\t\tDevice to Host >> [Bandwidth (GB/s): %f]\n", bytes * 1e-6 / time);

	  for (int i = 0; i < n; ++i) {
	    if (h_a[i] != h_b[i]) {
	      printf("*** %s transfers failed ***\n", desc);
	      break;
	    }
	  }

	  checkCuda( cudaEventDestroy(startEvent) );
	  checkCuda( cudaEventDestroy(stopEvent) );
}

int main()
{
	size_t nElements = 1024*1024*32;
	size_t bytes = nElements * sizeof(float);

	// host arrays
	float *h_aPageable, *h_bPageable;   
	float *h_aPinned, *h_bPinned;

	float *d_a;

	h_aPageable = (float*)malloc(bytes);                    // host pageable
	h_bPageable = (float*)malloc(bytes);                    // host pageable
	checkCuda( cudaMallocHost((void**)&h_aPinned, bytes) ); // host pinned
	checkCuda( cudaMallocHost((void**)&h_bPinned, bytes) ); // host pinned
	checkCuda( cudaMalloc((void**)&d_a, bytes) );           // device

	for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;      
	memcpy(h_aPinned, h_aPageable, bytes);
	memset(h_bPageable, 0, bytes);
	memset(h_bPinned, 0, bytes);


	printf("\n=============PINNED/PAGED TRANSFER BENCHMARK=================\n\n");

	// perform copies and report bandwidth

	for (size_t i = 4; i <= nElements; i*=2) {
		printf("\tPinned/Paged transfer on %ld bytes",i*sizeof(float));
		profileCopies(h_aPageable, h_bPageable, d_a, i, "\t\tPageable");
		profileCopies(h_aPinned, h_bPinned, d_a, i, "\t\tPinned");
	}


	cudaFree(d_a);
	cudaFreeHost(h_aPinned);
	cudaFreeHost(h_bPinned);
	free(h_aPageable);
	free(h_bPageable);

	return 0;
};
