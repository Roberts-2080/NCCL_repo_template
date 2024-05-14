# Benchmarking GPU Bandwidth / All Reduce (CUDA)
(Make sure NCCL & CUDA are installed)

# Compiling / Building 
Build tests with:
```
make
```
# Running benchmarks 
Run all tests with:
```
make run
```

- cudaMemcpy with: make cuda
- Pinned vs Paged CPU to GPU transfers with: make pinned
- NCCL allReduce() benchmark with: make reduce
- Naive allReduce() benchmark with: make naive
- Ring allReduce() on 2 GPUs benchmark with: make ring 
- Unified allReduce() on 2 GPUs benchmark with: make unified
