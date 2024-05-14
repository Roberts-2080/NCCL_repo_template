#!/bin/bash

# arguments for res50.py
# --batch: batch size,                          default=64
# --rate: learning rate,                        default=1e-4
# --mom: momentum,                              default=0.8
# --nodes: number of data loading workers,      default=1
# --gpus: number of gpus per node,              default=1
# --nr: ranking with nodes,                     default=0
# --epochs: number of epochs,                   default=2
# --cluster: super cluster of Strong Compute,   default=9
# --backend: communication backend,             default=nccl

# -f: force overwrite existing file
# -c: to profile the number of kernels provided in param
# -o: output to a file with name given in param
ncu -f --target-processes all -o profile python3 res50.py --gpus 2 --epochs 1
