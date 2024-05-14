#!/bin/bash/env

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

nsys profile -o profile python3 res50.py --gpus 2 --epochs 1
