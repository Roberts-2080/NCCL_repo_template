#!/bin/bash

# clean any existing builds/files
make clean
rm -f topo.xml
rm -f output.txt

# export environment variables for NCCL

export NCCL_TOPO_DUMP_FILE=topo.xml
export NCCL_GRAPH_DUMP_FILE=graph.xml

# values: VERSION/WARN/INFO/ABORT/TRACE
export NCCL_DEBUG=INFO

# values: INIT,COLL,P2P,SHM,NET,GRAPH,TUNING,ENV,ALLOC,CALL,ALL
export NCCL_DEBUG_SUBSYS=GRAPH,TUNING

# values: RING/TREE/COLLNET
export NCCL_ALGO=RING

# build
make build

# run
./build/main

echo "Success!"
