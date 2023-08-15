#!/bin/bash

DISTRIBUTED_ARGUMENTS="\
--nproc_per_node 1 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000"

python -m torch.distributed.launch \
       ${DISTRIBUTED_ARGUMENTS} \
       megablocks/layers/memory_test.py
