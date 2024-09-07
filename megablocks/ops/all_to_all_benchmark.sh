#!/bin/bash

DISTRIBUTED_ARGUMENTS="\
--nproc_per_node 8 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000"

python -m torch.distributed.launch \
       ${DISTRIBUTED_ARGUMENTS} \
       megablocks/ops/all_to_all_benchmark.py
