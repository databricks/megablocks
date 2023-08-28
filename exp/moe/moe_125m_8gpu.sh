#!/bin/bash

EXP_DIR="./checkpoint"

# 512 * 1k * 400k = 200b tokens.
# 512 * 1k * 200k = 100b tokens.
# 512 * 1k * 100k = 50b tokens (default).
# 512 * 1k * 20k = 10b tokens.
TRAINING_STEPS=20000
if [ -n "${2}" ]; then
    TRAINING_STEPS=$2;
fi

NUM_EXPERTS=64
if [ -n "${3}" ]; then
    NUM_EXPERTS=$3;
fi

CAPACITY_FACTOR=1
if [ -n "${4}" ]; then
    CAPACITY_FACTOR=$4;
fi

TOP_K=1
if [ -n "${5}" ]; then
    TOP_K=$5;
fi

LOSS_WEIGHT=0.1
if [ -n "${6}" ]; then
    LOSS_WEIGHT=$6;
fi

BATCH_SIZE=16
if [ -n "${7}" ]; then
    BATCH_SIZE=$7;
fi

##
### Pre-training for MoE 125M parameter.
##

# MoE hyperparameters.
MOE_ARGUMENTS="\
--moe-num-experts=${NUM_EXPERTS} \
--moe-capacity-factor=${CAPACITY_FACTOR} \
--moe-loss-weight=${LOSS_WEIGHT} \
--moe-top-k=${TOP_K}"

if [ -z "$MLP_WORKER_GPU" ]; then
# Distributed hyperparameters.
DISTRIBUTED_ARGUMENTS="\
--nproc_per_node 2 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000"
export GLOBLE_BATCH_SIZE=$((2 * $BATCH_SIZE))
else
echo "MLP_WORKER_GPU=$MLP_WORKER_GPU"
echo "MLP_WORKER_NUM=$MLP_WORKER_NUM"
echo "MLP_ROLE_INDEX=$MLP_ROLE_INDEX"
echo "MLP_WORKER_0_HOST=$MLP_WORKER_0_HOST"
echo "MLP_WORKER_0_PORT=$MLP_WORKER_0_PORT"
echo "RUN_ROOT=$RUN_ROOT"

DISTRIBUTED_ARGUMENTS="
    --nproc_per_node $MLP_WORKER_GPU \
    --nnodes $MLP_WORKER_NUM \
    --node_rank $MLP_ROLE_INDEX \
    --master_addr $MLP_WORKER_0_HOST \
    --master_port $MLP_WORKER_0_PORT
"
export GLOBLE_BATCH_SIZE=$(($BATCH_SIZE*$MLP_WORKER_GPU*$MLP_WORKER_NUM))
fi

# Model hyperparameters.
MODEL_ARGUMENTS="\
--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--seq-length 1024 \
--max-position-embeddings 1024"

# Training hyperparameters.
TRAINING_ARGUMENTS="\
--micro-batch-size ${BATCH_SIZE} \
--global-batch-size ${GLOBLE_BATCH_SIZE} \
--train-iters ${TRAINING_STEPS} \
--lr-decay-iters ${TRAINING_STEPS} \
--lr 0.00015 \
--min-lr 0.00001 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.01 \
--clip-grad 1.0 \
--init-method-std 0.01"

TOK=LLAMATokenizer
DATA_PATH=/share_nfs/process_data/pile_tokenized/chinese_llama/megatron/merged/tokenized
VOCAB_FILE=/share_nfs/process_data/pile_tokenized/chinese_llama/chinese_llama_tokenizer.model
# NOTE: We don't train for enough tokens for the
# split to matter.

DATA_ARGUMENTS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --data-impl mmap \
    --num-workers 4 \
    --split 969,30,1 \
    --tokenizer-type $TOK 
"

COMPUTE_ARGUMENTS="\
--fp16 \
--DDP-impl local \
--moe-expert-model-parallelism \
--tensor-model-parallel-size 2 \
--no-async-tensor-model-parallel-allreduce"

CHECKPOINT_ARGUMENTS="\
--save-interval 2000 \
--save ./${EXP_DIR}"

EVALUATION_ARGUMENTS="\
--eval-iters 100 \
--log-interval 100 \
--eval-interval 1000"

torchrun ${DISTRIBUTED_ARGUMENTS} \
       third_party/Megatron-LM/pretrain_gpt.py \
       ${MOE_ARGUMENTS} \
       ${MODEL_ARGUMENTS} \
       ${TRAINING_ARGUMENTS} \
       ${DATA_ARGUMENTS} \
       ${COMPUTE_ARGUMENTS} \
       ${CHECKPOINT_ARGUMENTS} \
       ${EVALUATION_ARGUMENTS} |& tee ./${EXP_DIR}/train.log
