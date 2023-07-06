FROM nvcr.io/nvidia/pytorch:23.01-py3

RUN pip install git+https://github.com/stanford-futuredata/stk.git@main

RUN pip install flash-attn

ENV PYTHONPATH="/mount/megablocks/third_party/Megatron-LM:${PYTHONPATH}"

WORKDIR /mount/megablocks