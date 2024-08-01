FROM nvcr.io/nvidia/pytorch:23.09-py3

RUN pip install stanford-stk==0.0.6

RUN pip install flash-attn

ENV PYTHONPATH="/mount/megablocks/third_party/Megatron-LM:${PYTHONPATH}"

WORKDIR /mount/megablocks
