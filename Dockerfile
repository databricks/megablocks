FROM nvcr.io/nvidia/pytorch:23.01-py3

RUN pip install stanford-stk

ENV PYTHONPATH="/mount/megablocks/third_party/Megatron-LM:${PYTHONPATH}"

WORKDIR /mount/megablocks