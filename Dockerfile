FROM nvcr.io/nvidia/pytorch:23.01-py3

RUN pip install stanford-stk>=0.0.4

ENV PYTHONPATH="/mount/megablocks/third_party/Megatron-LM:${PYTHONPATH}"

WORKDIR /mount/megablocks