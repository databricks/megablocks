from nvcr.io/nvidia/pytorch:21.12-py3

ENV STK_REPO=https://github.com/tgale96/stk/blob/main
RUN pip install ${STK_REPO}/dist/21.12-py3/stk-0.0.1-cp38-cp38-linux_x86_64.whl?raw=true

ENV PYTHONPATH="/mount/megablocks/third_party/Megatron-LM:${PYTHONPATH}"

WORKDIR /mount/megablocks