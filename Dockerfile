FROM nvcr.io/nvidia/pytorch:23.01-py3

RUN pip install stanford-stk>=0.0.4

# Install Infiniband stack

RUN wget https://content.mellanox.com/ofed/MLNX_OFED-5.4-3.6.8.1/MLNX_OFED_LINUX-5.4-3.6.8.1-ubuntu20.04-x86_64.tgz
RUN tar -xvzf MLNX_OFED_LINUX-5.4-3.6.8.1-ubuntu20.04-x86_64.tgz
RUN cd MLNX_OFED_LINUX-5.4-3.6.8.1-ubuntu20.04-x86_64/ && \
    ./mlnxofedinstall --force --without-mlnx-iproute2
# RUN /etc/init.d/openibd restart

ENV PYTHONPATH="/mount/megablocks/third_party/Megatron-LM:${PYTHONPATH}"

WORKDIR /mount/megablocks