# :robot: MegaBlocks

MegaBlocks is a system for efficient mixture-of-experts (MoE) training that builds on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). The core of the system is a "dropless-MoE" ([dMoE](https://github.com/tgale96/megablocks/blob/main/megablocks/layers/dmoe.py)) layer, which avoids dropping tokens by expressing MoE computation in terms of coarse-grained, block-sparse operations ([paper](https://arxiv.org/abs/2211.15841)). MegaBlocks supports data, expert and pipeline parallel training of MoEs

## Installation

We recommend using NGC's [`nvcr.io/nvidia/pytorch:21.12-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) PyTorch container. The [Dockerfile](https://github.com/tgale96/megablocks/blob/main/Dockerfile) builds on this image with additional dependencies. To build the image, run `docker build . -t megablocks-dev` and then `bash docker.sh` to launch the container.

Note that the block-sparse kernels used to implement dMoE are currently limited to A100 GPUs.

## Citation

```
@article{megablocks-arxiv,
  author    = {Trevor Gale and
               Deepak Narayanan and
               Cliff Young and
               Matei Zaharia},
  title     = {MegaBlocks: Efficient Sparse Training with Mixture-of-Experts},
  journal   = {CoRR},
  volume    = {abs/2211.15841},
  year      = {2022},
}
```
