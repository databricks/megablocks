# :robot: MegaBlocks

MegaBlocks is a light-weight library for mixture-of-experts (MoE) training. The core of the system is efficient ([MoE](megablocks/layers/moe.py)) and "dropless-MoE" ([dMoE](megablocks/layers/dmoe.py), [paper](https://arxiv.org/abs/2211.15841)) layers.

MegaBlocks is currently integrated with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), where we support data, expert and pipeline parallel training of MoEs. We're working on extending more frameworks to support MegaBlocks.

# :rocket: Performance

![MegaBlocks Performance](media/dropping_end_to_end.png)

MegaBlocks dMoEs outperforms MoEs trained with [Tutel](https://github.com/microsoft/tutel) by up to **40%** compared to Tutel's best performing `capacity_factor` configurations. In addition to being faster, MegaBlocks simplifies MoE training by removing the `capacity_factor` hyperparameter alltogether. Compared to dense Transformers trained with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), MegaBlocks dMoEs can accelerate training by as much as **2.4x**. Check out our [paper](https://arxiv.org/abs/2211.15841) for more details!

```
Dataset: The Pile
Hardware: 8x A100-SXM4-80GB
Software: CUDA 11.5, Tutel v0.2, CUTLASS 2.5
```

# :building_construction: Installation

We recommend using NGC's [`nvcr.io/nvidia/pytorch:21.12-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) PyTorch container. The [Dockerfile](Dockerfile) builds on this image with additional dependencies. To build the image, run `docker build . -t megablocks-dev` and then `bash docker.sh` to launch the container.

Note that the block-sparse kernels used to implement dMoE are currently limited to A100 GPUs.

# :steam_locomotive: Usage

We provide scripts for pre-training Transformer MoE and dMoE language models under the [top-level directory](megablocks/). The quickest way to get started is to use one of the [experiment launch scripts](exp/).

# :writing_hand: Citation

```
@article{megablocks-arxiv,
  author    = {Trevor Gale and Deepak Narayanan and Cliff Young and Matei Zaharia},
  title     = {MegaBlocks: Efficient Sparse Training with Mixture-of-Experts},
  journal   = {CoRR},
  volume    = {abs/2211.15841},
  year      = {2022},
}
```
