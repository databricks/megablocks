import torch


def repeat(x, tiling):
    if all([t == 1 for t in tiling]):
        return x
    return x.repeat(*tiling)
