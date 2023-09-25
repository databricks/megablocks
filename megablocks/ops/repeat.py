import torch


def repeat(x, tiling):
    if all([t == 1 in tiling]):
        return x
    return x.repeat(*tiling)
