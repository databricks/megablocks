import torch


def sum(x, dim=0):
    if x.shape[0] == 1:
        return x
    return x.sum(dim=dim)
