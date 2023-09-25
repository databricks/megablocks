import torch


def sum(x, dim=0):
    if x.shape[dim] == 1:
        return x.squeeze(dim=dim)
    return x.sum(dim=dim)
