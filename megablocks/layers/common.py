from megablocks.layers.arguments import Arguments
import torch

def dtype(args : Arguments):
    dtype = torch.float32
    if args.fp16:
        dtype = torch.float16
    elif args.bf16:
        dtype = torch.bfloat16
    return dtype
