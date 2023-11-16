from megablocks.layers.arguments import Arguments
import torch

def dtype(args : Arguments):
    if args.fp16:
        return torch.float16
    elif args.bf16:
        return torch.bfloat16
    return None


def cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor
