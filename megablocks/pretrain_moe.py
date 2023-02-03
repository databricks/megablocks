from megablocks.common import main
from megablocks.layers.megatron_layers import MoE
import megatron


# HACK: Shim our MoE layer into Megatron to replace the MLP block.
megatron.model.transformer.ParallelMLP = MoE


if __name__ == "__main__":
    main()
