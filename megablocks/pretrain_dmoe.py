from megablocks.common import main
from megablocks.layers.megatron_layers import dMoE
import megatron


# HACK: Shim our dMoE layer into Megatron to replace the MLP block.
megatron.model.transformer.ParallelMLP = dMoE


if __name__ == "__main__":
    main()
