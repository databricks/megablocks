import megablocks
import megatron


# HACK: Shim our dMoE layer into Megatron to replace the MLP block.
megatron.model.transformer.ParallelMLP = megablocks.layers.dmoe.dMoE


if __name__ == "__main__":
    megablocks.common.main()
