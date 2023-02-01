import megablocks
import megatron


# HACK: Shim our MoE layer into Megatron to replace the MLP block.
megatron.model.transformer.ParallelMLP = megablocks.layers.megatron_moe.MoE


if __name__ == "__main__":
    megablocks.common.main()
