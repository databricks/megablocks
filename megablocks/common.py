from megablocks.layers import moe
import megatron
from megatron.model import ModelType
from megatron import mpu
from megatron.utils import average_losses_across_data_parallel_group
import pretrain_gpt


# HACK: Shim in our loss function to handle the MoE load balancing loss.
megatron_loss_fn = pretrain_gpt.loss_func
def loss_func(loss_mask, output_tensor=None):
    if mpu.is_pipeline_last_stage():
        assert output_tensor is not None
        loss, loss_dict = megatron_loss_fn(
            loss_mask, output_tensor)
        assert loss.numel() == 1
    else:
        loss = None
        loss_dict = {}

    # Get the load balancing loss contribution for each layer.
    args = megatron.get_args()
    if args.recompute_granularity is not None:
        # Ignore losses computed during forward pass if activation
        # recomputation is turned on.
        load_balancing_loss = moe.get_load_balancing_loss()
        if args.num_layers * 2 == len(load_balancing_loss):
            load_balancing_loss = load_balancing_loss[args.num_layers:]
            moe.clear_load_balancing_loss()
            moe.save_load_balancing_loss(load_balancing_loss)
    lbl = moe.batched_load_balancing_loss()
    moe.clear_load_balancing_loss()

    averaged_lbl = average_losses_across_data_parallel_group([lbl])
    loss_dict["load balancing loss"] = averaged_lbl[0]
    if loss is None:
        return lbl, loss_dict
    return loss + lbl, loss_dict
pretrain_gpt.loss_func = loss_func


# HACK: Shim in our MoE arguments.
_megatron_add_training_args = megatron.arguments._add_training_args
def _add_moe_args(parser):
    parser = _megatron_add_training_args(parser)
    group = parser.add_argument_group(title="moe")
    group.add_argument("--moe-num-experts", type=int, default=1)
    group.add_argument("--moe-capacity-factor", type=int, default=1)
    group.add_argument("--moe-top-k", type=int, default=1)    
    group.add_argument("--moe-loss-weight", type=float, default=0.1)
    group.add_argument("--moe-jitter-eps", type=float, default=None)
    group.add_argument("--moe-lbl-in-fp32", type=bool, default=False)
    return parser
megatron.arguments._add_training_args = _add_moe_args


def main():
    megatron.training.pretrain(
        pretrain_gpt.train_valid_test_datasets_provider,
        pretrain_gpt.model_provider, ModelType.encoder_or_decoder_with_lbl,
        pretrain_gpt.forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})    
