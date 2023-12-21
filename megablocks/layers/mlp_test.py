import unittest
from functools import partial

from absl.testing import parameterized
from megablocks.layers.arguments import Arguments
from megablocks.layers import dmoe
from megablocks.layers import testing

import torch
import copy


def test_modules(
    hidden_size,
    ffn_hidden_size,
    moe_num_experts=1,
    moe_capacity_factor=1,
    mlp_impl='torch',
    moe_top_k=1,
):
    init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.1)
    grouped_dmoe_args = Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=moe_num_experts,
        moe_capacity_factor=moe_capacity_factor,
        moe_top_k=moe_top_k,
        init_method=init_method,
        memory_optimized_mlp=False,
        mlp_type='mlp',
        mlp_impl='grouped',
        moe_expert_model_parallelism=False,
        fp8_orig_dtype=torch.bfloat16,
    )
    test_dmoe_args = copy.deepcopy(grouped_dmoe_args)
    test_dmoe_args.mlp_impl = mlp_impl

    grouped_dmoe = dmoe.dMoE(grouped_dmoe_args)
    test_dmoe = dmoe.dMoE(test_dmoe_args)

    grouped_dmoe.cuda(torch.cuda.current_device()).to(torch.bfloat16)
    test_dmoe.cuda(torch.cuda.current_device()).to(torch.bfloat16)

    # Set the baseline parameters to match exactly.
    with torch.no_grad():
        # Copy router
        test_dmoe.router.layer.weight.copy_(grouped_dmoe.router.layer.weight)
        # Copy w1 and w2
        w1 = grouped_dmoe.experts.mlp.w1.reshape(moe_num_experts, -1, grouped_dmoe_args.hidden_size).transpose(1, 2) # ne, hs, fhs
        w2 = grouped_dmoe.experts.mlp.w2.reshape(moe_num_experts, -1, grouped_dmoe_args.hidden_size) # ne, fhs, hs
        if test_dmoe_args.mlp_impl == 'torch':
            test_dmoe.experts.mlp.w1.copy_(w1)
            test_dmoe.experts.mlp.w2.copy_(w2)
        else:
            # transformer engine transposes the weights so we need a tranpose (1, 0) here
            for i in range(moe_num_experts):
                test_dmoe.experts.mlp.w1[i].weight.data.copy_(w1[i, :, :].transpose(1, 0))
                test_dmoe.experts.mlp.w2[i].weight.data.copy_(w2[i, :, :].transpose(1, 0))
    return grouped_dmoe, test_dmoe


# bz, sl, hs, ne, top_k
_TORCH_TESTS = (
    (1, 16, 128, 2, 1),
    (4, 2048, 8192, 32, 4),
)

# bz, sl, hs, ne, top_k, autocast, threshold
# Threshold is large when transformer engine casts weights to fp8 resulting in numerical differences.
# Thresholds is small when autocast is not enabled and both moe's have weights in fp16. 
_TRANSFORMER_ENGINE_TESTS = (
    (1, 16, 128, 2, 1, True, 75),
    (1, 16, 128, 2, 1, False, 0.5),
    (2, 2048, 8192, 16, 4, True, 75),
    (2, 2048, 8192, 16, 4, False, 0.5),
)

class dMoeMLPTest(parameterized.TestCase):
    """ Test that grouped dmoe is approximately the same as te/torch dmoe """

    @parameterized.parameters(*_TORCH_TESTS)
    def testTorchMoE_ForwardBackward_VersusDMoE(self, bs, sl, hs, num_experts, top_k):
        torch.manual_seed(42)
        # create input for torch_moe
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()
        x.requires_grad_(True)
        # copy input for grouped_moe
        y = x.clone().detach()
        y.requires_grad_(True)
        # get grouped and torch moe
        grouped_moe, torch_moe = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs*2,
            moe_num_experts=num_experts,
            moe_capacity_factor=1,
            moe_top_k=top_k,
        )

        # perform torch moe forward backward
        torch_moe_out, _ = torch_moe(x)
        torch_moe_loss = torch_moe_out.sum()
        torch_moe_loss.backward()
        # perform dmoe forward backward
        grouped_moe_out, _ = grouped_moe(y)
        grouped_moe_loss = grouped_moe_out.sum()
        grouped_moe_loss.backward()

        # test forward pass outputs are approximately the same
        self.assertTrue(testing.allclose(torch_moe_out, grouped_moe_out))
        # test backward pass gradients are approximately the same
        self.assertTrue(testing.allclose(x.grad, y.grad))
    
    @parameterized.parameters(*_TRANSFORMER_ENGINE_TESTS)
    def testTransformerEngine_ForwardBackward_VersusDMoE(self, bs, sl, hs, num_experts, top_k, autocast, threshold):
        torch.manual_seed(42)
        # create input for te_moe
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()
        x.requires_grad_(True)
        # create input for grouped_moe
        y = x.clone().detach()
        y.requires_grad_(True)
        
        # get grouped and te moe
        grouped_moe, te_moe = test_modules(
            mlp_impl='te',
            hidden_size=hs,
            ffn_hidden_size=hs*2,
            moe_num_experts=num_experts,
            moe_capacity_factor=1,
            moe_top_k=top_k,
        )
        # perform torch moe forward backward
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Format, DelayedScaling
        fp8_format = Format.E4M3  # E4M3 during forward pass, E5M2 during backward pass
        fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
        if autocast:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                te_moe_out, _ = te_moe(x)
        else:
            te_moe_out, _ = te_moe(x)
        te_moe_loss = te_moe_out.sum()
        te_moe_loss.backward()
        # perform dmoe forward backward
        grouped_moe_out, _ = grouped_moe(y)
        grouped_moe_loss = grouped_moe_out.sum()
        grouped_moe_loss.backward()
    
        # test forward pass output
        self.assertTrue(testing.allclose(te_moe_out, grouped_moe_out, threshold)) 
        # test backward pass gradients
        self.assertTrue(testing.allclose(x.grad, y.grad, threshold))
    
if __name__ == '__main__':
    unittest.main()
