import unittest
from functools import partial

from absl.testing import parameterized
from megablocks import turbo_util as turbo
from megablocks.layers.arguments import Arguments
from megablocks.layers import dmoe
from megablocks.layers import moe
from megablocks.layers import testing
import torch
import copy
# from transformer_engine.quickstart_utils import cast_to_representable


def test_modules(
        hidden_size,
        ffn_hidden_size,
        moe_num_experts=1,
        moe_capacity_factor=1,
        grouped_mlp=True,
        moe_top_k=1,
        fp8=False):
    """ Test that grouped mlp is the same as fp8 and torch linear multiplication. """
    init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.1)
    dmoe_args = Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=moe_num_experts,
        moe_capacity_factor=moe_capacity_factor,
        moe_top_k=moe_top_k,
        init_method=init_method,
        memory_optimized_mlp=False,
        mlp_type='mlp',
        grouped_mlp=grouped_mlp,
        torch_mlp=False,
        fp16=False,
        bf16=True)
    torch_moe_args = copy.deepcopy(dmoe_args)
    torch_moe_args.grouped_mlp = True 
    torch_moe_args.torch_mlp = not fp8
    torch_moe_args.bf16 = True
    torch_moe_args.fp8 = fp8

    dmoe_mlp = dmoe.dMoE(dmoe_args)
    torch_moe_mlp = dmoe.dMoE(torch_moe_args)

    dmoe_mlp.cuda(torch.cuda.current_device()).to(torch.bfloat16)
    torch_moe_mlp.cuda(torch.cuda.current_device()).to(torch.bfloat16)

    # Set the baseline parameters to match exactly.
    with torch.no_grad():
        if not fp8:
            torch_moe_mlp.experts.mlp.w1.copy_(dmoe_mlp.experts.mlp.w1)
            torch_moe_mlp.experts.mlp.w2.copy_(dmoe_mlp.experts.mlp.w2)
        else:
            for i in range(moe_num_experts):
                assert torch_moe_mlp.experts.mlp.w1[i].weight_tensor.shape == dmoe_mlp.experts.mlp.w1.reshape(-1, ffn_hidden_size, hidden_size)[i, :, :].shape
                torch_moe_mlp.experts.mlp.w1[i].weight_tensor = dmoe_mlp.experts.mlp.w1.reshape(-1, ffn_hidden_size, hidden_size)[i, :, :]
                torch_moe_mlp.experts.mlp.w1[i].to(torch.bfloat16).cuda()
                assert torch_moe_mlp.experts.mlp.w2[i].weight_tensor.shape == dmoe_mlp.experts.mlp.w2.reshape(-1, hidden_size, ffn_hidden_size)[i, :, :].shape
                torch_moe_mlp.experts.mlp.w2[i].weight_tensor = dmoe_mlp.experts.mlp.w2.reshape(-1, ffn_hidden_size, hidden_size)[i, :, :]
                torch_moe_mlp.experts.mlp.w2[i].to(torch.bfloat16).cuda()
        torch_moe_mlp.router.layer.weight.copy_(dmoe_mlp.router.layer.weight)
    return dmoe_args, torch_moe_args, dmoe_mlp, torch_moe_mlp

# bz, sl, hs, ne, top_k
# min size: (1, 2, 128, 2, 1)
_FORWARD_TESTS = (
    (16, 1024, 512, 1, 1),
    (16, 1024, 512, 2, 1),
    (16, 1024, 512, 4, 1),
    (16, 1024, 512, 8, 1),
    (8, 2048, 512, 1, 1),
    (8, 2048, 512, 2, 1),
    (8, 2048, 512, 4, 1),
    (16, 1024, 512, 2, 2),
    (16, 1024, 512, 4, 2),
    (16, 1024, 512, 4, 4),
    (16, 1024, 512, 8, 2),
    (16, 1024, 512, 8, 4),
    (16, 1024, 512, 8, 8),
)


class torchMoETest(parameterized.TestCase):

    @staticmethod
    def tearDown():
        moe.clear_load_balancing_loss()

    @parameterized.parameters(*_FORWARD_TESTS)
    def testTorchMoE_ForwardBackward(self, bs, sl, hs, num_experts, top_k):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()
        x.requires_grad_(True)

        _, _, _, layer = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs*2,
            moe_num_experts=num_experts,
            moe_capacity_factor=1,
            grouped_mlp=True,
            moe_top_k=top_k,
            fp8=False,
        )
        out, _ = layer(x)
        self.assertSequenceEqual(out.shape, x.shape)
        out.sum().backward()
        self.assertTrue(x.grad is not None)

    @parameterized.parameters(*_FORWARD_TESTS)
    def testTorchMoE_ForwardBackward_VersusDMoE(self, bs, sl, hs, num_experts, top_k):
        torch.manual_seed(42)
        # create input for torch_moe
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()
        x.requires_grad_(True)

        # create input for dmoe
        y = x.clone().detach()
        y.requires_grad_(True)
        
        # get dmoe and torch moe
        _, _, dmoe_mlp, torch_moe_mlp = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs*2,
            moe_num_experts=num_experts,
            moe_capacity_factor=1,
            grouped_mlp=True,
            moe_top_k=top_k,
            fp8=False,
        )

        # perform torch moe forward backward
        torch_moe_out, _ = torch_moe_mlp(x)
        torch_moe_loss = torch_moe_out.sum()
        torch_moe_loss.backward()

        # perform dmoe forward backward
        dmoe_out, _ = dmoe_mlp(y)
        dmoe_loss = dmoe_out.sum()
        dmoe_loss.backward()

        # test forward pass shape 
        self.assertSequenceEqual(torch_moe_out.shape, x.shape)
        self.assertSequenceEqual(dmoe_out.shape, y.shape)
    
        # test forward pass output
        self.assertTrue(testing.allclose(torch_moe_out, dmoe_out))

        # test backward pass gradients
        self.assertTrue(testing.allclose(x.grad, y.grad))
    
    @parameterized.parameters(*_FORWARD_TESTS)
    def testTransformerEngineFp8MLP_ForwardBackward(self, bs, sl, hs, num_experts, top_k):
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()
        x.requires_grad_(True)

        _, _, _, layer = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs*2,
            moe_num_experts=num_experts,
            moe_capacity_factor=1,
            moe_top_k=top_k,
            fp8=True,
            grouped_mlp=False,
        )
        out, _ = layer(x)
        self.assertSequenceEqual(out.shape, x.shape)
        out.float().sum().backward()
        self.assertTrue(x.grad is not None)
    
    @parameterized.parameters(*_FORWARD_TESTS)
    def testTransformerEngineFp8MLP_VersusSparseMLPForwardBackward(self, bs, sl, hs, num_experts, top_k):
        torch.manual_seed(42)
        # create input for torch_moe
        x = torch.randn(sl, bs, hs).to(torch.bfloat16).cuda()
        x.requires_grad_(True)

        # create input for dmoe
        y = x.clone().detach()
        y.requires_grad_(True)
    
        _, _, dmoe_mlp, fp8_moe_mlp = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs*2,
            moe_num_experts=num_experts,
            moe_capacity_factor=1,
            moe_top_k=top_k,
            fp8=True,
            grouped_mlp=False,
        )
        # perform torch moe forward backward
        fp8_moe_out, _ = fp8_moe_mlp(x)
        fp8_moe_loss = fp8_moe_out.sum()
        fp8_moe_loss.backward()

        # perform dmoe forward backward
        dmoe_out, _ = dmoe_mlp(gity)
        dmoe_loss = dmoe_out.sum()
        dmoe_loss.backward()

        # test forward pass shape
        self.assertSequenceEqual(fp8_moe_out.shape, x.shape)
        self.assertSequenceEqual(dmoe_out.shape, y.shape)
    
        # test forward pass output
        # cast fp8_moe_out to bf16
        fp8_moe_out = fp8_moe_out.to(torch.bfloat16)
        self.assertTrue(testing.allclose(fp8_moe_out, dmoe_out))

        # test backward pass gradients
        self.assertTrue(testing.allclose(x.grad, y.grad))

if __name__ == '__main__':
    unittest.main()
