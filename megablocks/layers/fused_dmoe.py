from megablocks.layers import dmoe

class dMoE(dmoe.dMoE):

    def parallel_forward_once(self, x, expert_weights, top_experts):
        class ForwardBackward(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, expert_weights, top_experts):
                # Start all-to-all.
                # Compute parallel metadata
                # Compute topology
                # stop all-to-all
                # local permute
                # compute
                # return
                pass

            @staticmethod
            def backward(ctx, grad):
                # compute gradinet (no wgrad)
                # local un-permute
                # start all-to-all
                # wgrad
                # stop all-to-all
                pass
                
