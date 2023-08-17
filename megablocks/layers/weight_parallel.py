from megablocks.layers import gelu
import stk
import torch


def _gather_weights(w, group, parallel_w=None, async_op=False):
    """Gather the weights across the process group.

    Args:
      w: torch.Tensor, local shard of the weights.
      group: ProcessGroup, the group to gather across.
      parallel_w: torch.Tensor, option output tensor to use
       for the gather.
      async_op: Whether to gather asynchronously.

    Returns:
      The gathered weights tensor and a handle for asynchronous
      communication.
    """
    n, k = w.shape
    world_size = torch.distributed.get_world_size(group)

    if parallel_w is None:
        parallel_w = torch.empty(
            n * world_size, k, device=w.device, dtype=w.dtype)
    handle = torch.distributed.all_gather_into_tensor(
        parallel_w, w, group=group, async_op=async_op)
    return parallel_w, handle


def _scaled_reduce_scatter(parallel_dw, group, dw=None, async_op=False):
    """Scatter reduce the weights across the process group.

    Args:
      parallel_dw: torch.Tensor, local shard of the weights.
      group: ProcessGroup, the group to scatter-reduce across.
      dw: torch.Tensor, option output tensor to use for the op.
      async_op: Whether to scatter reduce asynchronously.

    Returns:
      The reduced weights tensor, scaled by 1 / world_size, and
      a handle for asynchronous communication.
    """
    n, k = parallel_dw.shape
    world_size = torch.distributed.get_world_size(group)
    assert (n % world_size) == 0

    # Pre-scale the gradients by the world size.
    #
    # NOTE: Reduce in float32, always.
    parallel_dw = parallel_dw.float() / world_size

    if dw is None:
        dw = torch.empty(
            n // world_size, k,
            device=parallel_dw.device,
            dtype=torch.float32)
    handle = torch.distributed.reduce_scatter_tensor(
        dw, parallel_dw, group=group, async_op=async_op)
    return dw, handle


class WeightParallelSddNt(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, w, topo, group):
        # [m, k] x [n, k] = [m, n]
        if not x.is_contiguous() or not w.is_contiguous():
            raise ValueError("Expected contiguous 'x' and 'w'.")

        ctx.group = group
        ctx.shape = topo.shape
        ctx.save_for_backward(
            x, w,
            topo.row_indices,
            topo.column_indices,
            topo.offsets,
            topo.column_indices_t,
            topo.offsets_t,
            topo.block_offsets_t)

        # TODO(tgale): Support prefetching forward weights.
        parallel_w, _ = _gather_weights(w, group)
        return stk.ops.sdd(x, parallel_w.t(), topo).data

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        x, w = ctx.saved_tensors[:2]
        grad = stk.Matrix(ctx.shape, grad, *ctx.saved_tensors[2:])

        # Start the weight gather asynchronously to overlap with the
        # weight gradient computation.
        parallel_w, handle = _gather_weights(w, ctx.group, async_op=True)
        parallel_dw = None
        if ctx.needs_input_grad[1]:
            parallel_dw = stk.ops.dsd(grad.t(), x)

        # Start the weight gradient reduce scatter to overlap with the
        # data gradient computation.
        handle.wait()
        dw, handle = _scaled_reduce_scatter(parallel_dw, ctx.group, async_op=True)
        dx = None
        if ctx.needs_input_grad[0]:
            dx = stk.ops.dsd(grad, parallel_w)

        # NOTE: Be careful to wait and only cast dw to the output dtype once
        # we've blocked on the asynchronous NCCL operation.
        handle.wait()
        dw = dw.to(w.dtype)
        return dx, dw, None, None


def sdd_nt(a, b, topo, group):
    return stk.Matrix(
        topo.size(),
        WeightParallelSddNt.apply(a, b, topo, group),
        topo.row_indices,
        topo.column_indices,
        topo.offsets,
        topo.column_indices_t,
        topo.offsets_t,
        topo.block_offsets_t)


class WeightParallelDsdNn(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx,
                shape,
                data,
                row_indices,
                column_indices,
                offsets,
                column_indices_t,
                offsets_t,
                block_offsets_t,
                w,
                group):
        # [m, k] x [k, n] = [m, n]
        if not data.is_contiguous() or not w.is_contiguous():
            raise ValueError("Expected contiguous 'data' and 'w'.")

        ctx.group = group
        ctx.shape = shape
        ctx.save_for_backward(
            data,
            row_indices,
            column_indices,
            offsets,
            column_indices_t,
            offsets_t,
            block_offsets_t,
            w)
        x = stk.Matrix(
            shape,
            data,
            row_indices,
            column_indices,
            offsets,
            column_indices_t,
            offsets_t,
            block_offsets_t)

        # TODO(tgale): Support prefetching forward weights.
        parallel_w, _ = _gather_weights(w, group)
        return stk.ops.dsd(x, parallel_w)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        x = stk.Matrix(ctx.shape, *ctx.saved_tensors[:-1])
        w = ctx.saved_tensors[-1]

        # Start the weight gather asynchronously to overlap with the
        # weight gradient computation.
        parallel_w, handle = _gather_weights(w, ctx.group, async_op=True)
        parallel_dw = None
        if ctx.needs_input_grad[-2]:
            parallel_dw = stk.ops.dsd(x.t(), grad)

        # Start the weight gradient reduce scatter to overlap with the
        # data gradient computation.
        handle.wait()
        dw, handle = _scaled_reduce_scatter(parallel_dw, ctx.group, async_op=True)
        dx = None
        if ctx.needs_input_grad[1]:
            dx = stk.ops.sdd(grad, parallel_w.t(), x)

        # NOTE: Be careful to wait and only cast dw to the output dtype once
        # we've blocked on the asynchronous NCCL operation.
        handle.wait()
        dw = dw.to(w.dtype)
        return None, dx.data, None, None, None, None, None, None, dw, None


def dsd_nn(a, b, group):
    return WeightParallelDsdNn.apply(
        a.size(),
        a.data,
        a.row_indices,
        a.column_indices,
        a.offsets,
        a.column_indices_t,
        a.offsets_t,
        a.block_offsets_t,
        b,
        group)


class MemoryOptimizedWeightParallelMLP(torch.autograd.Function):
    """Sparse MLP with manually scheduled memory reuse."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, w1, w2, topo, group):
        # x: [m, k], w1: [n, k], w2: [n, k]
        if (not x.is_contiguous() or not w1.is_contiguous() or
            not w2.is_contiguous()):
            raise ValueError("Expected contiguous 'x', 'w1' and 'w2'.")

        # Layer 0: x @ w1.t().
        parallel_w1, _ = _gather_weights(w1, group)
        sdd_out = stk.ops.sdd(x, parallel_w1.t(), topo)

        # GeLU.
        gelu_out = gelu.gelu(sdd_out)

        # Layer 1: x @ w2.
        #
        # NOTE: Reuse the buffer for the w1 weight gather.
        parallel_w2, _ = _gather_weights(w2, group, parallel_w1)
        dsd_out = stk.ops.dsd(gelu_out, parallel_w2)

        # NOTE: Save the input to the layer and the gelu input for
        # gradient computation. We'll re-compute the gelu forward
        # pass in the backward pass to avoid materializing another
        # intermediate.
        ctx.group = group
        ctx.shape = topo.shape
        ctx.save_for_backward(
            x, w1, w2, sdd_out.data,
            topo.row_indices,
            topo.column_indices,
            topo.offsets,
            topo.column_indices_t,
            topo.offsets_t,
            topo.block_offsets_t)
        return dsd_out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, ddsd_out):
        x, w1, w2 = ctx.saved_tensors[:3]
        sdd_out = stk.Matrix(ctx.shape, *ctx.saved_tensors[3:])

        if (not ctx.needs_input_grad[0] or
            not ctx.needs_input_grad[1] or
            not ctx.needs_input_grad[2]):
            raise ValueError("Expected all MLP inputs to need grad.")

        # Start the weight gather asynchronously to overlap with the
        # weight gradient computation and gelu recompute.
        parallel_w2, handle = _gather_weights(
            w2, ctx.group, async_op=True)

        # Compute dw2 with recomputed gelu output.
        gelu_out = gelu.gelu(sdd_out)
        parallel_dw2 = stk.ops.dsd(gelu_out.t(), ddsd_out)

        # Start the weight gradient reduce scatter to overlap with the
        # data gradient computation.
        handle.wait()
        dw2, handle = _scaled_reduce_scatter(
            parallel_dw2, ctx.group, async_op=True)

        # Compute dgelu_out.
        #
        # NOTE: We reuse the gelu_out allocation.
        stk.backend.triton_kernels.sdd(
            ddsd_out, parallel_w2.t(),
            sdd_out.shape,
            gelu_out.data,
            sdd_out.offsets,
            sdd_out.row_indices,
            sdd_out.column_indices)
        dgelu_out = gelu_out

        # NOTE: Be careful to wait and only cast dw to the output dtype once
        # we've blocked on the asynchronous NCCL operation.
        handle.wait()
        dw2 = dw2.to(w2.dtype)

        # Start the weight gather asynchronously to overlap with the
        # weight and gelu gradient computation.
        #
        # NOTE: Reuse the buffer from the w2 weight gather.
        parallel_w1, handle = _gather_weights(
            w1, ctx.group, parallel_w2, async_op=True)

        # Compute dsdd_out.
        #
        # NOTE: This reuses the dgelu_out allocation.
        dsdd_out = gelu.gelu_backward_(dgelu_out, sdd_out)

        # Compute dw1.
        #
        # NOTE: This reuses the parallel_dw2 allocation.
        stk.backend.triton_kernels.dsd(
            dsdd_out.t().shape,
            dsdd_out.data,
            dsdd_out.offsets,
            dsdd_out.row_indices,
            dsdd_out.column_indices,
            dsdd_out.offsets_t,
            dsdd_out.column_indices_t,
            dsdd_out.block_offsets_t,
            True,  # transpose_a
            x,
            parallel_dw2)
        parallel_dw1 = parallel_dw2

        # Start the weight gradient reduce scatter to overlap with the
        # data gradient computation.
        handle.wait()
        dw1, handle = _scaled_reduce_scatter(
            parallel_dw1, ctx.group, async_op=True)

        # Compute dx.
        #
        # NOTE: This reuses the ddsd_out allocation.
        stk.backend.triton_kernels.dsd(
            dsdd_out.shape,
            dsdd_out.data,
            dsdd_out.offsets,
            dsdd_out.row_indices,
            dsdd_out.column_indices,
            dsdd_out.offsets_t,
            dsdd_out.column_indices_t,
            dsdd_out.block_offsets_t,
            False,
            parallel_w1,
            ddsd_out)
        dx = ddsd_out

        # NOTE: Be careful to wait and only cast dw to the output dtype once
        # we've blocked on the asynchronous NCCL operation.
        handle.wait()
        dw1 = dw1.to(w1.dtype)
        return dx, dw1, dw2, None, None

memory_optimized_weight_parallel_mlp = MemoryOptimizedWeightParallelMLP.apply
