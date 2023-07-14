import torch
import triton
import triton.language as tl


def assert_is_tensor(x, ndim):
    if x.ndim != ndim:
        raise ValueError(f"Expected {ndim}-tensor but got {x.ndim}-tensor")


def assert_is_matrix(x):
    assert_is_tensor(x, 2)


def assert_is_vector(x):
    if x.ndim != 1:
        raise ValueError(f"Expected 1-tensor but got {x.ndim}-tensor")


def assert_equal(a, b):
    if a != b:
        raise ValueError(f"Expected dimensions to be equal but got {a} and {b}.")


# a: (tokens, hidden_size), real.
# indices: (tokens * top_k), integer.
# bin_ids: (tokens * top_k), integer.
# weights: (tokens * top_k), real.
# bins: (num_experts,), integer.
# padded_bins: (num_experts,), integer.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 64}, num_warps=2),
        triton.Config({'BLOCK_X': 128}, num_warps=2),
        triton.Config({'BLOCK_X': 256}, num_warps=2),
        triton.Config({'BLOCK_X': 128}, num_warps=4),
        triton.Config({'BLOCK_X': 256}, num_warps=4),
    ],
    key=['num_columns'],
)
@triton.jit
def _padded_copy(
        a,
        b,
        num_columns,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        TOP_K : tl.constexpr,
        BLOCK_X : tl.constexpr,
        A_TO_B : tl.constexpr,
        SCALE : tl.constexpr):
    # Our index into array 'a'.
    index_a = tl.load(indices + tl.program_id(0))

    # One threadblock per row in 'a'. Array 'b' has greater or equal
    # number of rows since they could be padded.
    bin_idx = tl.load(bin_ids + tl.program_id(0))

    # Now we know what bin we're assigned to, but we need to know how
    # many threadblocks were assigned to earlier bins so we can offset
    # in our bin properly.
    offset_in_bin = tl.program_id(0);
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)

    # Load the starting index of our bin in array 'b'.
    index_b = offset_in_bin;
    if bin_idx > 0:
        index_b += tl.load(padded_bins + bin_idx - 1)

    # Offset the input and output pointers.
    #
    # NOTE: Divide the input index
    a += (index_a // TOP_K) * num_columns
    b += index_b * num_columns
    offsets = tl.arange(0, BLOCK_X)

    # Load the scale, if requested.
    scale = tl.load(weights + index_a) if SCALE else 1

    # Swap the pointers depending on the direction.
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a

    iterations = tl.cdiv(num_columns, BLOCK_X)
    for i in range(tl.cdiv(num_columns, BLOCK_X)):
        mask = offsets < num_columns
        x = tl.load(iptr + offsets, mask=mask)
        x *= scale

        # If top_k > 1 and we're writing from B => A we need
        # to use atomics to accumulate the result.
        if (TOP_K == 1) or A_TO_B:
            tl.store(optr + offsets, x, mask=mask)
        else:
            tl.atomic_add(optr + offsets, x, mask=mask)

        offsets += BLOCK_X


def padded_gather(x, indices, bin_ids, weights, bins, padded_bins, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_is_vector(padded_bins)
    assert_equal(indices.shape[0], x.shape[0] * top_k)
    assert_equal(bin_ids.shape[0], x.shape[0] * top_k)
    assert_equal(bins.size(), padded_bins.size())

    if weights is not None:
        assert_equal(weights.shape[0], x.shape[0] * top_k)

    # NOTE: Because of the padding, the output size is dynamic.
    # We load the final padded bin bound to get the output rows.
    output_rows = padded_bins[-1].cpu().item()
    out = torch.zeros(
        (output_rows, x.shape[1]),
        dtype=x.dtype,
        device=x.device)
    _padded_copy[(indices.shape[0],)](
        x,
        out,
        x.shape[1],
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        A_TO_B=True,
        TOP_K=top_k,
        SCALE=weights is not None)
    return out


def padded_scatter(x, indices, bin_ids, weights, bins, padded_bins, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_is_vector(padded_bins)
    assert_equal(indices.shape[0], bin_ids.shape[0])
    assert_equal(bins.size(), padded_bins.size())

    if weights is not None:
        assert_equal(indices.shape[0], weights.shape[0])

    tokens = indices.shape[0] // top_k
    out = torch.zeros(
        (tokens, x.shape[1]),
        dtype=x.dtype,
        device=x.device)
    _padded_copy[(indices.shape[0],)](
        out,
        x,
        x.shape[1],
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        A_TO_B=False,
        TOP_K=top_k,
        SCALE=weights is not None)
    return out

# a: (tokens, hidden_size), real.
# b: (num_experts, expert_capacity, num_columns), real.
# indices: (tokens,), integer.
# bins: (num_experts,), integer.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 64}, num_warps=2),
        triton.Config({'BLOCK_X': 128}, num_warps=2),
        triton.Config({'BLOCK_X': 256}, num_warps=2),
        triton.Config({'BLOCK_X': 128}, num_warps=4),
        triton.Config({'BLOCK_X': 256}, num_warps=4),
    ],
    key=['num_columns'],
)
@triton.jit
def _binned_copy(
        a,
        b,
        num_experts,
        expert_capacity,
        num_columns,
        indices,
        bins,
        BLOCK_X : tl.constexpr,
        A_TO_B : tl.constexpr):
    # Load our indices into the output.
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)

    # Calculate our offset into the output.
    index_b = expert_idx * expert_capacity + entry_idx

    # Load the index bounds for our bin and calculate
    # the number of tokens assigned to our expert.
    start = 0
    if expert_idx > 0:
       start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx)
    num_tokens = end - start

    # Calculate our offset into the input. If we don't
    # have an input exit early.
    if entry_idx >= num_tokens:
        return
    index_a = tl.load(indices + start + entry_idx)

    # Offset the input and output pointers.
    a += index_a * num_columns
    b += index_b * num_columns
    offsets = tl.arange(0, BLOCK_X)

    # Swap the pointers depending on the direction.
    #
    # NOTE: We need to zero the output in both directions.
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a

    iterations = tl.cdiv(num_columns, BLOCK_X)
    for i in range(tl.cdiv(num_columns, BLOCK_X)):
        mask = offsets < num_columns
        x = tl.load(iptr + offsets, mask=mask)
        tl.store(optr + offsets, x, mask=mask)
        offsets += BLOCK_X


def binned_gather(x, indices, bins, expert_capacity):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], x.shape[0])

    num_experts = bins.shape[0]
    out = torch.zeros(
        (num_experts, expert_capacity, x.shape[1]),
        dtype=x.dtype,
        device=x.device)
    _binned_copy[(num_experts, expert_capacity)](
        x,
        out,
        num_experts,
        expert_capacity,
        x.shape[1],
        indices,
        bins,
        A_TO_B=True)
    return out


def binned_scatter(x, indices, bins):
    # Validate the input shapes.
    assert_is_tensor(x, 3)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(bins.shape[0], x.shape[0])

    num_experts, expert_capacity, hidden_size = x.shape
    out = torch.zeros(
        (indices.shape[0], hidden_size),
        dtype=x.dtype,
        device=x.device)
    _binned_copy[(num_experts, expert_capacity)](
        out,
        x,
        num_experts,
        expert_capacity,
        hidden_size,
        indices,
        bins,
        A_TO_B=False)
    return out
