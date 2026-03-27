import torch

def assert_is_tensor(x, ndim):
    if x.ndim != ndim:
        raise ValueError(f'Expected {ndim}-tensor but got {x.ndim}-tensor')

def assert_is_matrix(x):
    assert_is_tensor(x, 2)

def assert_is_vector(x):
    if x.ndim != 1:
        raise ValueError(f'Expected 1-tensor but got {x.ndim}-tensor')

def assert_equal(a, b):
    if a != b:
        raise ValueError(f'Expected dimensions to be equal but got {a} and {b}.',)

# -------------------------------------------------------------------------
# Padded Operations
# -------------------------------------------------------------------------

def padded_gather(x, indices, bin_ids, weights, bins, padded_bins, top_k):
    """
    Gathers tokens from 'x' based on 'indices' and organizes them into
    a padded layout defined by 'padded_bins'.
    """
    # Validate inputs
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_is_vector(padded_bins)
    assert_equal(indices.shape[0], x.shape[0] * top_k)
    
    output_rows = padded_bins[-1].item()
    out = torch.zeros((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)
    
    # We iterate over experts (bins). 
    # Since num_experts is usually small (8-64), this loop is negligible compared to data movement.
    # 'bins' acts as a CSR pointer array.
    
    num_experts = len(bins)
    current_idx_start = 0
    current_padded_start = 0
    
    for i in range(num_experts):
        # Determine the range of indices for this expert
        bin_end = bins[i].item()
        padded_end = padded_bins[i].item()
        
        count = bin_end - current_idx_start
        
        if count > 0:
            # 1. Get the source token indices for this expert
            # indices are sorted by expert in standard MoE usage
            src_indices = indices[current_idx_start:bin_end]
            
            # 2. Gather data: x[src_indices]
            gathered = x[src_indices]
            
            # 3. Apply weights if present
            if weights is not None:
                w = weights[current_idx_start:bin_end].unsqueeze(1)
                gathered = gathered * w
            
            # 4. Place into the padded output buffer
            # We copy 'count' rows into the allocated slot for this expert
            out[current_padded_start : current_padded_start + count] = gathered
            
        current_idx_start = bin_end
        current_padded_start = padded_end
        
    return out


def gather(x, indices, bin_ids, weights, bins, top_k):
    """
    Standard gather without padding gaps. 
    Equivalent to padded_gather where bins == padded_bins.
    """
    # Optimization: If no padding logic is needed, we can do a bulk gather
    # provided we just want the data in the order of 'indices'.
    
    # Validate inputs
    assert_is_matrix(x)
    assert_is_vector(indices)
    
    # Bulk operation
    out = x[indices]
    
    if weights is not None:
        out = out * weights.unsqueeze(1)
        
    return out


def padded_scatter(x, indices, bin_ids, weights, bins, padded_bins, top_k):
    """
    Scatters data from the padded expert output 'x' back to original token locations.
    Accumulates results if top_k > 1.
    """
    # Validate inputs
    assert_is_matrix(x)
    assert_is_vector(indices)
    
    tokens = indices.shape[0] // top_k
    hidden_size = x.shape[1]
    
    # We scatter into a flattened view first: (tokens * top_k, hidden)
    # Then we will reduce.
    # Note: Using index_add_ on a zero tensor handles the accumulation logic.
    scatter_buffer = torch.zeros((tokens * top_k, hidden_size), dtype=x.dtype, device=x.device)
    
    num_experts = len(bins)
    current_idx_start = 0
    current_padded_start = 0
    
    # Generate a range tensor to map the slice logic to absolute positions in scatter_buffer
    # This avoids creating a massive index tensor for the whole batch.
    
    for i in range(num_experts):
        bin_end = bins[i].item()
        padded_end = padded_bins[i].item()
        count = bin_end - current_idx_start
        
        if count > 0:
            # Data coming from the expert (removing padding implicitly by slicing)
            expert_out = x[current_padded_start : current_padded_start + count]
            
            # Scale by weights (if we are scattering A_TO_B=False, weights are applied here)
            if weights is not None:
                w = weights[current_idx_start:bin_end].unsqueeze(1)
                expert_out = expert_out * w
            
            # Target positions in the flattened (tokens*topk) array
            # We are writing to the range [current_idx_start : bin_end]
            # strictly speaking, in standard MoE, 'indices' maps specific slots.
            # However, padded_scatter logic in Triton implies reversing the mapping.
            # In Megablocks, 'indices' defines the token index for the sorted buffer.
            # So we map: scatter_buffer[range] = expert_out
            
            # Wait, standard scatter logic: out[indices[i]] += x[i]
            # Here 'indices' contains the original token row IDs.
            target_rows = indices[current_idx_start:bin_end]
            
            # Add to output (Atomic add equivalent)
            scatter_buffer.index_add_(0, target_rows, expert_out) # Maps to (tokens, hidden) effectively?
            
            # Logic Correction:
            # The Triton kernel does: `index_a = indices[idx]`. `out[index_a] = x[...]`.
            # But the output 'out' in Triton is shape (tokens, top_k, hidden).
            # The python wrapper calculates `out.sum(dim=1)` at the end.
            # To match the "out" shape of (tokens, top_k, hidden), we need to know
            # which "k" slot a specific token-expert pair occupies.
            # However, standard PyTorch implementation simplifies this:
            # We can scatter directly to (tokens, hidden) via index_add_ if we don't strictly need the intermediate (tokens, top_k, hidden).
            
    # Re-reading the Triton wrapper: 
    # It constructs `out` as (tokens, top_k, hidden).
    # Then `out.sum(dim=1)` or `view`.
    # Since `indices` usually points to the Token ID (0..N), it doesn't encode the "top_k slot".
    # Standard Megablocks usage: indices is (tokens * top_k).
    # If we want exact parity with the Intermediate Tensor shape:
    
    # Optimized PyTorch Logic:
    # 1. We essentially want to perform: out_tensor[indices] += expert_outputs
    # 2. But `indices` has repeats (a token appears top_k times).
    # 3. Direct scatter_add/index_add to (tokens, hidden) is mathematically equivalent to (tokens, topk, hidden).sum(1).
    
    final_out = torch.zeros((tokens, hidden_size), dtype=x.dtype, device=x.device)
    
    for i in range(num_experts):
        bin_end = bins[i].item()
        padded_end = padded_bins[i].item()
        count = bin_end - current_idx_start
        
        if count > 0:
            expert_out = x[current_padded_start : current_padded_start + count]
            if weights is not None:
                w = weights[current_idx_start:bin_end].unsqueeze(1)
                expert_out = expert_out * w
            
            target_indices = indices[current_idx_start:bin_end]
            final_out.index_add_(0, target_indices, expert_out)
            
        current_idx_start = bin_end
        current_padded_start = padded_end

    # Return shape behavior matching the original wrapper
    if top_k > 1:
        return final_out
    else:
        return final_out.view(tokens, hidden_size)


def scatter(x, indices, bin_ids, weights, bins, top_k):
    return padded_scatter(x, indices, bin_ids, weights, bins, bins, top_k)


# -------------------------------------------------------------------------
# Gradient Operations (wgrad)
# -------------------------------------------------------------------------

def padded_scatter_wgrad(x, grad, indices, bin_ids, bins, padded_bins, top_k):
    """
    Computes gradients for router weights.
    Dot product between expert_output (x) and upstream gradient (grad).
    """
    assert_is_matrix(x)
    assert_is_matrix(grad)
    
    tokens = indices.shape[0] // top_k
    out = torch.empty((tokens * top_k), dtype=x.dtype, device=x.device)
    
    num_experts = len(bins)
    current_idx_start = 0
    current_padded_start = 0
    
    for i in range(num_experts):
        bin_end = bins[i].item()
        padded_end = padded_bins[i].item()
        count = bin_end - current_idx_start
        
        if count > 0:
            # 1. Get expert outputs (padded source)
            x_expert = x[current_padded_start : current_padded_start + count]
            
            # 2. Get corresponding gradients from tokens
            # indices points to the token row.
            token_indices = indices[current_idx_start:bin_end]
            grad_tokens = grad[token_indices]
            
            # 3. Compute Dot Product (sum over hidden dim)
            # (Batch, Hidden) * (Batch, Hidden) -> (Batch, 1) -> Squeeze
            dot_prod = (x_expert * grad_tokens).sum(dim=1)
            
            # 4. Store in output (which is aligned with 'indices')
            out[current_idx_start:bin_end] = dot_prod
            
        current_idx_start = bin_end
        current_padded_start = padded_end
        
    return out


def scatter_wgrad(x, grad, indices, bin_ids, bins, top_k):
    return padded_scatter_wgrad(x, grad, indices, bin_ids, bins, bins, top_k)


# -------------------------------------------------------------------------
# Binned Operations (3D fixed layout: Experts, Capacity, Hidden)
# -------------------------------------------------------------------------

def binned_gather(x, indices, weights, bins, expert_capacity, top_k):
    """
    Gathers tokens from 'x' into a fixed 3D tensor organized by expert and capacity.
    """
    assert_is_matrix(x)
    assert_is_vector(indices)
    
    num_experts = bins.shape[0]
    hidden_size = x.shape[1]
    
    out = torch.zeros((num_experts, expert_capacity, hidden_size), dtype=x.dtype, device=x.device)
    
    current_idx_start = 0
    
    for i in range(num_experts):
        bin_end = bins[i].item()
        count = bin_end - current_idx_start
        
        # Clamp count to capacity (though typical usage implies count <= capacity)
        valid_count = min(count, expert_capacity)
        
        if valid_count > 0:
            src_indices = indices[current_idx_start : current_idx_start + valid_count]
            
            gathered = x[src_indices]
            
            if weights is not None:
                w = weights[current_idx_start : current_idx_start + valid_count].unsqueeze(1)
                gathered = gathered * w
            
            # Copy into the 3D tensor slice
            out[i, :valid_count, :] = gathered
            
        current_idx_start = bin_end
        
    return out


def binned_scatter(x, indices, weights, bins, top_k):
    """
    Scatters from 3D expert buffer back to token space.
    """
    assert_is_tensor(x, 3) # (Experts, Capacity, Hidden)
    
    num_experts, expert_capacity, hidden_size = x.shape
    tokens = indices.shape[0] // top_k
    
    # Output accumulator
    final_out = torch.zeros((tokens, hidden_size), dtype=x.dtype, device=x.device)
    
    current_idx_start = 0
    
    for i in range(num_experts):
        bin_end = bins[i].item()
        count = bin_end - current_idx_start
        valid_count = min(count, expert_capacity)
        
        if valid_count > 0:
            # Slice from 3D buffer
            expert_out = x[i, :valid_count, :]
            
            if weights is not None:
                w = weights[current_idx_start : current_idx_start + valid_count].unsqueeze(1)
                expert_out = expert_out * w
            
            target_indices = indices[current_idx_start : current_idx_start + valid_count]
            
            # Accumulate
            final_out.index_add_(0, target_indices, expert_out)
            
        current_idx_start = bin_end

    if top_k > 1:
        return final_out
    else:
        return final_out.view(tokens, hidden_size)


def binned_scatter_wgrad(x, grad, indices, bins, top_k):
    """
    Computes router weight gradients for the binned layout.
    """
    assert_is_tensor(x, 3)
    assert_is_matrix(grad)
    
    tokens = indices.shape[0] // top_k
    out = torch.empty((tokens * top_k), dtype=x.dtype, device=x.device)
    
    num_experts, expert_capacity, _ = x.shape
    current_idx_start = 0
    
    for i in range(num_experts):
        bin_end = bins[i].item()
        count = bin_end - current_idx_start
        valid_count = min(count, expert_capacity)
        
        if valid_count > 0:
            # x: (Experts, Capacity, Hidden)
            x_expert = x[i, :valid_count, :]
            
            # grad: (Tokens, Hidden)
            token_indices = indices[current_idx_start : current_idx_start + valid_count]
            grad_tokens = grad[token_indices]
            
            # Dot product
            dot_prod = (x_expert * grad_tokens).sum(dim=1)
            
            out[current_idx_start : current_idx_start + valid_count] = dot_prod
            
        current_idx_start = bin_end
        
    return out
