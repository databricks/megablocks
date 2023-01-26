#include "binned_scatter.h"
#include "cuda_util.h"
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace megablocks {
namespace scatter {

#define CUDA_CALL(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    TORCH_CHECK(status == cudaSuccess, err);		    \
  } while (0)

using VectorT = half2;
constexpr int kVectorWidth = sizeof(VectorT) / sizeof(__half);
const int kThreadsPerBlock = 64;

__global__ void __launch_bounds__(kThreadsPerBlock)
  BinnedScatterKernel(const c10::Half * __restrict__ in,
		      c10::Half * __restrict__ out,
		      int num_columns,
		      const int * __restrict__ indices,
		      const int * __restrict__ bins) {
  // Load the index bounds for our bin.
  int bin_idx = blockIdx.y;
  int start = 0;
  if (bin_idx > 0) start = __ldg(bins + bin_idx - 1);
  int end = __ldg(bins + bin_idx);
  int num_indices = end - start;

  // If this entry is out-of-bounds, return.
  int entry_idx = blockIdx.x;
  if (entry_idx >= num_indices) return;

  // Load this block's output index.
  int index = __ldg(indices + start + entry_idx);

  // Offset to this block's entry in the output.
  out += index * num_columns;
  VectorT *out_v = reinterpret_cast<VectorT*>(out) + threadIdx.x;

  // Offset to this block's entry in the input.
  int bin_size = gridDim.x;
  in += (bin_idx * bin_size + entry_idx) * num_columns;
  const VectorT *in_v = reinterpret_cast<const VectorT*>(in) + threadIdx.x;

  // Copy the input entry to the output.
  const int tid = threadIdx.x * kVectorWidth;
  constexpr int kValuesPerLoad = kThreadsPerBlock * kVectorWidth;
  for (; tid < num_columns; num_columns -= kValuesPerLoad) {
    Store(Load(in_v), out_v);
    in_v += kThreadsPerBlock;
    out_v += kThreadsPerBlock;
  }
}

cudaError_t BinnedScatter(c10::Half * in,
			  int num_rows,
			  c10::Half * out,
			  int num_bins,
			  int bin_size,
			  int num_columns,
			  const int * indices,
			  const int * bins,
			  cudaStream_t stream) {
  // Set the output to zero before scattering.
  //
  // TODO(tgale): We may want to make this a scatter-scale-add
  // for the summation at the end of the block and for the
  // gradient at the beginning of the MoE block to reduce memory
  // usage. It depends on whether this makes sense with the
  // gradient calculations.
  size_t output_bytes = num_rows * num_columns * sizeof(c10::Half);
  CUDA_CALL(cudaMemsetAsync(out, 0, output_bytes, stream));

  dim3 block_dim(kThreadsPerBlock, 1, 1);
  dim3 grid_dim(bin_size, num_bins, 1);
  BinnedScatterKernel<<<grid_dim, block_dim, 0, stream>>>(in,
							  out,
							  num_columns,
							  indices,
							  bins);
  return cudaGetLastError();
}

}  // namespace scatter

torch::Tensor binned_scatter(torch::Tensor in,
			     torch::Tensor indices,
			     torch::Tensor bins) {
  // Validate the inputs.
  TORCH_CHECK(in.is_cuda());
  TORCH_CHECK(in.ndimension() == 3);
  TORCH_CHECK(in.scalar_type() == torch::kFloat16);
  TORCH_CHECK(indices.is_cuda());
  TORCH_CHECK(indices.ndimension() == 1);
  TORCH_CHECK(indices.scalar_type() == torch::kInt);
  TORCH_CHECK(bins.is_cuda());
  TORCH_CHECK(bins.ndimension() == 1);
  TORCH_CHECK(bins.scalar_type() == torch::kInt);

  // The number of bins in the input.
  TORCH_CHECK(in.size(0) == bins.size(0));
  TORCH_CHECK(in.size(2) % 2 == 0);

  // Construct the output.
  int num_rows = indices.size(0);
  int num_columns = in.size(2);
  auto options = torch::TensorOptions()
    .dtype(torch::kFloat16)
    .device(in.device());
  torch::Tensor out = torch::empty({num_rows, num_columns}, options);

  // Exit early if there is not work to do.
  if (out.numel() == 0) return out;

  // Run the kernel.
  int num_bins = in.size(0);
  int bin_size = in.size(1);
  CUDA_CALL(scatter::BinnedScatter(in.data_ptr<c10::Half>(),
				   num_rows,
				   out.data_ptr<c10::Half>(),
				   num_bins,
				   bin_size,
				   num_columns,
				   indices.data_ptr<int>(),
				   bins.data_ptr<int>(),
				   c10::cuda::getCurrentCUDAStream()));
  return out;
}

}  // namespace megablocks

#undef CUDA_CALL
