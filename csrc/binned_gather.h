#include "cuda_util.h"
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace megablocks {
namespace gather {

#define CUDA_CALL(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    TORCH_CHECK(status == cudaSuccess, err);		    \
  } while (0)

using VectorT = half2;
constexpr int kVectorWidth = sizeof(VectorT) / sizeof(__half);
constexpr int kThreadsPerBlock = 64;

__global__ void __launch_bounds__(kThreadsPerBlock)
  BinnedGatherKernel(const c10::Half * __restrict__ in,
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

  // Load this block's input index.
  int entry_idx = blockIdx.x;
  int index = -1;
  if (entry_idx < num_indices) index = __ldg(indices + start + entry_idx);

  // Offset to this block's entry in the output.
  int bin_size = gridDim.x;
  out += (bin_idx * bin_size + entry_idx) * num_columns;
  VectorT *out_v = reinterpret_cast<VectorT*>(out) + threadIdx.x;

  // Offset to this block's entry in the input.
  in += index * num_columns;
  const VectorT *in_v = reinterpret_cast<const VectorT*>(in) + threadIdx.x;

  // Copy the input entry to the output.
  const int tid = threadIdx.x * kVectorWidth;
  constexpr int kValuesPerLoad = kThreadsPerBlock * kVectorWidth;
  VectorT value = Zero<VectorT>();
  for (; tid < num_columns; num_columns -= kValuesPerLoad) {
    // Only load if we have a valid index. Otherwise write zeros.
    if (index >= 0) value = Load(in_v);
    Store(value, out_v);

    in_v += kThreadsPerBlock;
    out_v += kThreadsPerBlock;
  }
}

cudaError_t BinnedGather(c10::Half * in,
			 c10::Half * out,
			 int num_bins,
			 int bin_size,
			 int num_columns,
			 const int * indices,
			 const int * bins,
			 cudaStream_t stream) {
  dim3 block_dim(kThreadsPerBlock, 1, 1);
  dim3 grid_dim(bin_size, num_bins, 1);
  BinnedGatherKernel<<<grid_dim, block_dim, 0, stream>>>(in,
							 out,
							 num_columns,
							 indices,
							 bins);
  return cudaGetLastError();
}

}  // namespace gather

torch::Tensor binned_gather(torch::Tensor in,
			    torch::Tensor indices,
			    torch::Tensor bins,
			    int bin_size) {
  // Validate the inputs.
  TORCH_CHECK(in.is_cuda());
  TORCH_CHECK(in.ndimension() == 2);
  TORCH_CHECK(in.scalar_type() == torch::kFloat16);
  TORCH_CHECK(indices.is_cuda());
  TORCH_CHECK(indices.ndimension() == 1);
  TORCH_CHECK(indices.scalar_type() == torch::kInt);
  TORCH_CHECK(bins.is_cuda());
  TORCH_CHECK(bins.ndimension() == 1);
  TORCH_CHECK(bins.scalar_type() == torch::kInt);

  // The number of tokens in the input.
  TORCH_CHECK(in.size(0) == indices.size(0));
  TORCH_CHECK(in.size(1) % gather::kVectorWidth == 0);

  // Construct the output.
  int num_columns = in.size(1);
  int num_bins = bins.size(0);
  auto options = torch::TensorOptions()
    .dtype(torch::kFloat16)
    .device(in.device());
  torch::Tensor out = torch::empty({num_bins, bin_size, num_columns}, options);

  // Exit early if there is not work to do.
  if (out.numel() == 0) return out;

  // Run the kernel.
  CUDA_CALL(gather::BinnedGather(in.data_ptr<c10::Half>(),
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
