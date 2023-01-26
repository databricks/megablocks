#undef CUB_WRAPPED_NAMESPACE
#define CUB_WRAPPED_NAMESPACE megablocks

#include <cstdint>

#include <cub/cub.cuh>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#define CUDA_CALL(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    TORCH_CHECK(status == cudaSuccess, err);		    \
  } while (0)

namespace megablocks {
namespace replicate {

template <typename T, int kThreadsPerBlock>
__global__ void __launch_bounds__(kThreadsPerBlock)
  ReplicateForwardKernel(T * __restrict__ x,
			 int * __restrict__ bins,
			 T * __restrict__ out,
			 int columns) {
  // Offset to this threadblocks batch.
  //
  // x is [batch_size, num_bins]
  // out is [batch_size, columns]
  // bins is [num_bins]
  int batch_idx = blockIdx.y;
  int num_bins = gridDim.x;
  x += batch_idx * num_bins;
  out += batch_idx * columns;

  // Load the start/end for this bin.
  int bin_idx = blockIdx.x;
  int start = 0;
  if (bin_idx > 0) start = __ldg(bins + bin_idx - 1);
  int end = __ldg(bins + bin_idx);

  // Load the value to replicate.
  T value = __ldg((T*)x + bin_idx);

  // Offset to this threadblocks bin and this threads
  // offset within the bin.
  int bin_offset = blockIdx.z * kThreadsPerBlock + threadIdx.x;
  out += start + bin_offset;

  // Replicate the value to the output.
  //
  // TODO(tgale): Vectorize these stores.
  int num_elements = end - start;
  const int kElementsPerLoop = gridDim.z * kThreadsPerBlock;
  T *out_ptr = (T*)out;
  for (; bin_offset < num_elements; num_elements -= kElementsPerLoop) {
    *out_ptr = value;
    out_ptr += kElementsPerLoop;
  }
}

template <typename T>
cudaError_t ReplicateForward(T *x,
			     int batch_size,
			     int num_bins,
			     int *bins,
			     T *out,
			     int columns,
			     cudaStream_t stream) {
  const int kThreadsPerBlock = 64;
  dim3 block_dim(kThreadsPerBlock, 1, 1);
  int group_size = std::ceil((float)columns / (num_bins * kThreadsPerBlock));
  dim3 grid_dim(num_bins, batch_size, group_size);
  ReplicateForwardKernel<T, kThreadsPerBlock><<<
    grid_dim, block_dim, 0, stream>>>(x, bins, out, columns);
  return cudaGetLastError();
}

void cub_segmented_reduce(torch::Tensor grad,
			  torch::Tensor bins,
			  torch::Tensor out,
			  cudaStream_t stream) {
  // Append a zero to the bin boundaries for CUB.
  torch::Tensor offsets = torch::empty(bins.numel() + 1, bins.options());
  CUDA_CALL(cudaMemsetAsync(offsets.data_ptr<int>(),
			    0,
			    offsets.numel() * sizeof(int),
			    stream));
  CUDA_CALL(cudaMemcpyAsync(offsets.data_ptr<int>() + 1,
			    bins.data_ptr<int>(),
			    bins.numel() * sizeof(int),
			    cudaMemcpyDeviceToDevice,
			    stream));

  // Get temporary buffer size.
  size_t scratchpad_bytes = 0;
  CUDA_CALL(cub::DeviceSegmentedReduce::Sum(nullptr,
					    scratchpad_bytes,
					    grad.data_ptr<c10::Half>(),
					    out.data_ptr<c10::Half>(),
					    bins.numel(),
					    offsets.data_ptr<int>(),
					    offsets.data_ptr<int>() + 1,
					    stream));

  // Allocate scratchpad.
  auto options = torch::TensorOptions()
    .dtype(torch::kInt8)
    .device(grad.device());
  torch::Tensor scratchpad = torch::empty(scratchpad_bytes, options);

  // Run the kernel for each batch item.
  for (int i = 0; i < grad.size(0); ++i) {
    int num_bins = out.size(1);
    int num_values = grad.size(1);
    CUDA_CALL(cub::DeviceSegmentedReduce::Sum(scratchpad.data_ptr<int8_t>(),
					      scratchpad_bytes,
					      grad.data_ptr<c10::Half>() + i * num_values,
					      out.data_ptr<c10::Half>() + i * num_bins,
					      bins.numel(),
					      offsets.data_ptr<int>(),
					      offsets.data_ptr<int>() + 1,
					      stream));
  }
}

}  // namespace replicate

void replicate_forward(torch::Tensor x,
		       torch::Tensor bins,
		       torch::Tensor out) {
  // Validate the inputs.
  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(x.ndimension() == 2);
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 ||
	      x.scalar_type() == torch::kInt16 ||
	      x.scalar_type() == torch::kInt32);
  TORCH_CHECK(bins.is_cuda());
  TORCH_CHECK(bins.ndimension() == 1);
  TORCH_CHECK(bins.scalar_type() == torch::kInt);
  TORCH_CHECK(out.is_cuda());
  TORCH_CHECK(out.ndimension() == 2);
  TORCH_CHECK(out.scalar_type() == x.scalar_type());

  // Batch dimensions should match for input/output.
  TORCH_CHECK(x.size(0) == out.size(0));

  // One input for each bin (in each batch).
  TORCH_CHECK(x.size(1) == bins.size(0));

  // Exit early if there is no work to do.
  if (out.numel() == 0) return;

  switch (x.scalar_type()) {
  case torch::kFloat16:
    CUDA_CALL(replicate::ReplicateForward(x.data_ptr<c10::Half>(),
					  x.size(0),
					  x.size(1),
					  bins.data_ptr<int>(),
					  out.data_ptr<c10::Half>(),
					  out.size(1),
					  c10::cuda::getCurrentCUDAStream()));
    return;
  case torch::kInt32:
    CUDA_CALL(replicate::ReplicateForward(x.data_ptr<int>(),
					  x.size(0),
					  x.size(1),
					  bins.data_ptr<int>(),
					  out.data_ptr<int>(),
					  out.size(1),
					  c10::cuda::getCurrentCUDAStream()));
    return;
  }
  TORCH_CHECK(x.scalar_type() == torch::kInt16);
  CUDA_CALL(replicate::ReplicateForward(x.data_ptr<short>(),
					x.size(0),
					x.size(1),
					bins.data_ptr<int>(),
					out.data_ptr<short>(),
					out.size(1),
					c10::cuda::getCurrentCUDAStream()));
}

void replicate_backward(torch::Tensor grad,
			torch::Tensor bins,
			torch::Tensor out) {
  // Validate the inputs.
  TORCH_CHECK(grad.is_cuda());
  TORCH_CHECK(grad.ndimension() == 2);
  TORCH_CHECK(grad.scalar_type() == torch::kFloat16);
  TORCH_CHECK(bins.is_cuda());
  TORCH_CHECK(bins.ndimension() == 1);
  TORCH_CHECK(bins.scalar_type() == torch::kInt);
  TORCH_CHECK(out.is_cuda());
  TORCH_CHECK(out.ndimension() == 2);
  TORCH_CHECK(out.scalar_type() == torch::kFloat16);

  // Batch dimensions should match for input/output.
  TORCH_CHECK(grad.size(0) == out.size(0));

  // One output for each bin (in each batch).
  TORCH_CHECK(out.size(1) == bins.size(0));

  replicate::cub_segmented_reduce(grad, bins, out, c10::cuda::getCurrentCUDAStream());
}

}  // namespace megablocks

#undef CUDA_CALL
#undef CUB_WRAPPED_NAMESPACE
