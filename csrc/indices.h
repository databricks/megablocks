#include <cstdint>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#define CUDA_CALL(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    TORCH_CHECK(status == cudaSuccess, err);		    \
  } while (0)

namespace megablocks {
namespace construct_indices {

// We expect the number of outputs per block to be small. For
// example, with ffn_hidden_size=4096, we only need to write
// 32 elements per block per iteration.
const int kThreadsPerBlock = 32;

__global__ void __launch_bounds__(kThreadsPerBlock)
  ConstructIndicesKernel(short * __restrict__ indices,
			 int num_columns,
			 int block_size,
			 const int * __restrict__ padded_bins) {
  // Load the offset for this bins indices.
  int start = 0;
  if (blockIdx.x > 0) start = __ldg(padded_bins + blockIdx.x - 1);
  int end = __ldg(padded_bins + blockIdx.x);

  // Divide the start and end into blocks.
  start /= block_size;
  end /= block_size;

  // Offset the output buffer to the start of the bin.
  indices += (start + blockIdx.y) * num_columns + threadIdx.x;

  // Write the indices to the output.
  int bin_offset = blockIdx.y;
  int num_rows = end - start;
  for (; bin_offset < num_rows; num_rows -= gridDim.y) {
    short *out = indices;
    for (int bid = threadIdx.x; bid < num_columns; bid += kThreadsPerBlock) {
      *out = bid + (blockIdx.x * num_columns);
      out += kThreadsPerBlock;
    }
    indices += gridDim.y * num_columns;
  }
}

cudaError_t ConstructIndices(short * __restrict__ indices,
			     int output_block_rows,
			     int output_block_columns,
			     int block_size,
			     const int * __restrict__ padded_bins,
			     int num_bins,
			     cudaStream_t stream) {
  dim3 block_dim(kThreadsPerBlock);
  dim3 grid_dim(num_bins, (int)std::ceil((float)output_block_rows / num_bins));
  ConstructIndicesKernel<<<grid_dim, block_dim, 0, stream>>>(indices,
							     output_block_columns,
							     block_size,
							     padded_bins);
  return cudaGetLastError();
}

}  // namespace construct_indices

void indices(torch::Tensor padded_bins,
	     int block_size,
	     int output_block_rows,
	     int output_block_columns,
	     torch::Tensor out) {
  TORCH_CHECK(padded_bins.is_cuda());
  TORCH_CHECK(padded_bins.ndimension() == 1);
  TORCH_CHECK(padded_bins.scalar_type() == torch::kInt);

  TORCH_CHECK(out.is_cuda());
  TORCH_CHECK(out.ndimension() == 1);
  TORCH_CHECK(out.scalar_type() == torch::kInt16);
  TORCH_CHECK(out.numel() == (output_block_rows * output_block_columns));

  // Exit early if there is no work to do.
  if (out.numel() == 0) return;

  CUDA_CALL(construct_indices::ConstructIndices(out.data_ptr<short>(),
						output_block_rows,
						output_block_columns,
						block_size,
						padded_bins.data_ptr<int>(),
						padded_bins.numel(),
						c10::cuda::getCurrentCUDAStream()));
}

}  // namespace megablocks
