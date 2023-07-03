#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#define CUDA_CALL(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    TORCH_CHECK(status == cudaSuccess, err);		    \
  } while (0)

namespace megablocks {
namespace padded {

template <typename T, int AToB, int kThreadsPerBlock, typename LoadType>
__global__ void __launch_bounds__(kThreadsPerBlock)
PaddedCopyKernel(T * __restrict__ a,
		 T * __restrict__ b,
		 int num_columns,
		 const int * __restrict__ indices,
		 const int * __restrict__ bin_ids,
		 const int * __restrict__ bins,
		 const int * __restrict__ padded_bins) {
  // Our index into array 'a'.
  int index_a = blockIdx.x;

  // One threadblock per row in 'a'. Array 'b' has
  // greater or equal number of rows (could be padded).
  int bin_idx = __ldg(bin_ids + index_a);

  // Now we know what bin we're assigned to, but we need to
  // know how many threadblocks were assigned to earlier bins
  // so we can offset in our bin properly.
  int offset_in_bin = blockIdx.x;
  if (bin_idx > 0) offset_in_bin -= __ldg(bins + bin_idx - 1);

  // Load the starting index of our bin in array 'b'.
  int index_b = offset_in_bin;
  if (bin_idx > 0) index_b += __ldg(padded_bins + bin_idx - 1);

  // Load the index to gather from/scatter to in array 'a'.
  index_a = __ldg(indices + index_a);

  // Copy between the two arrays as requested.
  //
  // NOTE: If array 'b' is the output we have zeroed it
  // prior to this kernel for correctness.
  a += index_a * num_columns;
  b += index_b * num_columns;
  LoadType *a_v = reinterpret_cast<LoadType*>(a) + threadIdx.x;
  LoadType *b_v = reinterpret_cast<LoadType*>(b) + threadIdx.x;
  constexpr int kVectorWidth = sizeof(LoadType) / sizeof(__half);
  constexpr int kValuesPerLoad = kThreadsPerBlock * kVectorWidth;
  const int tid = threadIdx.x * kVectorWidth;
  for (; tid < num_columns; num_columns -= kValuesPerLoad) {
    // Copy in the specified direction.
    if (AToB) {*b_v = __ldg(a_v);} else {*a_v = __ldg(b_v);}
    a_v += kThreadsPerBlock;
    b_v += kThreadsPerBlock;
  }
}

template <typename T>
cudaError_t PaddedGather(T * in,
			 int in_rows,
			 int in_columns,
			 T * out,
			 int out_rows,
			 const int * indices,
			 const int * bin_ids,
			 const int * bins,
			 const int * padded_bins,
			 cudaStream_t stream) {
  // Zero the output prior to gathering.
  size_t output_bytes = out_rows * in_columns * sizeof(T);
  CUDA_CALL(cudaMemsetAsync(out, 0, output_bytes, stream));

  // Launch the gather kernel.
  if (in_columns >= 128 && ((in_columns % 2) == 0)) {
    // For larger problems with aligned elements.
    const int kThreadsPerBlock = 64;
    dim3 block_dim(kThreadsPerBlock, 1, 1);
    dim3 grid_dim(in_rows, 1, 1);
    PaddedCopyKernel<T, 1, kThreadsPerBlock, __half2><<<
      grid_dim, block_dim, 0, stream>>>(in,
					out,
					in_columns,
					indices,
					bin_ids,
					bins,
					padded_bins);
    return cudaGetLastError();
  } else if (in_columns >= 32 && ((in_columns % 2) == 0)) {
    // For small problems with aligned elements.
    const int kThreadsPerBlock = 32;
    dim3 block_dim(kThreadsPerBlock, 1, 1);
    dim3 grid_dim(in_rows, 1, 1);
    PaddedCopyKernel<T, 1, kThreadsPerBlock, __half2><<<
      grid_dim, block_dim, 0, stream>>>(in,
					out,
					in_columns,
					indices,
					bin_ids,
					bins,
					padded_bins);
    return cudaGetLastError();
  }
  // Default. For small and unaligned problems.
  const int kThreadsPerBlock = 32;
  dim3 block_dim(kThreadsPerBlock, 1, 1);
  dim3 grid_dim(in_rows, 1, 1);
  PaddedCopyKernel<T, 1, kThreadsPerBlock, __half><<<
    grid_dim, block_dim, 0, stream>>>(in,
				      out,
				      in_columns,
				      indices,
				      bin_ids,
				      bins,
				      padded_bins);
  return cudaGetLastError();
}

template <typename T>
cudaError_t PaddedScatter(T * in,
			  int in_rows,
			  int in_columns,
			  T * out,
			  int out_rows,
			  const int * indices,
			  const int * bin_ids,
			  const int * bins,
			  const int * padded_bins,
			  cudaStream_t stream) {
  // Launch the scatter kernel.
  if (in_columns >= 128 && ((in_columns % 2) == 0)) {
    // For larger problems with aligned elements.
    const int kThreadsPerBlock = 64;
    dim3 block_dim(kThreadsPerBlock, 1, 1);
    dim3 grid_dim(out_rows, 1, 1);
    PaddedCopyKernel<T, 0, kThreadsPerBlock, __half2><<<
      grid_dim, block_dim, 0, stream>>>(out,
					in,
					in_columns,
					indices,
					bin_ids,
					bins,
					padded_bins);
    return cudaGetLastError();
  } else if (in_columns >= 32 && ((in_columns % 2) == 0)) {
    // For small problems with aligned elements.
    const int kThreadsPerBlock = 32;
    dim3 block_dim(kThreadsPerBlock, 1, 1);
    dim3 grid_dim(out_rows, 1, 1);
    PaddedCopyKernel<T, 0, kThreadsPerBlock, __half2><<<
      grid_dim, block_dim, 0, stream>>>(out,
					in,
					in_columns,
					indices,
					bin_ids,
					bins,
					padded_bins);
    return cudaGetLastError();
  }
  // Default. For small and unaligned problems.
  const int kThreadsPerBlock = 32;
  dim3 block_dim(kThreadsPerBlock, 1, 1);
  dim3 grid_dim(out_rows, 1, 1);
  PaddedCopyKernel<T, 0, kThreadsPerBlock, __half><<<
    grid_dim, block_dim, 0, stream>>>(out,
				      in,
				      in_columns,
				      indices,
				      bin_ids,
				      bins,
				      padded_bins);
  return cudaGetLastError();
}

}  // namespace padded

torch::Tensor padded_gather(torch::Tensor in,
			    torch::Tensor indices,
			    torch::Tensor bin_ids,
			    torch::Tensor bins,
			    torch::Tensor padded_bins) {
  // Validate the inputs.
  TORCH_CHECK(in.is_cuda());
  TORCH_CHECK(in.ndimension() == 2);
  TORCH_CHECK(in.scalar_type() == torch::kFloat16 ||
	      in.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(indices.is_cuda());
  TORCH_CHECK(indices.ndimension() == 1);
  TORCH_CHECK(indices.scalar_type() == torch::kInt);
  TORCH_CHECK(bin_ids.is_cuda());
  TORCH_CHECK(bin_ids.ndimension() == 1);
  TORCH_CHECK(bin_ids.scalar_type() == torch::kInt);
  TORCH_CHECK(bins.is_cuda());
  TORCH_CHECK(bins.ndimension() == 1);
  TORCH_CHECK(bins.scalar_type() == torch::kInt);
  TORCH_CHECK(padded_bins.is_cuda());
  TORCH_CHECK(padded_bins.ndimension() == 1);
  TORCH_CHECK(padded_bins.scalar_type() == torch::kInt);

  // Shape validation.
  TORCH_CHECK(in.size(0) == indices.size(0));
  TORCH_CHECK(in.size(0) == bin_ids.size(0));
  TORCH_CHECK(bins.size(0) == padded_bins.size(0));

  // Construct the output.
  //
  // NOTE: Because of the padding, the output size is dynamic.
  // We load the final padded bin bound to get the output rows.
  int num_bins = bins.size(0);
  int out_rows = 0;
  CUDA_CALL(cudaMemcpyAsync(&out_rows,
			    padded_bins.data_ptr<int>() + num_bins - 1,
			    sizeof(int),
			    cudaMemcpyDeviceToHost,
			    c10::cuda::getCurrentCUDAStream()));
  int in_columns = in.size(1);
  torch::Tensor out = torch::empty({out_rows, in_columns}, in.options());

  // Exit early if there is not work to do.
  if (out.numel() == 0) return out;


  if (in.scalar_type() == torch::kFloat16) {
    CUDA_CALL(padded::PaddedGather(in.data_ptr<c10::Half>(),
				   in.size(0),
				   in_columns,
				   out.data_ptr<c10::Half>(),
				   out_rows,
				   indices.data_ptr<int>(),
				   bin_ids.data_ptr<int>(),
				   bins.data_ptr<int>(),
				   padded_bins.data_ptr<int>(),
				   c10::cuda::getCurrentCUDAStream()));
  } else {
    CUDA_CALL(padded::PaddedGather(in.data_ptr<c10::BFloat16>(),
				   in.size(0),
				   in_columns,
				   out.data_ptr<c10::BFloat16>(),
				   out_rows,
				   indices.data_ptr<int>(),
				   bin_ids.data_ptr<int>(),
				   bins.data_ptr<int>(),
				   padded_bins.data_ptr<int>(),
				   c10::cuda::getCurrentCUDAStream()));
  }
  return out;
}

torch::Tensor padded_scatter(torch::Tensor in,
			     torch::Tensor indices,
			     torch::Tensor bin_ids,
			     torch::Tensor bins,
			     torch::Tensor padded_bins) {
  // Validate the inputs.
  TORCH_CHECK(in.is_cuda());
  TORCH_CHECK(in.ndimension() == 2);
  TORCH_CHECK(in.scalar_type() == torch::kFloat16 ||
	      in.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(indices.is_cuda());
  TORCH_CHECK(indices.ndimension() == 1);
  TORCH_CHECK(indices.scalar_type() == torch::kInt);
  TORCH_CHECK(bin_ids.is_cuda());
  TORCH_CHECK(bin_ids.ndimension() == 1);
  TORCH_CHECK(bin_ids.scalar_type() == torch::kInt);
  TORCH_CHECK(bins.is_cuda());
  TORCH_CHECK(bins.ndimension() == 1);
  TORCH_CHECK(bins.scalar_type() == torch::kInt);
  TORCH_CHECK(padded_bins.is_cuda());
  TORCH_CHECK(padded_bins.ndimension() == 1);
  TORCH_CHECK(padded_bins.scalar_type() == torch::kInt);

  // Shape validation.
  TORCH_CHECK(bin_ids.size(0) == indices.size(0));
  TORCH_CHECK(bins.size(0) == padded_bins.size(0));

  // Construct the output.
  int out_rows = indices.size(0);
  int in_columns = in.size(1);
  torch::Tensor out = torch::empty({out_rows, in_columns}, in.options());

  // Exit early if there is not work to do.
  if (out.numel() == 0) return out;

  if (in.scalar_type() == torch::kFloat16) {
  CUDA_CALL(padded::PaddedScatter(in.data_ptr<c10::Half>(),
				  in.size(0),
				  in_columns,
				  out.data_ptr<c10::Half>(),
				  out_rows,
				  indices.data_ptr<int>(),
				  bin_ids.data_ptr<int>(),
				  bins.data_ptr<int>(),
				  padded_bins.data_ptr<int>(),
				  c10::cuda::getCurrentCUDAStream()));
  } else {
    CUDA_CALL(padded::PaddedScatter(in.data_ptr<c10::BFloat16>(),
				    in.size(0),
				    in_columns,
				    out.data_ptr<c10::BFloat16>(),
				    out_rows,
				    indices.data_ptr<int>(),
				    bin_ids.data_ptr<int>(),
				    bins.data_ptr<int>(),
				    padded_bins.data_ptr<int>(),
				    c10::cuda::getCurrentCUDAStream()));
  }
  return out;
}

}  // namespace megablocks
