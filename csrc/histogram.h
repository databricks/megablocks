#undef CUB_WRAPPED_NAMESPACE
#define CUB_WRAPPED_NAMESPACE megablocks

#include <cstdint>

#include <cub/cub.cuh>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#define CUDA_CALL(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    TORCH_CHECK(status == cudaSuccess, err);		    \
  } while (0)

namespace megablocks {

template <typename T>
torch::Tensor cub_histogram(torch::Tensor x, int num_bins) {
  // Allocate the count buffer.
  auto options = torch::TensorOptions()
    .dtype(torch::kInt32)
    .device(x.device());
  torch::Tensor out = torch::empty({x.size(0), num_bins}, options);

  // Exit early if there is not work to do.
  if (out.numel() == 0) return out;

  // Get scratchpad size.
  size_t scratchpad_bytes = 0;
  CUDA_CALL(cub::DeviceHistogram::HistogramEven(nullptr,
						scratchpad_bytes,
						x.data_ptr<T>(),
						out.data_ptr<int>(),
						/*num_levels=*/num_bins + 1,
						/*lower_level=*/0,
						/*upper_level=*/num_bins,
						/*num_samples=*/int(x.size(1)),
						c10::cuda::getCurrentCUDAStream()));

  // Allocate scratchpad.
  options = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
  torch::Tensor scratchpad = torch::empty(scratchpad_bytes, options);

  // Run the kernel.
  for (int i = 0; i < x.size(0); ++i) {
    CUDA_CALL(cub::DeviceHistogram::HistogramEven(scratchpad.data_ptr(),
						  scratchpad_bytes,
						  x.data_ptr<T>() + x.size(1) * i,
						  out.data_ptr<int>() + out.size(1) * i,
						  /*num_levels=*/num_bins + 1,
						  /*lower_level=*/0,
						  /*upper_level=*/num_bins,
						  /*num_samples=*/int(x.size(1)),
						  c10::cuda::getCurrentCUDAStream()));
  }
  return out;
}

torch::Tensor histogram(torch::Tensor x, int num_bins) {
  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(x.ndimension() == 1 || x.ndimension() == 2);
  TORCH_CHECK(x.scalar_type() == torch::kInt16 ||
	      x.scalar_type() == torch::kInt32 ||
	      x.scalar_type() == torch::kInt64);
  bool no_batch = x.ndimension() == 1;
  if (no_batch) x = x.view({1, x.numel()});

  if (x.scalar_type() == torch::kInt16) {
    auto out = cub_histogram<short>(x, num_bins);
    return no_batch ? out.flatten() : out;
  } else if (x.scalar_type() == torch::kInt32) {
    auto out = cub_histogram<int>(x, num_bins);
    return no_batch ? out.flatten() : out;
  } else {
    TORCH_CHECK(x.scalar_type() == torch::kInt64);
    auto out = cub_histogram<long>(x, num_bins);
    return no_batch ? out.flatten() : out;
  }
}

}  // namespace megablocks

#undef CUDA_CALL
#undef CUB_WRAPPED_NAMESPACE
