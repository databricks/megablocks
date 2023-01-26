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
void cub_radix_sort(torch::Tensor x,
		    int end_bit,
		    torch::Tensor x_out,
		    torch::Tensor iota_out) {
  // Get iota for values in sort.
  torch::Tensor iota = torch::arange(0, x.numel(), x.options());

  // Get temporary buffer size.
  size_t scratchpad_bytes = 0;
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(nullptr,
  					    scratchpad_bytes,
  					    x.data_ptr<T>(),
  					    x_out.data_ptr<T>(),
  					    iota.data_ptr<T>(),
  					    iota_out.data_ptr<T>(),
  					    x.numel(),
  					    /*begin_bit*/0,
  					    /*end_bit=*/end_bit,
  					    c10::cuda::getCurrentCUDAStream()));

  // Allocate scratchpad.
  auto options = torch::TensorOptions()
    .dtype(torch::kInt8)
    .device(x.device());
  torch::Tensor scratchpad = torch::empty(scratchpad_bytes, options);

  // Run the kernel.
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(scratchpad.data_ptr(),
  					    scratchpad_bytes,
  					    x.data_ptr<T>(),
  					    x_out.data_ptr<T>(),
  					    iota.data_ptr<T>(),
  					    iota_out.data_ptr<T>(),
  					    x.numel(),
  					    /*begin_bit=*/0,
  					    /*end_bit=*/end_bit,
  					    c10::cuda::getCurrentCUDAStream()));
}

void sort(torch::Tensor x,
	  int end_bit,
	  torch::Tensor x_out,
	  torch::Tensor iota_out) {
  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(x.ndimension() == 1);
  TORCH_CHECK(x.scalar_type() == torch::kInt16 ||
  	      x.scalar_type() == torch::kInt32 ||
  	      x.scalar_type() == torch::kInt64);
  TORCH_CHECK(x_out.is_cuda());
  TORCH_CHECK(x_out.ndimension() == 1);
  TORCH_CHECK(x_out.scalar_type() == x.scalar_type());
  TORCH_CHECK(iota_out.is_cuda());
  TORCH_CHECK(iota_out.ndimension() == 1);
  TORCH_CHECK(iota_out.scalar_type() == x.scalar_type());

  // Exit early if there is not work to do.
  if (x_out.numel() == 0) return;

  switch (x.scalar_type()) {
  case torch::kInt16:
    return cub_radix_sort<short>(x, end_bit, x_out, iota_out);
  case torch::kInt32:
    return cub_radix_sort<int>(x, end_bit, x_out, iota_out);
  }
  TORCH_CHECK(x.scalar_type() == torch::kInt64);
  return cub_radix_sort<long>(x, end_bit, x_out, iota_out);
}

}  // namespace megablocks

#undef CUDA_CALL
#undef CUB_WRAPPED_NAMESPACE
