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

struct Inclusive {};
struct Exclusive {};

template <typename Type> struct Cumsum {

  template<
    typename InputIteratorT,
    typename OutputIteratorT>
  static void Run(void * d_temp_storage,
		  size_t & temp_storage_bytes,
		  InputIteratorT d_in,
		  OutputIteratorT d_out,
		  int num_items,
		  cudaStream_t stream = 0,
		  bool debug_synchronous = false) {
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(d_temp_storage,
					    temp_storage_bytes,
					    d_in,
					    d_out,
					    num_items,
					    stream,
					    debug_synchronous));
  }
};

template <> struct Cumsum<Inclusive> {
  template<
    typename InputIteratorT,
    typename OutputIteratorT>
  static void Run(void * d_temp_storage,
		  size_t & temp_storage_bytes,
		  InputIteratorT d_in,
		  OutputIteratorT d_out,
		  int num_items,
		  cudaStream_t stream = 0,
		  bool debug_synchronous = false) {
    CUDA_CALL(cub::DeviceScan::InclusiveSum(d_temp_storage,
					    temp_storage_bytes,
					    d_in,
					    d_out,
					    num_items,
					    stream,
					    debug_synchronous));
  }
};

template <typename SumType, typename T>
void cub_cumsum(torch::Tensor x, int dim, torch::Tensor out) {
  // Get temporary storage size.
  size_t scratchpad_bytes = 0;
  Cumsum<SumType>::Run(nullptr,
		       scratchpad_bytes,
		       x.data_ptr<T>(),
		       out.data_ptr<T>(),
		       x.size(1),
		       c10::cuda::getCurrentCUDAStream());

  // Allocate scratchpad.
  //
  // NOTE: We scale for the batch dimension so we can run in parallel.
  auto options = torch::TensorOptions()
    .dtype(torch::kInt8)
    .device(x.device());
  torch::Tensor scratchpad = torch::empty(scratchpad_bytes * x.size(0),
  					  options);

  // Run the kernel.
  //
  // NOTE: Using different streams for each issue does not appear to
  // yield performance gains for our problem set. The overhead of
  // event/stream synchronization appears to outweigh the benfits.
  // We could write a true batched cumsum, but this would require
  // significant code duplication from cub and we might move away
  // from this formulation anyways.
  for (int i = 0; i < x.size(0); ++i) {
    void* scratchpad_ptr = (int8_t*)scratchpad.data_ptr() + scratchpad_bytes * i;
    Cumsum<SumType>::Run(scratchpad_ptr,
			 scratchpad_bytes,
			 x.data_ptr<T>() + x.size(1) * i,
			 out.data_ptr<T>() + x.size(1) * i,
			 x.size(1),
			 c10::cuda::getCurrentCUDAStream());
  }
}

void exclusive_cumsum(torch::Tensor x, int dim, torch::Tensor out) {
  // Validate the input matrix.
  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(x.ndimension() == 2);
  TORCH_CHECK(x.scalar_type() == torch::kInt16 ||
	      x.scalar_type() == torch::kInt32 ||
	      x.scalar_type() == torch::kInt64);
  TORCH_CHECK(out.is_cuda());
  TORCH_CHECK(out.ndimension() == 2);
  TORCH_CHECK(out.scalar_type() == x.scalar_type());

  // NOTE: We currently only support contraction across the contiguous
  // dimension in the matrix.
  TORCH_CHECK(dim == 1);

  switch (x.scalar_type()) {
  case torch::kInt16:
    cub_cumsum<Exclusive, short>(x, dim, out);
    return;
  case torch::kInt32:
    cub_cumsum<Exclusive, int>(x, dim, out);
    return;
  }
  TORCH_CHECK(x.scalar_type() == torch::kInt64);
  cub_cumsum<Exclusive, long>(x, dim, out);
}

void inclusive_cumsum(torch::Tensor x, int dim, torch::Tensor out) {
  // Validate the input matrix.
  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(x.ndimension() == 2);
  TORCH_CHECK(x.scalar_type() == torch::kInt16 ||
	      x.scalar_type() == torch::kInt32 ||
	      x.scalar_type() == torch::kInt64);
  TORCH_CHECK(out.is_cuda());
  TORCH_CHECK(out.ndimension() == 2);
  TORCH_CHECK(out.scalar_type() == x.scalar_type());

  // NOTE: We currently only support contraction across the contiguous
  // dimension in the matrix.
  TORCH_CHECK(dim == 1);

  switch (x.scalar_type()) {
  case torch::kInt16:
    cub_cumsum<Inclusive, short>(x, dim, out);
    return;
  case torch::kInt32:
    cub_cumsum<Inclusive, int>(x, dim, out);
    return;
  }
  TORCH_CHECK(x.scalar_type() == torch::kInt64);
  cub_cumsum<Inclusive, long>(x, dim, out);
}

} // namespace megablocks

#undef CUB_WRAPPED_NAMESPACE
