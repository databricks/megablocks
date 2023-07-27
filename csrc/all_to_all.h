#include <vector>

#include <torch/extension.h>
#include <ATen/cuda/CUDAEvent.h>

namespace megablocks {

torch::Tensor block_current_stream(torch::Tensor x);

torch::Tensor all_to_all(torch::Tensor x,
			 const std::vector<size_t> &recv_counts,
			 const std::vector<size_t> &send_counts,
			 int world_size, int rank);

}  // namespace megablocks
