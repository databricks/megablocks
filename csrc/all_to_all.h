#include <vector>

#include <torch/extension.h>
#include <ATen/cuda/CUDAEvent.h>

namespace megablocks {


torch::Tensor nccl_get_unique_id();

void create_nccl_comm(torch::Tensor unique_id, int world_size, int rank);

torch::Tensor block_current_stream(torch::Tensor x);

torch::Tensor all_to_all(torch::Tensor x,
			 const std::vector<size_t> &recv_counts,
			 const std::vector<size_t> &send_counts);

}  // namespace megablocks
