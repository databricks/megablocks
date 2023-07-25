#include <vector>

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace megablocks {

void all_to_all(torch::Tensor x,
		const std::vector<int> &output_split_sizes,
		const std::vector<int> &input_split_sizes,
		c10d::ProcessGroupNCCL &pg) {
  // blah blah.
}

}  // namespace megablocks
