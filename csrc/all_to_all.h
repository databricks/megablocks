#include <vector>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/extension.h>

namespace megablocks {

#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR >= 2)

void all_to_all(torch::Tensor x,
		const std::vector<int> &output_split_sizes,
		const std::vector<int> &input_split_sizes,
		c10d::ProcessGroup &pg);

#else

void all_to_all(torch::Tensor x,
		const std::vector<int> &output_split_sizes,
		const std::vector<int> &input_split_sizes,
		c10d::ProcessGroupNCCL &pg);

#endif  // defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR >= 2)

}  // namespace megablocks
