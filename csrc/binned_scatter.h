#include <torch/extension.h>

namespace megablocks {

torch::Tensor binned_scatter(torch::Tensor in,
			     torch::Tensor indices,
			     torch::Tensor bins);

}  // namespace megablocks
