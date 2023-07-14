#include "cumsum.h"
#include "histogram.h"
#include "indices.h"
#include "replicate.h"
#include "sort.h"

#include <torch/extension.h>

namespace megablocks {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("exclusive_cumsum", &exclusive_cumsum, "batched exclusive cumsum.");
  m.def("histogram", &histogram, "even width histogram.");
  m.def("inclusive_cumsum", &inclusive_cumsum, "batched inclusive cumsum");
  m.def("indices", &indices, "indices construction for sparse matrix.");
  m.def("replicate_forward", &replicate_forward, "(fwd) replicate a vector dynamically.");
  m.def("replicate_backward", &replicate_backward, "(bwd) replicate a vector dynamically.");
  m.def("sort", &sort, "key/value sort.");
}

}  // namespace megablocks
