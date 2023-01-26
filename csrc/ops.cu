#include "binned_gather.h"
#include "binned_scatter.h"
#include "cumsum.h"
#include "histogram.h"
#include "indices.h"
#include "padded_gather_scatter.h"
#include "replicate.h"
#include "sort.h"

#include <torch/extension.h>

namespace megablocks {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("binned_gather", &binned_gather, "binned gather operation.");
  m.def("binned_scatter", &binned_scatter, "binned scatter operation.");
  m.def("exclusive_cumsum", &exclusive_cumsum, "batched exclusive cumsum.");
  m.def("histogram", &histogram, "even width histogram.");
  m.def("inclusive_cumsum", &inclusive_cumsum, "batched inclusive cumsum");
  m.def("indices", &indices, "indices construction for sparse matrix.");
  m.def("padded_gather", &padded_gather, "padded gather operation.");
  m.def("padded_scatter", &padded_scatter, "padded scatter operation.");
  m.def("replicate_forward", &replicate_forward, "(fwd) replicate a vector dynamically.");
  m.def("replicate_backward", &replicate_backward, "(bwd) replicate a vector dynamically.");
  m.def("sort", &sort, "key/value sort.");
}

}  // namespace megablocks
