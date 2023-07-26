#include "all_to_all.h"

#include <string>

#include <c10/cuda/CUDAStream.h>


namespace megablocks {

class MBPG : public c10d::ProcessGroupNCCL {
public:

  ncclComm_t GetNcclComm(const at::Device &device) {
    auto device_key = std::to_string(device.index());

    // TODO(tgale): Maybe do this inline.
    std::vector<at::Device> devices = {device};
    auto &comms = this->getNCCLComm(device_key,
				    devices,
				    c10d::OpType::ALLTOALL_BASE);

    // NOTE: We should only get a single communicator back.
    TORCH_CHECK(comms.size() == 1);
    return comms[0]->getNcclComm();
  }

};

void all_to_all(torch::Tensor x,
		const std::vector<int> &output_split_sizes,
		const std::vector<int> &input_split_sizes,
		c10d::ProcessGroupNCCL &pg) {
  MBPG* mbpg = (MBPG*)(void*)&pg;
  ncclComm_t comm = mbpg->GetNcclComm(x.device());
}

#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR >= 2)

void all_to_all(torch::Tensor x,
		const std::vector<int> &output_split_sizes,
		const std::vector<int> &input_split_sizes,
		c10d::ProcessGroup &pg) {
  c10d::ProcessGroupNCCL* nccl_pg = pg.getBackend(c10d::ProcessGroup::NCCL).get();
  all_to_all(x, output_split_sizes, input_split_sizes, *nccl_pg);
}

#endif  // defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR >= 2)

}  // namespace megablocks
