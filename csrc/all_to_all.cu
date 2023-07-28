#include "all_to_all.h"

#include <cstdint>
#include <cstring>
#include <string>

#include <nccl.h>
#include <c10/cuda/CUDAStream.h>

namespace megablocks {
namespace internal {

// TODO(tgale): Is it safe to assume we have these features?
// #ifndef HAS_NCCL_BF16_DATATYPE
// #error "Expected support for bf16 data"
// #endif  // HAS_NCCL_BF16_DATATYPE

// #ifndef NCCL_HAS_COMM_NONBLOCKING
// #error "Expected support for non-blocking communicators"
// #endif  // NCCL_HAS_COMM_NONBLOCKING

#define NCCL_CHECK(code)				    \
  do {                                                      \
    ncclResult_t status = code;				    \
    TORCH_CHECK(status == ncclSuccess, status);		    \
  } while (0)

ncclUniqueId ExtractNcclUniqueId(torch::Tensor unique_id) {
  TORCH_CHECK(unique_id.is_cpu());
  TORCH_CHECK(unique_id.ndimension() == 1);
  TORCH_CHECK(unique_id.size(0) == sizeof(ncclUniqueId));

  ncclUniqueId id;
  std::memcpy(&id, unique_id.data_ptr(), sizeof(ncclUniqueId));
  return id;
}

ncclComm_t CreateNcclComm(torch::Tensor unique_id, int world_size, int rank) {
  ncclUniqueId id = ExtractNcclUniqueId(unique_id);
  NCCL_CHECK(ncclGroupStart());
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, rank));
  NCCL_CHECK(ncclGroupEnd());
  return comm;
}

ncclComm_t& GetNcclComm(torch::Tensor unique_id, int world_size, int rank) {
  static ncclComm_t comm = CreateNcclComm(unique_id, world_size, rank);
  return comm;
}

at::cuda::CUDAStream& GetNcclStream() {
  static auto stream = at::cuda::getStreamFromPool();
  return stream;
}

std::vector<at::cuda::CUDAEvent>& GetNcclEvents() {
  static std::vector<at::cuda::CUDAEvent> events(2);
  return events;
}

ncclDataType_t GetNcclDataType(torch::Tensor x) {
  switch (x.scalar_type()) {
    case at::kFloat:
      return ncclDataType_t::ncclFloat;
    case at::kHalf:
      return ncclDataType_t::ncclHalf;
    case at::kDouble:
      return ncclDataType_t::ncclDouble;
    case at::kLong:
      return ncclDataType_t::ncclInt64;
    case at::kInt:
      return ncclDataType_t::ncclInt;
    case at::kChar:
      return ncclDataType_t::ncclChar;
    case at::kByte:
      return ncclDataType_t::ncclUint8;
    case at::kBool:
      return ncclDataType_t::ncclUint8;
    case at::kBFloat16:
      return ncclDataType_t::ncclBfloat16;
    default:
      TORCH_CHECK(false, "Unconvertible NCCL type ", x.scalar_type());
  }
}

size_t BytesPerElement(torch::Tensor x) {
  switch (x.scalar_type()) {
  case at::kFloat:
    return 4;
  case at::kHalf:
    return 2;
  case at::kDouble:
    return 8;
  case at::kLong:
    return 8;
  case at::kInt:
    return 4;
  case at::kChar:
    return 1;
  case at::kByte:
    return 1;
  case at::kBFloat16:
    return 2;
  default:
    TORCH_CHECK(false, "Unsupported type ", x.scalar_type());
  }
}

#if defined(NCCL_MAJOR) && ((NCCL_MAJOR > 2) || ((NCCL_MAJOR == 2) && (NCCL_MINOR > 13)))

bool NcclShouldSendRecv(size_t value) { return true; }

#else

// Older versions of NCLL use 0 byte messages for synchronization.
bool NcclShouldSendRecv(size_t value) { return value != 0; }

#endif

}  // namespace internal


torch::Tensor nccl_get_unique_id() {
  auto options = torch::TensorOptions()
    .dtype(at::kByte).device(torch::kCPU);
  auto out = torch::empty(sizeof(ncclUniqueId), options);

  ncclUniqueId id;
  NCCL_CHECK(ncclGetUniqueId(&id));
  std::memcpy(out.data_ptr(), &id, sizeof(ncclUniqueId));
  return out;
}

void create_nccl_comm(torch::Tensor unique_id, int world_size, int rank) {
  // NOTE: The first call initializes the communicator.
  internal::GetNcclComm(unique_id, world_size, rank);
}

torch::Tensor block_current_stream(torch::Tensor x) {
  // NOTE: The second event is used to block the current stream.
  auto& event = internal::GetNcclEvents()[1];
  event.block(at::cuda::getCurrentCUDAStream(x.device().index()));
  return x;
}

// x.shape: (tokens, hidden_size)
// recv_counts: (world_size)
// send_counts: (world_size)
torch::Tensor all_to_all(torch::Tensor x,
			 const std::vector<size_t> &recv_counts,
			 const std::vector<size_t> &send_counts) {
  // Verify the input tensor is on GPU.
  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(x.ndimension() == 2);
  const size_t tokens = x.size(0), hidden_size = x.size(1);
  const size_t element_bytes = internal::BytesPerElement(x);

  // Validate the input split sizes. The total number of elements
  // across the splits must be equal to the number of input tokens.
  std::vector<size_t> send_offsets(send_counts.size(), 0);
  size_t total_send = 0;
  for (size_t i = 0; i < send_counts.size(); ++i) {
    send_offsets[i] = total_send * hidden_size * element_bytes;
    total_send += send_counts[i];
  }
  TORCH_CHECK(total_send == tokens);

  // Calculate the number of tokens that will be received. Create
  // the output tensor for the all-to-all. Note that this tensor
  // is associated with the current stream.
  std::vector<size_t> recv_offsets(recv_counts.size(), 0);
  size_t total_recv = 0;
  for (size_t i = 0; i < recv_counts.size(); ++i) {
    recv_offsets[i] = total_recv * hidden_size * element_bytes;
    total_recv += recv_counts[i];
  }
  auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
  torch::Tensor out = torch::empty(
    {(int64_t)total_recv, (int64_t)hidden_size}, options);

  // Get NCLL metadata from the process group.
  auto& comm = internal::GetNcclComm(x, 0, 0);
  auto& stream = internal::GetNcclStream();
  auto& events = internal::GetNcclEvents();

  // Validate the number of ranks.
  int world_size;
  NCCL_CHECK(ncclCommCount(comm, &world_size));
  TORCH_CHECK(world_size == send_counts.size());
  TORCH_CHECK(world_size == recv_counts.size());

  // PyTorch runs NCCL kernels on a special stream s.t. they can overlap with
  // computation. The input tensors are allocated on the compute stream, but
  // communication kernels must wait for pending operations on the input
  // tensor to finish before they can start. We insert an event into the
  // current stream and force the NCCL stream to block on it to maintain
  // correctness.
  events[0].record(at::cuda::getCurrentCUDAStream(x.device().index()));
  events[0].block(stream);

  // Issue the communication primitives.
  auto type = internal::GetNcclDataType(x);
  auto send_ptr = (uint8_t*)x.data_ptr();
  auto recv_ptr = (uint8_t*)out.data_ptr();

  NCCL_CHECK(ncclGroupStart());
  for (int rank = 0; rank < world_size; ++rank) {
    if (internal::NcclShouldSendRecv(send_counts[rank])) {
      NCCL_CHECK(ncclSend(send_ptr + send_offsets[rank],
			  send_counts[rank],
			  type,
			  rank,
			  comm,
			  stream));
    }
    if (internal::NcclShouldSendRecv(recv_counts[rank])) {
      NCCL_CHECK(ncclRecv(recv_ptr + recv_offsets[rank],
			  recv_counts[rank],
			  type,
			  rank,
			  comm,
			  stream));
    }
  }
  NCCL_CHECK(ncclGroupEnd());

  // The input and output tensor are allocated on the compute stream. Record
  // the NCCL stream to avoid having these freed before communication finishes.
  c10::cuda::CUDACachingAllocator::recordStream(x.storage().data_ptr(), stream);
  c10::cuda::CUDACachingAllocator::recordStream(out.storage().data_ptr(), stream);

  // Record an event in the NCCL stream s.t. the caller can block on the results
  // of the communication. Use a different event than we used to guarantee the
  // communication ops wait until the input is ready.
  events[1].record(stream);
  return out;
}

}  // namespace megablocks
