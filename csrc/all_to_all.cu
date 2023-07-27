#include "all_to_all.h"

#include <cstdint>
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

ncclComm_t CreateNcclComm(int world_size, int rank) {
  ncclUniqueId id;
  std::cout << "creating id" << std::endl;
  NCCL_CHECK(ncclGetUniqueId(&id));
  std::cout << "start group" << std::endl;
  NCCL_CHECK(ncclGroupStart());
  std::cout << "create comm" << std::endl;
  ncclComm_t comm;
  std::cout << "comm init" << std::endl;
  std::cout << "world_size, rank = " << world_size << ", " << rank << std::endl;
  NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, rank));
  std::cout << "end group" << std::endl;
  NCCL_CHECK(ncclGroupEnd());
  std::cout << "return" << std::endl;
  return comm;
}

ncclComm_t& GetNcclComm(int world_size, int rank) {
  static ncclComm_t comm = CreateNcclComm(world_size, rank);
  return comm;
}

at::cuda::CUDAStream& GetNcclStream() {
  static auto stream = at::cuda::getStreamFromPool();
  return stream;
}

at::cuda::CUDAEvent& GetNcclEvent() {
  static auto event = at::cuda::CUDAEvent();
  return event;
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

torch::Tensor block_current_stream(torch::Tensor x) {
  auto& event = internal::GetNcclEvent();
  event.block(at::cuda::getCurrentCUDAStream(x.device().index()));
  return x;
}

// x.shape: (tokens, hidden_size)
// recv_counts: (world_size)
// send_counts: (world_size)
torch::Tensor all_to_all(torch::Tensor x,
			 const std::vector<size_t> &recv_counts,
			 const std::vector<size_t> &send_counts,
			 int world_size, int rank) {
  std::cout << "inside all2all" << std::endl;
  // Verify the input tensor is on GPU.
  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(x.ndimension() == 2);
  const size_t tokens = x.size(0), hidden_size = x.size(1);
  const size_t element_bytes = internal::BytesPerElement(x);

  // Validate the number of ranks.
  TORCH_CHECK(world_size == send_counts.size());
  TORCH_CHECK(world_size == recv_counts.size());

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
  std::cout << "Making NCCL communicator" << std::endl;
  auto& comm = internal::GetNcclComm(world_size, rank);
  std::cout << "Got communicator" << std::endl;
  auto& stream = internal::GetNcclStream();
  auto& event = internal::GetNcclEvent();
  std::cout << "got comm/stream/event" << std::endl;

  // PyTorch runs NCCL kernels on a special stream s.t. they can overlap with
  // computation. The input tensors are allocated on the compute stream, but
  // communication kernels must wait for pending operations on the input
  // tensor to finish before they can start. We insert an event into the
  // current stream and force the NCCL stream to block on it to maintain
  // correctness.
  event.record(at::cuda::getCurrentCUDAStream(x.device().index()));
  event.block(stream);

  std::cout << "blocked on the event" << std::endl;

  // Issue the communication primitives.
  auto type = internal::GetNcclDataType(x);
  auto send_ptr = (uint8_t*)x.data_ptr();
  auto recv_ptr = (uint8_t*)out.data_ptr();

  std::cout << "done with type" << std::endl;

  std::cout << "starting send/recv" << std::endl;
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
  std::cout << "done with send/recv" << std::endl;

  // The input and output tensor are allocated on the compute stream. Record
  // the NCCL stream to avoid having these freed before communication finishes.
  c10::cuda::CUDACachingAllocator::recordStream(x.storage().data_ptr(), stream);
  c10::cuda::CUDACachingAllocator::recordStream(out.storage().data_ptr(), stream);

  // Record the event in the NCCL stream s.t. the caller can block on the results
  // of the communication.
  event.record(stream);
  std::cout << "returning" << std::endl;
  return out;
}

}  // namespace megablocks
