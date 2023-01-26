#ifndef BLOCKPARTY_CSRC_CUDA_UTIL_H_
#define BLOCKPARTY_CSRC_CUDA_UTIL_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace megablocks {

typedef __half2 half2;

struct __align__(8) half4 {
  half2 x, y;
};

struct __align__(16) half8 {
  half2 x, y, z, w;
};

template <class To, class From>
__device__ __forceinline__ To BitCast(const From& src) noexcept {
  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T>
__device__ __forceinline__ void Store(const T& value, T* ptr) {
  *ptr = value;
}

template <typename T>
__device__ __forceinline__ T Load(const T* address) {
  return __ldg(address);
}

__device__ __forceinline__ half4 Load(const half4* address) {
  float2 x = __ldg(reinterpret_cast<const float2*>(address));
  return BitCast<half4>(x);
}

__device__ __forceinline__ half8 Load(const half8* address) {
  float4 x = __ldg(reinterpret_cast<const float4*>(address));
  return BitCast<half8>(x);
}

template <typename T>
__device__ __forceinline__ T Zero() { return 0; };

template <>
__device__ __forceinline__ half2 Zero<half2>() {
  return {(c10::Half)0., (c10::Half)0.};
};

template <>
__device__ __forceinline__ half4 Zero<half4>() {
  return {Zero<half2>(), Zero<half2>()};
};

}  // namespace megablocks

#endif  // BLOCKPARTY_CSRC_CUDA_UTIL_H_
