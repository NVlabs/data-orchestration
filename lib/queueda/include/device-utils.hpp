/* Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION & AFFILIATES nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */



#pragma once

#ifdef __CUDACC__

#define __CUDA_CALLABLE__ __host__ __device__
#define __CUDA_DEVICE__ __device__

#define __CUDA_GLOBAL__ __global__

#define __CUDA_LAUNCH_BOUNDS__(T, B) __launch_bounds__(T, B)

#define __CUDA_SHARED__ __shared__


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int64_t line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line); 
        if (abort) exit(code);
    }
}

template <
  typename T
>
T* AllocOnDevice() {
  T* res;
  gpuErrchk(cudaMalloc((void**)&res, sizeof(T)));
  return res;
}

template <
  typename T
>
T* AllocArrayOnDevice(const size_t& len) {
  T* res;
  gpuErrchk(cudaMalloc((void**)&res, sizeof(T) * len));
  return res;
}
    
template <
  typename T
>
void SetDeviceValue(T* dst_ptr, T* new_p_val) {
  gpuErrchk(cudaMemcpy(dst_ptr, new_p_val, sizeof(T), cudaMemcpyHostToDevice));
}

template <
  typename T
>
void SetDeviceArray(T* dst_ptr, T* src_ptr, int arr_size) {
  gpuErrchk(cudaMemcpy(dst_ptr, src_ptr, sizeof(T)*arr_size, cudaMemcpyHostToDevice));
}


void SetDeviceString(char** dst_ptr, char* src_ptr) {
  auto len = std::strlen(src_ptr);
  char* dst;
  gpuErrchk(cudaMalloc((void**)&dst, sizeof(char)*(len+1)));
  SetDeviceArray<char>(dst, src_ptr, len+1);
  SetDeviceValue<char*>(dst_ptr, &dst);
}

template <
  typename T
>
void SetHostValue(T* dst_ptr, T* new_p_val) {
  gpuErrchk(cudaMemcpy(dst_ptr, new_p_val, sizeof(T), cudaMemcpyDeviceToHost));
}

template <
  typename T
>
void SetHostArray(T* dst_ptr, T* src_ptr, int arr_size) {
  gpuErrchk(cudaMemcpy(dst_ptr, src_ptr, sizeof(T)*arr_size, cudaMemcpyDeviceToHost));
}

#include <cuda/atomic>
#include <nvfunctional>
#include "options.hpp"


__CUDA_DEVICE__
size_t GetBlock() {
  return blockIdx.x;
}

__CUDA_DEVICE__
size_t GetWarp() {
  return threadIdx.x / queueda::options::kMaxThreadsPerWarp;
}

__CUDA_DEVICE__
size_t GetThread() {
  return threadIdx.x % queueda::options::kMaxThreadsPerWarp;
}


#else

#define __CUDA_CALLABLE__
#define __CUDA_DEVICE__

#define __CUDA_GLOBAL__

#define __CUDA_LAUNCH_BOUNDS__(T, B)

#define __CUDA_SHARED__

#define gpuErrchk(ans) ans

template <
  typename T
>
T* AllocOnDevice() {
  T* res = static_cast<T*>(std::malloc(sizeof(T)));
  assert(res);
  return res;
}

template <
  typename T
>
T* AllocArrayOnDevice(const size_t& len) {
  T* res = static_cast<T*>(std::malloc(sizeof(T) * len));
  assert(res);
  return res;
}
    
template <
  typename T
>
void SetDeviceValue(T* dst_ptr, T* new_p_val) {
  std::memcpy(dst_ptr, new_p_val, sizeof(T));
}

template <
  typename T
>
void SetDeviceArray(T* dst_ptr, T* src_ptr, int arr_size) {
  std::memcpy(dst_ptr, src_ptr, sizeof(T)*arr_size);
}


void SetDeviceString(char** dst_ptr, char* src_ptr) {
  std::strcpy(*dst_ptr, src_ptr);
}

template <
  typename T
>
void SetHostValue(T* dst_ptr, T* new_p_val) {
  std::memcpy(dst_ptr, new_p_val, sizeof(T));
}

template <
  typename T
>
void SetHostArray(T* dst_ptr, T* src_ptr, int arr_size) {
  std::memcpy(dst_ptr, src_ptr, sizeof(T)*arr_size);
}

#include <atomic>
#include <functional>
#include "options.hpp"

size_t current_block__;
size_t current_warp__;
size_t current_thread__;

__CUDA_DEVICE__
size_t GetBlock() {
  return current_block__;
}

__CUDA_DEVICE__
size_t GetWarp() {
  return current_warp__;
}

__CUDA_DEVICE__
size_t GetThread() {
  return current_thread__;
}

#endif

// Utility classes to replace std::lib on device

namespace queueda {

#ifdef __CUDACC__

using BuilderFunction = nvstd::function<void()>;
using NodeFunction = nvstd::function<void()>;

#else

using BuilderFunction = std::function<void()>;
using NodeFunction = std::function<void()>;

#endif

template <
  typename T
>
struct Tagged {
  bool is_tag_;
  union {
    T value_;
    Tag tag_;
  };

  __CUDA_CALLABLE__ inline
  Tagged() : is_tag_(false) {}

  __CUDA_CALLABLE__ inline
  Tagged(const T& val) : is_tag_(false), value_(val) {
  }

  __CUDA_CALLABLE__ inline
  Tagged(const Tag& t) : is_tag_(true), tag_(t) {
  }
};

template <
  typename A,
  typename B
>
struct Tuple {
  A a;
  B b;
};  // Assert that C++17 structured binding should work for this type.

template <
  typename A,
  typename B,
  typename C
>
struct Triple {
  A a;
  B b;
  C c;
};  // Assert that C++17 structured binding should work for this type.

template <
  typename A,
  typename B,
  typename C,
  typename D
>
struct Quadruple {
  A a;
  B b;
  C c;
  D d;
};  // Assert that C++17 structured binding should work for this type.


template <
  typename A,
  typename B
>
__CUDA_CALLABLE__ inline
Tuple<A, B>
MakeTuple(const A& a, const B& b) {
  Tuple<A, B> result;
  result.a = a;
  result.b = b;
  return result;
}

template <
  typename A,
  typename B,
  typename C
>
__CUDA_CALLABLE__ inline
Triple<A, B, C>
MakeTriple(const A& a, const B& b, const C& c) {
  Triple<A, B, C> result;
  result.a = a;
  result.b = b;
  result.c = c;
  return result;
}

template <
  typename A,
  typename B,
  typename C,
  typename D
>
__CUDA_CALLABLE__ inline
Quadruple<A, B, C, D>
MakeQuadruple(const A& a, const B& b, const C& c, const D& d) {
  Quadruple<A, B, C, D> result;
  result.a = a;
  result.b = b;
  result.c = c;
  result.d = d;
  return result;
}

}  // namespace queueda
