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


#include <stdarg.h>
#include <stdio.h>
#include <utility>
#include <cassert>
#include <cstring>
#include <memory>

using Tag = uint8_t;

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

#define DEFAULT_Q_SIZE 8
#define MAX_Q_BUFFERING 49152

#else

#define __CUDA_CALLABLE__
#define __CUDA_DEVICE__

#define __CUDA_GLOBAL__

#define __CUDA_LAUNCH_BOUNDS__(T, B)

#define __CUDA_SHARED__

#include <atomic>
#include <functional>

#define DEFAULT_Q_SIZE 1024
#define MAX_Q_BUFFERING 10000000

#endif 

#define TRACE_LEVEL 4
#define MAX_QS_PER_NODE 4
#define MAX_Q_FANOUT 16
#define MAX_TENSOR_RANKS 8

// Utility functions to replace std::lib on device

namespace queueda {

#ifdef __CUDACC__

using BuilderFunction = nvstd::function<void()>;

#else

using BuilderFunction = std::function<void()>;

#endif

template <
  typename T
>
__CUDA_CALLABLE__ inline
void PrintArray(T* arr, int size) {
  for (int x = 0; x < size; x++) {
    printVal(arr[x]);
  }
}

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

// Type defs

// TODO: something more useful here, with instance numbering.
using Name = const char[];

// TODO: GENERALIZE THIS A BIT
#define QUEUDA_MAX_BLOCKS_PER_GPU 128
#define QUEUDA_MAX_WARPS_PER_BLOCK 64
#define QUEUDA_MAX_THREADS_PER_WARP 32
#define QUEUDA_MAX_SEQUENTIAL_NODES 8

class Node;

// Node tracking
// These must be initialized to NULL
__CUDA_DEVICE__
static Node* registry__[QUEUDA_MAX_BLOCKS_PER_GPU][QUEUDA_MAX_WARPS_PER_BLOCK][QUEUDA_MAX_THREADS_PER_WARP][QUEUDA_MAX_SEQUENTIAL_NODES];


// Queue buffering
__CUDA_DEVICE__
static __CUDA_SHARED__ char buffering__[MAX_Q_BUFFERING];
__CUDA_DEVICE__
static int buffer_end__;


class Node 
{
 public:
  
  char* instance_name_;
  int block_;
  int warp_;
  int thread_;
 
  __CUDA_DEVICE__
  Node(Name instance_name) : block_(-1), warp_(-1), thread_(-1)
  {
    // Work around lack of strlen/strdup
    int cur = 0;
    while (instance_name[cur]) {
      cur++;
      assert(cur < 1024);
    }
    char* copy = new char[cur+1]();
    for (int x = 0; x < cur; x++) {
      copy[x] = instance_name[x];
    }
    copy[cur] = 0;
    instance_name_ = copy;
  }
  __CUDA_DEVICE__ inline
  void Bind(const int & b, const int & w, const int& t) {
    for (int x = 0; x < QUEUDA_MAX_SEQUENTIAL_NODES; x++) {
      if (registry__[b][w][t][x] == 0) {
        registry__[b][w][t][x] = this;
        block_ = b;
        warp_ = w;
        thread_ = t;
        //printf("SUCCCESSSS %s_%d_%d_%d: %lu\n", instance_name_, b, w, t, (unsigned long)registry__[b][w][t][x]);
        return;
      }
    }
    //assert(false);
    //printf("WARRNNN %s_%d_%d_%d\n", instance_name_, b, w, t);
  }
  
  __CUDA_DEVICE__ inline
  virtual void Run() {
  }

  template <
    typename... Arguments
  >
  __CUDA_DEVICE__ inline
  void Trace(int level, const char* format, Arguments... args) {

    if (level > TRACE_LEVEL) return;
    // Work around the lack of strcpy in CUDA.
    char msgbuff[1024];
    msgbuff[0] = '%';
    msgbuff[1] = 's';
    msgbuff[2] = '_';
    msgbuff[3] = '%';
    msgbuff[4] = 'd';
    msgbuff[5] = '_';
    msgbuff[6] = '%';
    msgbuff[7] = 'd';
    msgbuff[8] = '_';
    msgbuff[9] = '%';
    msgbuff[10] = 'd';
    msgbuff[11] = ':';
    msgbuff[12] = ' ';
    int cur = 0;
    while (format[cur]) {
      assert(cur+13 < 1024);
      msgbuff[cur+13] = format[cur];
      cur++;
    }
    msgbuff[cur+13] = '\n';
    msgbuff[cur+14] = 0;
    printf(msgbuff, instance_name_, block_, warp_, thread_, args...);
  }
};


template <
  typename T,
  int N
>
class QI;


template <
  typename T,
  int N
>
class QO {

  __CUDA_CALLABLE__ inline
  bool PrimIsFull() {
    for (int x = 0; x < num_connections_; x++) {
      if (connections_[x]->queue_.full()) return true;
    }
    return false;
  }
 public:

  Node* producer_ = NULL;
  QI<T, N>* connections_[MAX_Q_FANOUT];
  int num_connections_ = 0;
  
  __CUDA_CALLABLE__ inline
  QO<T, N>(Node* prod) : producer_(prod), num_connections_(0) {
    for (int x = 0; x < MAX_Q_FANOUT; x++) {
      connections_[x] = NULL;
    }
  }
  
  __CUDA_CALLABLE__ inline
  QO<T, N>() : QO<T, N>(NULL) {}

  ~QO<T, N>() = default;
  
  __CUDA_CALLABLE__ inline
  bool IsFull() {
    return PrimIsFull();
  }

  __CUDA_CALLABLE__ inline
  void Push(const T& value) {
#ifndef __CUDACC__
    assert(!this->IsFull());
#else
    while (PrimIsFull()) {}
#endif
    // Note: if we have no connections, purposely do nothing.
    for (int x = 0; x < num_connections_; x++) {
      connections_[x]->queue_.push_back(value);
    }
  }
  
  __CUDA_CALLABLE__ inline
  void Finish(const Tag& t = 1) {
#ifndef __CUDACC__
    assert(!this->IsFull());
#else
    while (PrimIsFull()) {}
#endif
    for (int x = 0; x < num_connections_; x++) {
      connections_[x]->queue_.finish(t);
    }
  }
  
};

template <
  typename T
>
class SPSCQueue {

 public:
  Tagged<T>* data_;
  int size_; // Don't make this unsigned unless you know what you are doing.

#ifdef __CUDACC__
  cuda::atomic<int, cuda::thread_scope_block> head_ = 0;
  cuda::atomic<int, cuda::thread_scope_block> tail_ = 0;
#else
  std::atomic<int> head_ = 0;
  std::atomic<int> tail_ = 0;
#endif

  int head_cache_ = 0;
  int tail_cache_ = 0;
 

 // TODO: parallelize and use sh_mem
 // TODO: const correctness

  __CUDA_DEVICE__ inline
  int ModIncr(const int& x) {
    if (x+1 == size_) {
      return 0;
    }
    return x+1;
  }

  __CUDA_DEVICE__ inline
  SPSCQueue<T>(int size = DEFAULT_Q_SIZE) : size_(size) {
    data_ = reinterpret_cast<Tagged<T>*>(&buffering__[buffer_end__]);
    buffer_end__ += size_ * sizeof(Tagged<T>);
    assert(buffer_end__ < MAX_Q_BUFFERING);
    for (int x = 0; x < size_; x++) {
      data_[x] = Tagged<T>();
    }
  }

  __CUDA_DEVICE__ inline
  bool full() {
    if (ModIncr(tail_) != head_cache_) {
      return false;
    }
    else
    {
#ifdef __CUDACC__
      head_cache_ = head_.load(cuda::memory_order_acquire);
#else
      head_cache_ = head_.load(std::memory_order_acquire);
#endif
    }
    return ModIncr(tail_) == head_cache_;
  }

  __CUDA_DEVICE__ inline
  void push_back(const T& val) {
    assert(!full());
    data_[tail_] = Tagged<T>(val);
#ifdef __CUDACC__
    tail_.store(ModIncr(tail_), cuda::memory_order_release);
#else
    tail_.store(ModIncr(tail_), std::memory_order_release);
#endif
  }
  
  __CUDA_DEVICE__ inline
  void finish(const Tag& t = 1) {
    assert(!full());
    data_[tail_] = Tagged<T>(t);
#ifdef __CUDACC__
    tail_.store(ModIncr(tail_), cuda::memory_order_release);
#else
    tail_.store(ModIncr(tail_), std::memory_order_release);
#endif
  }
  
  __CUDA_DEVICE__ inline
  bool is_done() {
    return data_[head_].is_tag_;
  }

  __CUDA_DEVICE__ inline
  bool empty() {
    if (head_ != tail_cache_) {
      return false;
    }
    else
    {
#ifdef __CUDACC__
      tail_cache_ = tail_.load(cuda::memory_order_acquire);
#else
      tail_cache_ = tail_.load(std::memory_order_acquire);
#endif
    }
    return head_ == tail_cache_;
  }
  
  __CUDA_DEVICE__ inline
  T& front() {
    assert(!empty());
    assert(!data_[head_].is_tag_);
    return data_[head_].value_;
  }
  
  __CUDA_DEVICE__ inline
  T pop_front() {
    assert(!empty());
    assert(!data_[head_].is_tag_);
    T tmp = data_[head_].value_;
#ifdef __CUDACC__
    head_.store(ModIncr(head_), cuda::memory_order_release);
#else
    head_.store(ModIncr(head_), std::memory_order_release);
#endif
    return tmp;
  }
  
  __CUDA_DEVICE__ inline
  Tag resume() {
    assert(!empty());
    assert(data_[head_].is_tag_);
    Tag tmp = data_[head_].tag_;
#ifdef __CUDACC__
    head_.store(ModIncr(head_), cuda::memory_order_release);
#else
    head_.store(ModIncr(head_), std::memory_order_release);
#endif
    return tmp;
  }
};



template<
  typename T,
  int N
>
using Q = QO<T, N>;

template <
  typename T,
  int N
>
class QI {

  __CUDA_CALLABLE__ inline
  bool PrimIsEmpty() {
    return queue_.empty();
  }

 public:

  Node* producer_ = NULL;
  SPSCQueue<T> queue_;
  
  QI<T, N>() = default;
  ~QI<T, N>() = default;

  __CUDA_CALLABLE__ inline
  QI<T, N>(QO<T, N>* qo, int size = DEFAULT_Q_SIZE) : queue_(size) {
    assert(qo->producer_ != NULL);
    producer_ = qo->producer_;
    assert(qo->num_connections_ <= MAX_Q_FANOUT);
    qo->connections_[qo->num_connections_] = this;
    qo->num_connections_++;
  }
  
  __CUDA_CALLABLE__ inline
  bool IsEmpty() {
    return queue_.empty();
  }

  __CUDA_CALLABLE__ inline
  bool IsDone() {
#ifndef __CUDACC__
    assert(!this->PrimIsEmpty());
#else
    while (this->PrimIsEmpty()) {}
#endif
    return queue_.is_done();
  }
  
  __CUDA_CALLABLE__ inline
  T& Peek() {
#ifndef __CUDACC__
    assert(!this->PrimIsEmpty());
#else
    while (this->PrimIsEmpty()) {}
#endif
    return queue_.front();
  }

  __CUDA_CALLABLE__ inline
  T Pop() {
#ifndef __CUDACC__
    assert(!this->PrimIsEmpty());
#else
    while (this->PrimIsEmpty()) {}
#endif
    auto tmp = this->Peek();
    queue_.pop_front();
    return tmp;
  }
  
  __CUDA_CALLABLE__ inline
  Tag Resume() {
    assert(this->IsDone());
    return queue_.resume();
  }

  __CUDA_CALLABLE__ inline
  void DrainUntilDone() {
    while (!this->IsDone()) {
      this->Pop();
    }
  }
  
};

template <typename T>
__CUDA_CALLABLE__ inline
bool AnyIsDone(T* arr, int size) {
    bool done=false;
    for(int i=0; i< size; i++){
        done |= arr[i]->IsDone();
    } 
    return done;
};

template <typename T>
__CUDA_CALLABLE__ inline
bool AllIsDone(T* arr, int size) {
    bool done=true;
    for(int i=0; i< size; i++){
        done &= arr[i]->IsDone();
    } 
    return done;
};

template <typename T>
__CUDA_CALLABLE__ inline
Tag AllResume(T* arr, int size) {
    if (size == 0) return 0;
    Tag last = arr[0]->Resume();
    for(int i=1; i< size; i++){
        Tag t = arr[i]->Resume();
        assert(t == last);
        last = t;
    }
    return last;
};

template <typename T>
__CUDA_CALLABLE__ inline
void AllFinish(T* arr, int size, const Tag& t = 1) {
    for(int i=0; i< size; i++){
        arr[i]->Finish(t);
    } 
};

#ifdef __CUDACC__


__device__ inline
void Build(BuilderFunction build) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int s = 0; s < QUEUDA_MAX_SEQUENTIAL_NODES; s++) {
      for (int b = 0; b < QUEUDA_MAX_BLOCKS_PER_GPU; b++) {
        for (int w = 0; w < QUEUDA_MAX_WARPS_PER_BLOCK; w++) {
          for (int t = 0; t < QUEUDA_MAX_THREADS_PER_WARP; t++) {
            registry__[b][w][t][s] = NULL;
          }
        }
      }
    }
    buffer_end__ = 0;
    printf("Starting build.\n");

    if (build)
      build();

    printf("Build Complete!");
  }

    // Everyone waits until build completes
    __syncthreads();
}

__device__ inline
void Run() {
 
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("=============================\nBeginning Run\n=============================\n");
  }
  
  // TODO: THESE BOUNDS SHOULD BE DYNAMIC
  int my_bid = blockIdx.x;
  assert(my_bid < QUEUDA_MAX_BLOCKS_PER_GPU);
  int my_tid = threadIdx.x % QUEUDA_MAX_THREADS_PER_WARP;
  int my_wid = threadIdx.x / QUEUDA_MAX_THREADS_PER_WARP;
  
  for (int s = 0; s < QUEUDA_MAX_SEQUENTIAL_NODES; s++) {
    if (registry__[my_bid][my_wid][my_tid][s] != NULL) {
      printf("=============================\nRunning Node %s_%d_%d_%d (%d)\n=============================\n", registry__[my_bid][my_wid][my_tid][s]->instance_name_, my_bid, my_wid, my_tid, s);
      registry__[my_bid][my_wid][my_tid][s]->Run();
    }
  }
}

#else

inline
void Build(BuilderFunction build) {
  for (int s = 0; s < QUEUDA_MAX_SEQUENTIAL_NODES; s++) {
    for (int b = 0; b < QUEUDA_MAX_BLOCKS_PER_GPU; b++) {
      for (int w = 0; w < QUEUDA_MAX_WARPS_PER_BLOCK; w++) {
        for (int t = 0; t < QUEUDA_MAX_THREADS_PER_WARP; t++) {
          registry__[b][w][t][s] = NULL;
        }
      }
    }
  }
  buffer_end__ = 0;
  
  printf("Starting build.\n");

  build();
  
  printf("Build Complete!\n");
  
}

inline
void Run() {
  printf("=============================\nBeginning Run\n=============================\n");
  for (int s = 0; s < QUEUDA_MAX_SEQUENTIAL_NODES; s++) {
    for (int b = 0; b < QUEUDA_MAX_BLOCKS_PER_GPU; b++) {
      for (int w = 0; w < QUEUDA_MAX_WARPS_PER_BLOCK; w++) {
        for (int t = 0; t < QUEUDA_MAX_THREADS_PER_WARP; t++) {
          if (registry__[b][w][t][s] != NULL) {
            printf("=============================\nRunning Node %d %d %d %d: %s\n=============================\n", b, w, t, s, registry__[b][w][t][s]->instance_name_);
            registry__[b][w][t][s]->Run();
          }
        }
      }
    }
  }
}



#endif

}  // namespace: queueda
