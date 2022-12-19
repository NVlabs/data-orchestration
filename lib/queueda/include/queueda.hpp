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

#include "device-utils.hpp"

namespace queueda {

// Type defs

// TODO: something more useful here, with instance numbering.
using Name = const char[];


// Node tracking
// Work around inability to call constructor in CUDA global memory;
__CUDA_DEVICE__
static char registry__[options::kMaxBlocksPerGPU][options::kMaxWarpsPerBlock][options::kMaxThreadsPerWarp][options::kMaxSequentialNodes][sizeof(NodeFunction)];

__CUDA_DEVICE__
NodeFunction* GetNode(int b, int w, int t, int s) {
  auto p = reinterpret_cast<NodeFunction*>(&registry__[b][w][t][s][0]);
  return p;
}

__CUDA_DEVICE__
void SetNode(int b, int w, int t, int s, NodeFunction f) {
  auto p = reinterpret_cast<NodeFunction*>(&registry__[b][w][t][s][0]);
  *p = f;
}


// Queue buffering
__CUDA_DEVICE__
static __CUDA_SHARED__ char buffering__[options::kMaxQBuffering];
__CUDA_DEVICE__
static int buffer_end__[options::kMaxBlocksPerGPU];



__CUDA_DEVICE__ inline
void Bind(const int & b, const int & w, const int& t, NodeFunction f) {
  
  for (int x = 0; x < options::kMaxSequentialNodes; x++) {
    auto node = GetNode(b, w, t, x);
    // Check if it holds a function, not for NULL pointer.
    if (!(*node)) {
      SetNode(b, w, t, x, f);
      return;
    }
  }
  //assert(false); TEMPORARY: CAUSING WARNING. NEEDS REPLACEMENT.
}

template <
  typename... Arguments
>
__CUDA_DEVICE__ inline
void Trace(int level, const char* format, Arguments... args) {

  if (level > options::device_->kCurrentLibraryTraceLevel) {
    //printf("%d\n", options::device_->kCurrentLibraryTraceLevel);
    return;
  }
  
  // Work around the lack of strcpy in CUDA.
  char msgbuff[options::kMsgBufferSize];
  size_t mcur = 0; 
  msgbuff[mcur++] = 'N';
  msgbuff[mcur++] = 'o';
  msgbuff[mcur++] = 'd';
  msgbuff[mcur++] = 'e';
  msgbuff[mcur++] = '_';
  msgbuff[mcur++] = '%';
  msgbuff[mcur++] = 'l';
  msgbuff[mcur++] = 'l';
  msgbuff[mcur++] = 'd';
  msgbuff[mcur++] = '_';
  msgbuff[mcur++] = '%';
  msgbuff[mcur++] = 'l';
  msgbuff[mcur++] = 'l';
  msgbuff[mcur++] = 'd';
  msgbuff[mcur++] = '_';
  msgbuff[mcur++] = '%';
  msgbuff[mcur++] = 'l';
  msgbuff[mcur++] = 'l';
  msgbuff[mcur++] = 'd';
  msgbuff[mcur++] = ':';
  msgbuff[mcur++] = ' ';

  int cur = 0;
  while (format[cur]) {
    msgbuff[mcur++] = format[cur++];
  }

  msgbuff[mcur++] = '\n';
  msgbuff[mcur++] = 0;

  printf(msgbuff, GetBlock(), GetWarp(), GetThread(), args...);
  return;
}

class Node {
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

  QI<T, N>* connections_[options::kMaxQFanout];
  int num_connections_ = 0;
  
  __CUDA_CALLABLE__ inline
  QO<T, N>() : num_connections_(0) {
    for (int x = 0; x < options::kMaxQFanout; x++) {
      connections_[x] = NULL;
    }
  }
  
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
  SPSCQueue<T>(int size = options::kDefaultQSize) : size_(size) {
    data_ = reinterpret_cast<Tagged<T>*>(&buffering__[buffer_end__[GetBlock()]]);
    buffer_end__[GetBlock()] += size_ * sizeof(Tagged<T>);
    assert(buffer_end__[GetBlock()] < options::kMaxQBuffering);
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

  SPSCQueue<T> queue_;
  
  QI<T, N>() = default;
  ~QI<T, N>() = default;

  __CUDA_CALLABLE__ inline
  QI<T, N>(QO<T, N>* qo, int size = options::kDefaultQSize) : queue_(size) {
    assert(qo->num_connections_ <= options::kMaxQFanout);
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


__CUDA_DEVICE__ inline
void Build(BuilderFunction build) {

  // One thread from every block does this.
  if (GetWarp() == 0 & GetThread() == 0) {
    for (int s = 0; s < options::kMaxSequentialNodes; s++) {
      for (int w = 0; w < options::kMaxWarpsPerBlock; w++) {
        for (int t = 0; t < options::kMaxThreadsPerWarp; t++) {
          // Use C++ "placement" new operator to control memory placement. 
          new (&registry__[GetBlock()][w][t][s]) NodeFunction();
        }
      }
    }
    
    //printf("Block %lld Starting build.\n", GetBlock());
    buffer_end__[GetBlock()] = 0;

    if (build)
      build();

    //printf("Block %lld build Complete!\n", GetBlock());

  }
}


__device__ inline
void Run() {
 
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("=============================\nBeginning Run\n=============================\n");
  }
  
  int my_bid = blockIdx.x;
  assert(my_bid < options::kMaxBlocksPerGPU);
  int my_tid = threadIdx.x % options::kMaxThreadsPerWarp;
  int my_wid = threadIdx.x / options::kMaxThreadsPerWarp;
  
  for (int s = 0; s < options::kMaxSequentialNodes; s++) {
    auto node = GetNode(my_bid, my_wid, my_tid, s);
    // Check if it holds a function, not for NULL pointer
    if (*node) {
      Trace(0, "======= Beginning (%d) =======", s);
      (*node)();
    }
  }
}

#else

inline
void Build(BuilderFunction build) {

  for (int s = 0; s < options::kMaxSequentialNodes; s++) {
    for (int b = 0; b < options::kMaxBlocksPerGPU; b++) {
      buffer_end__[b] = 0;
      for (int w = 0; w < options::kMaxWarpsPerBlock; w++) {
        for (int t = 0; t < options::kMaxThreadsPerWarp; t++) {
          // Use C++ "placement" new operator to control memory placement. 
          new (&registry__[b][w][t][s]) NodeFunction();
        }
      }
    }
  }
  
  for (int b = 0; b < options::host_->kActiveBlocksPerGPU; b++) {
    current_block__ = b;
    //printf("Block %lld Starting build.\n", GetBlock());
    if (build) {
      build();
    }
    //printf("Block %lld build Complete!\n", GetBlock());
  }


}

inline
void Run() {
  printf("=============================\nBeginning Run\n=============================\n");
  for (int s = 0; s < options::kMaxSequentialNodes; s++) {
    for (int b = 0; b < options::kMaxBlocksPerGPU; b++) {
      current_block__ = b;
      for (int w = 0; w < options::kMaxWarpsPerBlock; w++) {
        current_warp__ = w;
        for (int t = 0; t < options::kMaxThreadsPerWarp; t++) {
          current_thread__ = t;
          auto node = GetNode(b, w, t, s);
          // Check if it holds a function, not for NULL pointer.
          if (*node) {
            Trace(2, "======= Beginning (%d) =======", s);
            (*node)();
          }
        }
      }
    }
  }
}



#endif

void Init(int B, int W, int T)
{
  AddOptions();
  
  options::host_->kActiveBlocksPerGPU = B;
  options::host_->kActiveWarpsPerBlock = W;
  options::host_->kActiveThreadsPerWarp = T;

  ReadOptions();
}

}  // namespace: queueda
