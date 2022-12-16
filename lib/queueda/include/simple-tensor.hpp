/* Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include <queueda.hpp>

namespace queueda {


// Some simple typedefs and data types.
using Coordinate = int;
using Point = std::vector<Coordinate>;
using Handle = int;
using Value = int;


inline
void PrintPoint(const Point& p) {

  printf("(");
  for (auto it = p.begin(); it != p.end(); it++) {
    printf("% 4d", *it);
    if (std::next(it) != p.end()) {
      printf(", ");
    }
  }
  printf(")");
}

class SimpleTensor {
 
 public:
  char* name_;
  Point shape_;
  Coordinate size_;
  Value* v_;
  
  SimpleTensor(const Point& shape, Name name) :
    name_(strdup(name)),
    shape_(shape) {
    
    size_ = 1;
    for (auto dim : shape_) {
      size_ *= dim;
    }
    
    v_ = new Value[size_]();
    
    for (int x = 0; x < size_; x++) {
      v_[x] = 0;
    }
  }

  SimpleTensor(const Point& shape, Name name, Tuple<Value, Value> rand_range)
    : SimpleTensor(shape, name) {
    
    auto& [min, max] = rand_range;
    Value diff = max - min;
    
    for (int x = 0; x < size_; x++) {
      v_[x] = (rand() % diff) + min;
    }
  }
  
  inline
  Value* CopyArrayToDevice() {
    auto dv = AllocArrayOnDevice<Value>(size_);
    SetDeviceArray<Value>(dv, v_, size_);
    return dv;
  }
  
  inline
  void CopyArrayFromDevice(Value* dv) {
    SetHostArray<Value>(v_, dv, size_);
  }
  
  
  inline
  Coordinate Flatten(const Point& p) {
    assert(p.size() == shape_.size());
    Coordinate res = 0;
    for (int x = 0; x < shape_.size() - 1; x++)  {
      int factor = 1;
      for (int y = x; y < shape_.size(); y++) {
        factor *= shape_[y];
      }
      res += p[x] * factor;
    }
    return res;
  }
  
  inline
  Point ModuloIncr(const Point& p) {
    assert(p.size() == shape_.size());
    Point res = p;
    std::vector<bool> need_incr(shape_.size(), false);
    need_incr[shape_.size() - 1] = true;
    for (int x = shape_.size() - 1; x >= 0; x--)  {
      if (need_incr[x]) {
        res[x] += 1;
        if (res[x] == shape_[x]) {
          res[x] = 0;
          if (x > 0) {
            need_incr[x-1] = true;
          }
        }
      }
    }
    return res;
  }


  inline
  void Print() {
    printf("\n");
    printf("       %s\n", name_);
    printf("=================\n");
    Point cur(shape_.size(), 0);
    for (int c = 0; c < size_; c++) {
      PrintPoint(cur);
      printf(": %d\n", v_[c]);
      cur = ModuloIncr(cur);
    }
    printf("=================\n\n");
  }
  
  inline
  int CheckMismatches(SimpleTensor* t2) {
    
    int num_mismatches = 0;
    Point cur(shape_.size(), 0);
    
    for (int c = 0; c < size_; c++) {
      if (v_[c] != t2->v_[c]) {
        printf("WARNING: Mismatch at ");
        PrintPoint(cur);
        printf(" - %s: %d, %s: %d\n", name_, v_[c], t2->name_, t2->v_[c]);
        num_mismatches++;
      }
      cur = ModuloIncr(cur);
    }
   
    return num_mismatches;
  }
};

}  // namespace queueda
