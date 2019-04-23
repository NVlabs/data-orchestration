/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
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

#ifndef SWOOP_HPP_
#define SWOOP_HPP_


#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include <assert.h>

#include "strace.hpp"

namespace swoop
{

using Index = uint64_t;
using Value = Index;
using Coordinate = Index;
using Size = Index;
using Order = std::map<Index, Index>;

//class Tensor;
template <typename T>
class DomainIterator;
template <typename T>
class ScanResult;
  
enum class PreDefinedOrder
{
 InnerToOuter,
 OuterToInner,
 DiagonalInnerToOuter,
 DiagonalOuterToInner,
 Permutation,
 SkewedPermutation,
 Custom
};


class Point
{
 public:
  std::vector<Coordinate> coordinates_;
  
  Point() = default;
  
  explicit Point(Size size) : coordinates_(size)
  {
    
  }

  Point(const std::vector<Coordinate>& val) : coordinates_(val)
  {
    
  }

  Point(const std::initializer_list<Coordinate>& val) : coordinates_(std::rbegin(val), std::rend(val))
  {
    
  }
  
  Coordinate Flatten(const Point& dim_sizes) const
  {
    uint64_t res = 0;
    uint64_t amplification_factor = 1;
    assert(GetSize() == dim_sizes.GetSize());
    auto size_it = dim_sizes.coordinates_.rbegin();
    for (auto rit = coordinates_.rbegin(); rit != coordinates_.rend(); rit++)
    {
      res += (*rit) * amplification_factor;
      amplification_factor *= (*size_it);
      size_it++;
    }
    return res;
  }

  Coordinate FlattenSizes() const
  {
    Coordinate res = 1;
    for (auto it = coordinates_.begin(); it != coordinates_.end(); it++)
    {
      res *= (*it);
    }
    return res;
  }

  bool Bounded1DTranslate(Coordinate amount, Index dim, Point bound)
  {
    coordinates_[dim] += amount;
    if (coordinates_[dim] >= bound[dim])
    {
      coordinates_[dim] = 0;
      return true;
    }
    return false;
  }
  
  Point TranslateAll(const Coordinate& factor) const
  {
    Point result(GetSize());
    for (int x = 0; x < GetSize(); x++)
    {
      result[x] = coordinates_[x] + factor;
    }
    return result;
  }

  Point Translate(const Point& factor) const
  {
    assert(factor.GetSize() == GetSize());
    Point result(GetSize());
    for (int x = 0; x < factor.GetSize(); x++)
    {
      result[x] = coordinates_[x] + factor[x];
    }
    return result;
  }
  
  Point Scale(const Point& factor) const
  {
    assert(factor.GetSize() == GetSize());
    Point result(GetSize());
    for (int x = 0; x < factor.GetSize(); x++)
    {
      result[x] = coordinates_[x] * factor[x];
    }
    return result;
  }
  
  bool IsMultipleOf(const Point& other) const
  {
    assert(other.GetSize() == GetSize());
    for (int x = 0; x < other.GetSize(); x++)
    {
      if ((coordinates_[x] % other[x]) != 0) return false;
    }
    return true;
  }
  
  Point ShiftDimensionsLeft(Coordinate new_val)
  {
    Point result(GetSize());
    result[0] = new_val;
    for (int x = 1; x < GetSize(); x++)
    {
      result[x] = coordinates_[x-1];
    }
    return result;
  }
  
  void AllTo(const Coordinate& coord)
  {
    for (int x = 0; x < coordinates_.size(); x++)
    {
      coordinates_[x] = coord;
    }
  }

  Point Permute(const Order& order) const
  {
    assert(order.size() == GetSize());

    Point result(GetSize());
    for (Index x = 0; x < GetSize(); x++)
    {
      Index new_x = order.at(x);
      result[new_x] = coordinates_[x];
    }
    return result;
  }

  Size GetSize() const
  {
    return coordinates_.size();
  }
  
  bool operator <=(const Point& other)
  {
    assert(other.GetSize() == GetSize());
    for (int x = 0; x < other.GetSize(); x++)
    {
      if (coordinates_[x] > other[x]) return false;
    }
    return true;
  }

  bool operator >=(const Point& other)
  {
    assert(other.GetSize() == GetSize());
    for (int x = 0; x < other.GetSize(); x++)
    {
      if (coordinates_[x] < other[x]) return false;
    }
    return true;
  }

  bool operator >(const Point& other)
  {
    assert(other.GetSize() == GetSize());
    for (int x = 0; x < other.GetSize(); x++)
    {
      if (coordinates_[x] <= other[x]) return false;
    }
    return true;
  }

  bool operator ==(const Point& other)
  {
    assert(other.GetSize() == GetSize());
    for (int x = 0; x < other.GetSize(); x++)
    {
      if (coordinates_[x] != other[x]) return false;
    }
    return true;
  }
  
  Index operator [] (const Index& idx) const { return coordinates_[idx]; }
  Index& operator [] (const Index& idx) { return coordinates_[idx]; }
  
  std::string Show() const
  {
    std::stringstream res;
    res << "{";
    for (int x = GetSize() - 1; x >= 0; x--)
    {
      res << coordinates_[x];
      if (x != 0)
      {
        res << ", ";
      }
    }
    res << "}";
    return res.str();
  }
  
  Point DropDimension(Index dim) const
  {
    Point result(GetSize() - 1);
    int cur = 0;
    for (int x = 0; x < GetSize(); x++)
    {
      if (x != dim)
      {
        result[cur] = coordinates_[x];
        cur++;
      }
    }
    return result;
  }
};


template <typename T>
class OptimizedPoint
{
 public:
  T* parent_;
  Index index_;
  OptimizedPoint(T* parent, Index idx) : parent_(parent), index_(idx)
  {
  }
  
  Index GetIndex() const { return index_; }
  // TODO: Define math operators that lower, like:
  // Point operator +(const Point& other);
  // Point operator +(const OptimizedPoint& other);
  
  std::string Show()
  {
    return parent_->Reconstruct(*this).Show();
  }
};

using Counts = Point;
using Strides = Point;
/*
class Tensor
{
 public:
  //virtual ScanResult Scan(const Point& bases, const Counts& counts, const Strides& strides) = 0;
  //virtual ScanResult Scan(const Point& bases, const Counts& counts, const Order& order, const Strides& strides) = 0;

  virtual Value operator [](const OptimizedPoint& p) const = 0;
  virtual Value& operator [](const OptimizedPoint& p) = 0;

  virtual bool Has(const OptimizedPoint& p) const = 0;
  
  virtual Value operator [](const Point& p) const = 0;
  virtual Value& operator [](const Point& p) = 0;
  
  virtual bool Has(const Point& p) const = 0;
  
  virtual bool IsConcordant(const Order& order) const = 0;
  
  // Domain iteration methods
  template <typename T>
  DomainIterator<T> GetDomainIterator(const Point& base, const Point& counts, const Point& strides, const Order& order);
  bool NextConcordant(OptimizedPoint& idx);
  OptimizedPoint GetConcordant();
  bool NextDiscordant(OptimizedPoint& idx);
  Point GetDiscordant();
};
*/

class DenseTensor : public Traceable
{
 protected: 
  std::vector<Value> vals_;
  Point dim_sizes_;
  
  // Iteration scoreboarding
  Index bound_;
  Point ordered_count_;
  Point ordered_bound_;
  Point ordered_stride_;
  Point ordered_offset_;

  Point UnflattenIndex(const Index& x)
  {
    Index cur = x;
    Index num_dims = dim_sizes_.GetSize();
    Point res(num_dims);
    for (auto k = 0; k < num_dims; k++)
    {
      res[k] = cur % dim_sizes_[k];
      cur = cur / dim_sizes_[k];
    }
    return res;
  }

  void PrimInitialize(const Index& start_idx, 
                      const Index& end_idx,
                      Value (*func)(const Point& idxs))
  {
    assert(start_idx <= end_idx);
    for (auto x = start_idx; x < end_idx; x++)
    {
      auto idxs = UnflattenIndex(x);
      vals_[x] = func(idxs);
    }
  }
  
 public:
  DenseTensor(const Point& dim_sizes, Value (*init_func)(const Point&), const std::string& nm = "") :
    Traceable(nm),
    vals_(dim_sizes.FlattenSizes()),
    dim_sizes_(dim_sizes),
    bound_(0),
    ordered_count_(dim_sizes.GetSize()),
    ordered_bound_(dim_sizes.GetSize()),
    ordered_stride_(dim_sizes.GetSize()),
    ordered_offset_(dim_sizes.GetSize())
  {
    PrimInitialize(0, vals_.size(), init_func);
  }

  ScanResult<DenseTensor> Scan(const Point& bases, const Counts& counts, const Strides& strides);
  
  ScanResult<DenseTensor> Scan(const Point& bases, const Counts& counts, const Order& order, const Strides& strides);

  Value operator [](const OptimizedPoint<DenseTensor>& p) const
  {
    return vals_[p.GetIndex()];
  }

  Value& operator [](const OptimizedPoint<DenseTensor>& p)
  {
    return vals_[p.GetIndex()];
  }
  
  bool Has(const OptimizedPoint<DenseTensor>& p) const
  {
    // Can assert in bounds here...
    return true;
  }
  
  bool IsConcordant(const Order& order, const Strides& strides) const
  {
    // Concordant iterator order is out->in.
    assert(order.size() == dim_sizes_.GetSize());
    int cur = dim_sizes_.GetSize() - 1;
    for (int x = 0; x < dim_sizes_.GetSize(); x++)
    {
      if (order.at(x) != cur) return false;
      cur--;
    }
    // All strides must be 1.
    assert(strides.GetSize() == dim_sizes_.GetSize());
    for (int x = 0; x < dim_sizes_.GetSize(); x++)
    {
      if (strides[x] != 1) return false;
    }
    return true;

    return true;
  }

  Value operator [](const Point& p) const
  {
    assert(Has(p));
    return vals_[p.Flatten(dim_sizes_)];
  }

  Value& operator [](const Point& p)
  {
    assert(Has(p));
    return vals_[p.Flatten(dim_sizes_)];
  }
  
  bool Has(const Point& p) const
  {
    assert(p.GetSize() == dim_sizes_.GetSize());
    // Do a bounds check. 
    for (int x = 0; x < dim_sizes_.GetSize(); x++)
    {
      if (p[x] >= dim_sizes_[x]) return false;
    }
    return true;
  }

  DomainIterator<DenseTensor> GetDomainIterator(const Point& base, const Counts& counts, const Point& stride, const Order& order);
  
  bool NextConcordant(OptimizedPoint<DenseTensor>& cur)
  {
    cur.index_++;
    return cur.index_ == bound_;
  }

  bool NextDiscordant(OptimizedPoint<DenseTensor>& cur)
  {
    if (++ordered_count_[0] < ordered_bound_[0])
    {
      cur.index_ += ordered_stride_[0] * ordered_offset_[0];
      // TODO: Support skew.
      //if (skewed) 
      //  for m = 0..N
      //    if (ordered_is_skewed[m])
      //    cur.index_ += cur_inner * ordered_offset[m]
      return false;
    }
    else
    {
      cur.index_ = 0;
      ordered_count_[0] = 0;
      for (int x = 1; x < dim_sizes_.GetSize(); x++)
      {
        if (++ordered_count_[x] < ordered_bound_[x])
        {
          cur.index_ += ordered_count_[x] * ordered_stride_[x] * ordered_offset_[x];
          return false;
        }
        else
        {
          ordered_count_[x] = 0;
        }
      }
      // We are done!
      return true;
    }
  }
  
  Point Reconstruct(OptimizedPoint<DenseTensor>& cur)
  {
    return UnflattenIndex(cur.index_);
  }
};


class CompressedKeyDimensionTensor : public Traceable
{
 protected: 
  Point dim_max_sizes_;
  Index key_dim_;
  
  std::vector<Value> vals_; // Size: NNZ
  std::vector<Index> i_; // Size: Key Dim
  std::vector<Point> j_; // Size: NNZ
  
  //Iteration tracking
  Index cur_key_ = 0;
  Index bound_ = 0;
  Coordinate key_stride_ = 0;
  Point rest_bound_;
  Point rest_stride_;

  Point UnflattenIndex(const Index& x)
  {
    Index cur = x;
    Index num_dims = dim_max_sizes_.GetSize();
    Point res(num_dims);
    for (auto k = 0; k < num_dims; k++)
    {
      res[k] = cur % dim_max_sizes_[k];
      cur = cur / dim_max_sizes_[k];
    }
    return res;
  }

  void PrimInitialize(const Index& start_idx, 
                      const Index& end_idx,
                      bool (*domain_func)(const Point&),
                      Value (*func)(const Point&))
  {
    assert(start_idx <= end_idx);
    int cur = 0;
    for (auto x = start_idx; x < end_idx; x++)
    {
      auto p = UnflattenIndex(x);
      if (domain_func(p))
      {
        vals_[cur] = func(p);
        j_[cur] = p.DropDimension(key_dim_);
        i_[p[key_dim_] + 1]++;
        cur++;
      }
    }
    for (auto x = 1; x < dim_max_sizes_[key_dim_] + 1; x++)
    {
      i_[x] += i_[x - 1];
    }
  }
  
  Coordinate FindKey(const Index& idx)
  {
    Index cur_i = 0;
    for (cur_i = 0; cur_i < dim_max_sizes_[key_dim_] + 1; cur_i++)
    {
      if (i_[cur_i] > idx) break;
    }
    return cur_i - 1;
  }

 public:
  CompressedKeyDimensionTensor(const Point& dim_max_sizes, Size domain_size, Index key_dim, bool (*init_domain)(const Point&), Value (*init_func)(const Point&), const std::string& nm = "") :
    Traceable(nm),
    dim_max_sizes_(dim_max_sizes), 
    key_dim_(key_dim),
    vals_(domain_size),
    i_(dim_max_sizes[key_dim] + 1),
    j_(domain_size),
    rest_bound_(dim_max_sizes.GetSize() - 1),
    rest_stride_(dim_max_sizes.GetSize() - 1)
    
  {
    PrimInitialize(0, dim_max_sizes_.FlattenSizes(), init_domain, init_func);
  }

  ScanResult<CompressedKeyDimensionTensor> Scan(const Point& bases, const Counts& counts, const Strides& strides);
  ScanResult<CompressedKeyDimensionTensor> Scan(const Point& bases, const Counts& counts, const Order& order, const Strides& strides);

  Value& operator [](const OptimizedPoint<CompressedKeyDimensionTensor>& p)
  {
    // Can assert p.index_ == cur_j
    return vals_[p.GetIndex()];
  }

  Value operator [](const OptimizedPoint<CompressedKeyDimensionTensor>& p) const
  {
    // Can assert p.index_ == cur_j
    return vals_[p.GetIndex()];
  }

  bool Has(const OptimizedPoint<CompressedKeyDimensionTensor>& p)
  {
    // Can assert in bounds
    //return p.IsValid(); TODO
    return true;
  }

  DomainIterator<CompressedKeyDimensionTensor> GetDomainIterator(const Point& base, const Counts& counts, const Point& stride, const Order& order);

  bool IsConcordant(const Order& order, const Strides& strides) const
  {
    // Concordant iterator order is key is outermost, then for the rest out->in.
    assert(order.size() == dim_max_sizes_.GetSize());
    int cur = dim_max_sizes_.GetSize() - 1;
    if (order.at(cur) != key_dim_) return false;
    if (key_dim_ == cur) cur--;
    for (int x = dim_max_sizes_.GetSize() - 2; x >= 0; x--)
    {
      if (order.at(x) != cur) return false;
      cur--;
      // Skip the key, since we processed it above.
      if (x == key_dim_)
      {
        cur--;
      } 
    }
    // All strides must be 1.
    assert(strides.GetSize() == dim_max_sizes_.GetSize());
    for (int x = 0; x < dim_max_sizes_.GetSize(); x++)
    {
      if (strides[x] != 1) return false;
    }
    return true;
  }

  // Fully optimized version... would have to use FindKey if needed.
  bool NextConcordant(OptimizedPoint<CompressedKeyDimensionTensor>& cur)
  {
    // Advance to the next item.
    cur.index_++;
    return cur.index_ == bound_;
  }
  
  // Version that also tracks cur_i coordinate so no FindKey is needed.
  bool NextConcordantC(OptimizedPoint<CompressedKeyDimensionTensor>& cur)
  {
    cur.index_++;
    if (cur.GetIndex() >= i_[cur_key_ + 1])
    {
      cur_key_++;
    }
    return cur.index_ == bound_;
  }

  // Fully general iteration.
  bool NextDiscordant(OptimizedPoint<CompressedKeyDimensionTensor>& cur)
  {
    // Search for the next point until we return.
    while (1)
    {
      // Advance to the next item.
      cur.index_++;
      // Check to see if we are done.
      if (cur.index_ == bound_) return true;
      Point next = j_[cur.GetIndex()];
      // Advance the key dimension if we have used up all points
      // or gone past the user defined bound for the rest of the coords.
      if (cur.GetIndex() >= i_[cur_key_ + 1] || next > rest_bound_)
      {
        cur_key_ += key_stride_;
        // Reset to the next point (may be the same as current value if key_stride_ == 1)
        cur.index_ = i_[cur_key_];
        // Check to see if we are done.
        if (cur.index_ >= bound_) return true;
        next = j_[cur.GetIndex()];
      }
      // Stride essentially becomes a modulo check.
      if (next.IsMultipleOf(rest_stride_))
      {
        return false;
      }
      // If we get here, try again on the next point.
    }
  }

  Point Reconstruct(OptimizedPoint<CompressedKeyDimensionTensor>& cur)
  {
    Point result(dim_max_sizes_.GetSize());
    int cidx = 0;
    // Use this if using NextConcordant() instead of NextConcordantC()
    //result[key_dim_] = FindKey(cur.GetIndex());
    result[key_dim_] = cur_key_;
    Point rest = j_[cur.GetIndex()];
    for (int x = 0; x < dim_max_sizes_.GetSize(); x++)
    {
      if (x != key_dim_)
      {
        result[x] = rest[cidx];
        cidx++;
      }
    }
    return result;
  }
  
  bool Has(const Point& p)
  {
    Point tgt = p.DropDimension(key_dim_);
    for (int x = i_[p[key_dim_]]; x < i_[p[key_dim_] + 1]; x++)
    {
      Point cur(dim_max_sizes_.GetSize() - 1);
      cur = j_[x];
      if (tgt == cur) return true;
    }
    return false;
  }

  Value& operator [](const Point& p)
  {
    Point tgt = p.DropDimension(key_dim_);
    for (int x = i_[p[key_dim_]]; x < i_[p[key_dim_] + 1]; x++)
    {
      Point cur(dim_max_sizes_.GetSize() - 1);
      cur = j_[x];
      if (tgt == cur) return vals_[x];
    }
    assert(false);
  }

  Value operator [](const Point& p) const
  {
    Point tgt = p.DropDimension(key_dim_);
    for (int x = i_[p[key_dim_]]; x < i_[p[key_dim_] + 1]; x++)
    {
      Point cur(dim_max_sizes_.GetSize() - 1);
      cur = j_[x];
      if (tgt == cur) return vals_[x];
    }
    assert(false);
  }  
  
};


template <typename T>
class DomainIterator
{
 protected:
  T* target_;
  bool (T::*MyNext) (OptimizedPoint<T>&) = NULL;
  OptimizedPoint<T> cur_;
  bool is_done_;
  
  
 public:
  DomainIterator
  (
    T* target,
    Index base,
    bool (T::*Nxt)(OptimizedPoint<T>&)
  ) :
    target_(target),
    MyNext(Nxt),
    cur_(target, base),
    is_done_(false)
  {
  }

  bool operator !=(const DomainIterator& other)
  {
    return !is_done_;
  }
  
  void operator ++()
  {
    is_done_ = (target_->*MyNext)(cur_);
  }
  
  OptimizedPoint<T> operator *()
  {
    return cur_;
  }
  
};


template <typename T>
class ScanResult
{
 protected:
  T* target_;
  Point base_;
  Counts counts_;
  Strides strides_;
  Order order_;

 public:
 
  ScanResult
  (
    T* tgt, 
    const Point& base, 
    const Counts& counts, 
    const Strides& strides,
    const Order& order
  ) :
    target_(tgt),
    base_(base),
    counts_(counts),
    strides_(strides),
    order_(order)
  {
  }
  
  DomainIterator<T> begin()
  {
    return target_->GetDomainIterator(base_, counts_, strides_, order_);
  }
  
  DomainIterator<T> end()
  {
    // This won't really be used for anything. It's just the "other" in !=
    // In the future it might be slightly more efficient to use an empty
    // dummy class for this purpose, if that works...
    DomainIterator<T> it(NULL, 0, NULL);
    return it;
  }
  /*
  ScanResult operator &&(const ScanResult& other)
  {
    IntersectionScanResult sc(this, other);
    return sc;
  }
  
  operator ||(const ScanResult& other)
  {
    UnionScanResult sc(this, other);
    return sc;
  }*/
};



//Tracer& T(int l = 0);


}  // end namespace: swoop

#endif /* SWOOP_HPP_ */

