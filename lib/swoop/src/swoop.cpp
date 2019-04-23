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


#include "strace.hpp"
#include "swoop.hpp"

namespace swoop
{

int num_warnings_ = 0;

Order InnerToOuter(Size sz)
{
  Order result;
  for (int x = 0; x < sz; x++)
  {
    result[x] = x;
  }
  return result;
}


DomainIterator<DenseTensor> DenseTensor::GetDomainIterator(const Point& base, const Counts& counts, const Point& stride, const Order& order)
{
  if (IsConcordant(order, stride))
  {
    T(0) << "Beginning concordant iteration!" << EndT;
    Point offset = dim_sizes_.ShiftDimensionsLeft(1);
    bound_ = counts.Scale(offset).FlattenSizes();
    DomainIterator<DenseTensor> it(this, base.Flatten(dim_sizes_), &DenseTensor::NextConcordant);
    return it;
  }
  else
  {
    T(0) << "Beginning discordant iteration!" << EndT;
    ordered_count_.AllTo(0);
    ordered_bound_ = counts.Permute(order);
    ordered_stride_ = stride.Permute(order);
    Point offset = dim_sizes_.ShiftDimensionsLeft(1);
    ordered_offset_ = offset.Permute(order); 
    DomainIterator<DenseTensor> it(this, base.Flatten(dim_sizes_), &DenseTensor::NextDiscordant);
    return it;
  }
}

DomainIterator<CompressedKeyDimensionTensor> CompressedKeyDimensionTensor::GetDomainIterator(const Point& base, const Counts& counts, const Point& stride, const Order& order)
{
  cur_key_ = base[key_dim_];
  Point target = base.DropDimension(key_dim_);
  // Find the beginning index.
  Index beg_idx = 0;
  for (beg_idx = i_[cur_key_]; beg_idx < i_[cur_key_ + 1]; beg_idx++)
  {
    if (j_[beg_idx] >= target) break;
  }
  // Find the bound index.
  Point bound = base.Translate(counts.TranslateAll(-1).Scale(stride));
  // Make sure the bound point is actually legal.
  if (bound[key_dim_] >= i_.size())
  {
    // Just stop at the final coordinate.
    // TODO: This is a bit conservative. Could pre-find a tighter bound
    // within the last coordinate in the key_dim.
    bound_ = j_.size();
  }
  else
  {
    Point bound_target = bound.DropDimension(key_dim_);
    for (bound_ = i_[bound[key_dim_]]; bound_ < i_[bound[key_dim_] + 1]; bound_++)
    {
      if (j_[bound_] > bound_target) break;
    }
  }
  if (IsConcordant(order, stride))
  {
    T(0) << "Beginning concordant iteration!" << EndT;
    DomainIterator<CompressedKeyDimensionTensor> it(this, beg_idx, &CompressedKeyDimensionTensor::NextConcordantC);
    return it;
  }
  else
  {
    T(0) << "Beginning discordant iteration!" << EndT;
    rest_bound_ = bound.DropDimension(key_dim_);
    key_stride_ = stride[key_dim_];
    rest_stride_ = stride.DropDimension(key_dim_);
    DomainIterator<CompressedKeyDimensionTensor> it(this, beg_idx, &CompressedKeyDimensionTensor::NextDiscordant);
    return it;
  }
}

ScanResult<DenseTensor> DenseTensor::Scan(const Point& bases, const Counts& counts, const Strides& strides)
{
  ScanResult<DenseTensor> sc(this, bases, counts, strides, InnerToOuter(bases.GetSize()));
  return sc;
}

ScanResult<DenseTensor> DenseTensor::Scan(const Point& bases, const Counts& counts, const Order& order, const Strides& strides)
{
  ScanResult<DenseTensor> sc(this, bases, counts, strides, order);
  return sc;
}

ScanResult<CompressedKeyDimensionTensor> CompressedKeyDimensionTensor::Scan(const Point& bases, const Counts& counts, const Strides& strides)
{
  ScanResult<CompressedKeyDimensionTensor> sc(this, bases, counts, strides, InnerToOuter(bases.GetSize()));
  return sc;
}

ScanResult<CompressedKeyDimensionTensor> CompressedKeyDimensionTensor::Scan(const Point& bases, const Counts& counts, const Order& order, const Strides& strides)
{
  ScanResult<CompressedKeyDimensionTensor> sc(this, bases, counts, strides, order);
  return sc;
}

Value RandomValue(const Point& idx)
{
  Value v = static_cast<Value>(rand() % 255);
  return v;
}


}  // namespace: swoop

bool IsEven(const swoop::Point& p)
{
  return (p[0] % 2) == 0;
}

int main(int argc, char **argv)
{
  swoop::CompressedKeyDimensionTensor foo({10, 10}, 50, 1, IsEven, swoop::RandomValue, "foo");

  for (swoop::Index x = 0; x < 10; x++)
  {
    for (swoop::Index y = 0; y < 10; y++)
    {
      if (foo.Has({x, y}))
      {
        std::cout << "{" << x << ", " << y << "}: " << foo[{x, y}] << std::endl;
      }
    }
  }

  for (auto xy : foo.Scan({0, 0}, {5, 10}, {2, 1}))
  {
    std::cout << xy.Show() << ": " << foo[xy] << std::endl;
  }
  return 0;
}
