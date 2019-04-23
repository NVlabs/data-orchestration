/* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
 
#include "whoop.hpp"

void PrintTensor(const int& val, const std::vector<int>& cur_idx)
{
  whoop::T(0) << whoop::ast::ShowIndices(cur_idx) << ": " << val << whoop::EndT;
}

int main(int argc, char** argv)
{

  whoop::TensorIn input("input");

  whoop::Init(argc, argv);
  
  auto dim_sizes = input.DimensionSizes();
  
  whoop::T(0) << "Tensor has " << dim_sizes.size() << " dimensions (" << input.PrimSize() << " total elements)." << whoop::EndT;
  for (int x = 0; x < dim_sizes.size(); x++)
  {
    whoop::T(0) << "  Dimension " << x << ", size: " << dim_sizes[x] << whoop::EndT;
  }

  whoop::T(0) << "Values: " << whoop::EndT;
  
  input.TraverseAll(PrintTensor);
  
  whoop::Done();
}
