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
 
#include <utility>
#include <vector>

#include "whoop.hpp"
#include "compressed-tensor.hpp"


void PrintTensor(const int& val, const std::vector<int>& cur_idx)
{
  whoop::T(0) << whoop::ast::ShowIndices(cur_idx) << ": " << val << whoop::EndT;
}

int main(int argc, char** argv)
{

  // Intermediate representation: list of non-zero values in sorted COO format.
  std::list<std::pair<whoop::Point, std::vector<int>>> compressed_values;
  
  // Note: currently assumes that dimension order of in-tensor = representation order of out-tensor.
  
  whoop::TensorIn input("input");
  whoop::CompressedTensorOut output("output");

  whoop::Init(argc, argv);
  
  auto dim_sizes = input.DimensionSizes();
  output.Resize(dim_sizes);
  
  whoop::T(0) << "Tensor has " << dim_sizes.size() << " dimensions (" << input.PrimSize() << " total original elements)." << whoop::EndT;
  for (int x = 0; x < dim_sizes.size(); x++)
  {
    whoop::T(0) << "  Dimension " << x << ", original size: " << dim_sizes[x] << whoop::EndT;
  }
  
  // Iterate through the tensor, looking for non-zero values.
  // We do this in the flat representation on the assumption that this
  // is pretty sparse to begin with.
  for (int x = 0; x < input.PrimSize(); x++)
  {
    if (input.PrimAt(x) != 0)
    {
       
      // Found a non-zero value.
      // Get the indices in the N-D tensor.
      auto point = input.PrimUnflattenIndex(x);
      whoop::T(0) << "  Generator: Found nonzero at : " << x << whoop::EndT;
      output.Insert(point, input.PrimAt(x));

    }
  }
  
  // Cleanup any over-allocations.
  output.TrimFat();
  
  // Final print-outs.
  output.PrintCompressedTensor(true);
  //output.DumpOutput();
  
  whoop::Done();
}
