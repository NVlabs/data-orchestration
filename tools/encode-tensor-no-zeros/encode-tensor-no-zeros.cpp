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

void PrintTensor(const int& val, const std::vector<int>& cur_idx)
{
  whoop::T(0) << whoop::ast::ShowIndices(cur_idx) << ": " << val << whoop::EndT;
}

int main(int argc, char** argv)
{

  std::list<std::pair<int, std::vector<int>>> compressed_values;
  
  int key_dimension_index = 1;
  
  whoop::AddOption(&key_dimension_index, "key_dim_index,d", "Index of the dimension to not compress.");

  whoop::TensorIn input("input");
  whoop::VecOut output_v("output_v"); // Non-zero values
  whoop::VecOut output_i("output_i"); // Indices into the J table
  whoop::TensorOut output_j("output_j"); // Indices of non-zero values in the original tensor.
  
  whoop::Init(argc, argv);
  
  auto dim_sizes = input.DimensionSizes();
  
  whoop::T(0) << "Tensor has " << dim_sizes.size() << " dimensions (" << input.PrimSize() << " total elements)." << whoop::EndT;
  for (int x = 0; x < dim_sizes.size(); x++)
  {
    whoop::T(0) << "  Dimension " << x << ", size: " << dim_sizes[x] <<
       ((x == key_dimension_index) ? "  ** KEY **" : "") << whoop::EndT;
  }
  
  // We are compresssing all dimensions except the key.
  const int num_compressed_dims = input.NumDimensions() - 1;
  // The number of entries in the index table is the size of the key dimension.
  const int index_table_num_entries = input.Size(key_dimension_index);
  // Resize the index table and initialize all entries to 0.
  output_i.Resize(index_table_num_entries + 1);
  for (int x = 0; x < index_table_num_entries + 1; x++)
  {
    output_i[x] = 0;
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
      auto indices = input.PrimUnflattenIndex(x);
      // The value of the key dimension is where we record this.
      int index_in_key_dim = indices[key_dimension_index];
      // Record that we have another value in this entry in the key dim.
      output_i.At(index_in_key_dim + 1)++;
      // Erase out the key since it will be implicit.
      indices.erase(indices.begin() + key_dimension_index);
      // Record the commpressed index and the actual value.
      compressed_values.push_back(std::make_pair(input.PrimAt(x), indices));
    }
  }
  
  // Now transfer from std::list to VecOut.
  output_v.Resize(compressed_values.size());
  output_j.Resize({num_compressed_dims, static_cast<int>(compressed_values.size())});
  int k = 0;
  for (auto it = compressed_values.begin(); it != compressed_values.end(); it++)
  {
    // The v array holds values.
    output_v.At(k) = (*it).first;
    // The j array holds the indices.
    auto indices = (*it).second;
    for (int i = 0; i < indices.size(); i++)
    {
      output_j.At({k, i}) = indices[i];
    }
    k++;
  }
  
  // Re-base the i table into an absolute reference.
  int total = 0;
  for (int x = 0; x < index_table_num_entries + 1; x++)
  {
    int old_val = output_i.At(x);
    output_i.At(x) += total;
    total += old_val;
  }
  
  // Final print-outs.
  for (int x = 0; x < index_table_num_entries; x++)
  {
    whoop::T(0) << "Number of non-zero entries for index " << x << ": " << output_i.At(x + 1) - output_i.At(x) << whoop::EndT;
  }
  whoop::T(0) << "========================================" << whoop::EndT;
  whoop::T(0) << "Total number of non-zero values: " << compressed_values.size() << whoop::EndT;
  
  
  whoop::Done();
}
