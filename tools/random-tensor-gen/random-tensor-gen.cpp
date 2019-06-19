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

int main(int argc, char** argv)
{
  std::vector<int> dim_sizes; 
  int seed = 1717;
  int min_val = 0;
  int max_val = 255;
  int monotonically_increasing = 0;

  whoop::AddOption(&dim_sizes, "dim_sizes,d", "List of dimension sizes, e.g. --dim_sizes 100 200 300");
  whoop::AddOption(&seed,      "seed,s",      "Seed for random number generator.");
  whoop::AddOption(&max_val,   "max,x",       "Maximum possible random number.");
  whoop::AddOption(&min_val,   "min,m",       "Minimum possible random number.");
  whoop::AddOption(&monotonically_increasing,   "monotonic,mm",       "Should the numbers monotonically increase?");

  whoop::TensorOut output("output");

  whoop::Init(argc, argv);
  
  output.Resize(dim_sizes);
  
  srand(seed);
  
  whoop::T(0) << "Generating random tensor, seed: " << seed << whoop::EndT;
  for (int x = dim_sizes.size() - 1; x >= 0; x--)
  {
    whoop::T(0) << "  Dimension " << x << ", size: " << dim_sizes[dim_sizes.size() - x - 1] << whoop::EndT;
  }
  
  int size = output.PrimSize();
  for (int x = 0; x < size; x++)
  {
    output.PrimAt(x) = (rand() % (max_val - min_val)) + min_val;
  }
  if (monotonically_increasing != 0)
  {
    for (int x = 1; x < size; x++)
    {
      output.PrimAt(x) += output.PrimAt(x-1);
    }
  }
  
  whoop::Done();
}
