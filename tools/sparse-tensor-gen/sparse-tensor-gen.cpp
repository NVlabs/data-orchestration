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
#include "randomdist.h"


int main(int argc, char** argv)
{
  std::vector<int> dim_sizes; //fraction is 1/val
  std::vector<int> dim_nz; 
  int fract_nz = 1; // fract is 1/val
  int seed = 1717;
  int min_val = 0;
  int max_val = 255;

  whoop::AddOption(&dim_sizes, "dim_sizes,d", "List of dimension sizes, e.g. --dim_sizes 100 200 300");
  whoop::AddOption(&dim_nz, "dim_nz,dnz", "Fraction of nonzeroes for each dimension , e.g. --dim_nz 1 2 3 (100%, 50%, 33%)");
  whoop::AddOption(&fract_nz, "fract_nz,nz", "Fraction nonzeroes overall, e.g. --fract_nz 2 (50%)");
  whoop::AddOption(&seed,      "seed,s",      "Seed for random number generator.");
  whoop::AddOption(&max_val,   "max,x",       "Maximum possible random number.");
  whoop::AddOption(&min_val,   "min,m",       "Minimum possible random number.");
  
  // Note: currently assumes that dimension order of in-tensor = representation order of out-tensor.
  
  whoop::CompressedTensorOut output("output");
  whoop::Init(argc, argv);

  output.Resize(dim_sizes);

  srand(seed);

  
  //seed the random numbers by the current system time
  srand(time(NULL));

  int total_size = 1;
  whoop::T(0) << "Output Tensor will have " << dim_sizes.size() << " dimensions." << whoop::EndT;
  for (int i = 0; i < dim_sizes.size(); i++)
  {
    whoop::T(0) << "  Dimension " << i << ", size: " << dim_sizes[dim_sizes.size() - i - 1] << whoop::EndT;
  }

  //generate nonzeroes
  CTNonzeroGenerator generator = CTNonzeroGenerator(dim_sizes[0], dim_sizes[1]);
  generator.SetSymmetric();
  generator.Generate();
  generator.Dump2D();
  generator.SortDegree();
  generator.Dump2D();


   //done attempting to generate values, move to next non-zero coordinates 
  //  for outer dimensions
  std::vector<int> curr_position(2); 
  for (Node node : generator.nonzeroes_) {
      curr_position[1] = node.x_;
      curr_position[0] = node.y_;
      output.Insert(curr_position, node.val_);
  }
  
   
  // Cleanup any over-allocations.
  output.TrimFat();
  
  // Final print-outs.
  output.PrintCompressedTensor(true);
  //output.DumpOutput();
  
  whoop::Done();
}
