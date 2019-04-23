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
 
#ifndef BELLMAN_FORD_HPP
#define BELLMAN_FORD_HPP

namespace dims
{

enum Connections
{
  S,
  D
};

}  // namespace dims

// Helper function to tile a 2D compressed tensor.

void TileCompressedTensor2D(const int& kNumDstsL0, 
                            const int& kNumDstsL1, 
                            whoop::Vec& connections_i, 
                            whoop::Tensor& connections_j, 
                            whoop::Tensor& connections_i_d0)
{
  int kNumNodes = connections_i.Size() - 1;
  // Fill in connections_i_d0 version.  
  // The first row starts out with the base number
  for (int s = 0; s < kNumNodes; s++)
  {
    connections_i_d0.At({s, 0}) = connections_i.At(s);
  }
  for (int s = 0; s < kNumNodes; s++)
  {
    // Now divide the points into tiles based on actual index.
    for (int di = connections_i.At(s); di < connections_i.At(s + 1); di++)
    {
      // Get the original, actual index.
      int d = connections_j.At({di, 0});
      // Figure out what tile it belongs to.
      int d1_cur = d / kNumDstsL0;
      // (If it's the last tile, no need to increment, as that is captured
      //  by the base of the next column.)
      if (d1_cur != kNumDstsL1 - 1)
      {
        connections_i_d0.At({s, d1_cur + 1})++;
      }
    }
    // Transform the counts to absolute indexes by adding the above
    // column bases.
    for (int d1 = 0; d1 < kNumDstsL1 - 1; d1++)
    {
      connections_i_d0.At({s, d1 + 1}) += connections_i_d0.At({s, d1});
    }
  }
  // Set the last column's entries by hand. These are just the end limits.
  for (int x = 0; x < kNumDstsL1; x++)
  {
    connections_i_d0.At({kNumNodes, x}) = connections_i.At(kNumNodes);
  }
}

#endif
