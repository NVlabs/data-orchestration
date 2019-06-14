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
 
 #include <limits>

#include "whoop.hpp"


//Bellman-Ford algorithm w/o negative cycle check 
int main(int argc, char** argv)
{

  using namespace whoop;

  TensorIn connections("connections_dense");
  VecIn seeds("seeds");
  TensorOut pagerank("pagerank");

  Tensor residuals("residuals");  
  Tensor old_residuals("old_residuals");
  Tensor is_in_frontier("is_in_frontier");
  Vec    degree("degree");
  
  whoop::Init(argc, argv);
  
  int kNumSrcs = connections.Size(0);
  int kNumDsts = connections.Size(1);
  int kNumSeeds = seeds.Size();
  
  assert(kNumSrcs == kNumDsts);
  whoop::T(0) << "Number of Nodes: " << kNumDsts << whoop::EndT;
  whoop::T(0) << "Number of Seeds: " << kNumSeeds << whoop::EndT;
  
  is_in_frontier.Resize({kNumSeeds, kNumDsts});
  pagerank.Resize({kNumSeeds, kNumDsts});
  residuals.Resize({kNumSeeds, kNumDsts});
  old_residuals.Resize({kNumSeeds, kNumDsts});
  degree.Resize(kNumDsts);

  // Initialize starting residuals
  for (int n = 0; n < kNumSeeds; n++)
  {
    is_in_frontier.At({n, seeds.At(n)}) = 1;
    old_residuals.At({n, seeds.At(n)}) = 1;
    residuals.At({n, seeds.At(n)}) = 1;
  }
  
  // Initialize degree
  for (int s = 0; s < kNumSrcs; s++)
  {
    for (int d = 0; d < kNumDsts; d++)
    {
      if (connections.At({s, d}) != 0)
      {
        degree.At(s) = degree.At(s) + 1;
      }
    }
  }

  static const int N = kNumSeeds;
  static const int V = kNumSrcs;
  static const int weight1 = 0xAA;
  static const int weight2 = 0xBB;
  static const int epsilon = 12;

  std::vector<bool> active(N, true);

  while (std::find(active.begin(), active.end(), true) != active.end())
  {
    whoop::T(1) << "Starting iteration..." << whoop::EndT;
    for (int n = 0; n < N; n++)
    {
      if (!active[n]) continue;
      whoop::T(1) << "Starting seed node: " << seeds.At(n) << whoop::EndT;
      active[n] = false;
      for (int s = 0; s < V; s++)
      {
        if (is_in_frontier.At({n, s}) != 0)
        {
          whoop::T(2) << "Ranking frontier node: " << s << whoop::EndT;
          active[n] = true;
          pagerank.At({n, s}) = pagerank.At({n, s}) + weight1 * old_residuals.At({n, s});
          residuals.At({n, s}) = 0;
        }
      }
      if (active[n])
      {
        for (int s = 0; s < V; s++)
        {
          if (is_in_frontier.At({n, s}) != 0)
          {
            int update = weight2 * old_residuals.At({n, s}) / degree.At(s);
            for (int d = 0; d < V; d++)
            {
              if (connections.At({s, d}) != 0)
              {
                whoop::T(2) << "Propagating " << s << " to: " << d << whoop::EndT;
                residuals.At({n, d}) = residuals.At({n, d}) + update;
              }
            }
          }
        }
        for (int s = 0; s < V; s++)
        {
          is_in_frontier.At({n, s}) = (residuals.At({n, s}) >= degree.At(s) * epsilon) && (degree.At(s) > 0);
          old_residuals.At({n, s}) = residuals.At({n, s});
        }
      }
      else
      {
        whoop::T(0) << "Seed node " << seeds.At(n) << " finished." << whoop::EndT;
      }
    }
  }
  
  whoop::Done();
}
