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

  static const int starting_node = 0;

  TensorIn connections("connections_dense");
  VecOut pagerank("pagerank");

  Vec residuals("residuals");  
  Vec old_residuals("old_residuals");
  Vec is_in_frontier("is_in_frontier");
  Vec degree("degree");
  
  whoop::Init(argc, argv);
  
  int kNumSrcs = connections.Size(0);
  int kNumDsts = connections.Size(1);
  
  assert(kNumSrcs == kNumDsts);
  whoop::T(0) << "Number of Nodes: " << kNumDsts << whoop::EndT;
  
  is_in_frontier.Resize(kNumDsts);
  pagerank.Resize(kNumDsts);
  residuals.Resize(kNumDsts);
  old_residuals.Resize(kNumDsts);
  degree.Resize(kNumDsts);

  // Initialize starting residuals
  is_in_frontier.At(starting_node) = 1;
  old_residuals.At(starting_node) = 1;
  residuals.At(starting_node) = 1;
  
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

  static const int V = kNumSrcs;
  static const int weight1 = 0xAA;
  static const int weight2 = 0xBB;
  static const int epsilon = 12;

  bool active = true;
  while (active)
  {
    whoop::T(1) << "Starting iteration..." << whoop::EndT;
    active = false;
    for (int s = 0; s < V; s++)
    {
      if (is_in_frontier.At(s) != 0)
      {
        whoop::T(2) << "Ranking frontier node: " << s << whoop::EndT;
        active = true;
        pagerank.At(s) = pagerank.At(s) + weight1 * old_residuals.At(s);
        residuals.At(s) = 0;
      }
    }
    if (active)
    {
      for (int s = 0; s < V; s++)
      {
        if (is_in_frontier.At(s) != 0)
        {
          int update = weight2 * old_residuals.At(s) / degree.At(s);
          for (int d = 0; d < V; d++)
          {
            if (connections.At({s, d}) != 0)
            {
              whoop::T(2) << "Propagating " << s << " to: " << d << whoop::EndT;
              residuals.At(d) = residuals.At(d) + update;
            }
          }
        }
      }
      for (int s = 0; s < V; s++)
      {
        is_in_frontier.At(s) = (residuals.At(s) >= degree.At(s) * epsilon) && (degree.At(s) > 0);
        old_residuals.At(s) = residuals.At(s);
      }
    }
  }
  
  whoop::Done();
}
