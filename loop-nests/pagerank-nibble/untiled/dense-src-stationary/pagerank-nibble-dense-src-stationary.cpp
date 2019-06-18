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
  
  Var s;
  Var d;
  Var done;
  Var update;
  
  TensorPort is_in_frontier2(&is_in_frontier);
  TensorPort is_in_frontier3(&is_in_frontier);
  TensorPort old_residuals2(&old_residuals);
  TensorPort old_residuals3(&old_residuals);
  TensorPort residuals2(&residuals);
  TensorPort residuals3(&residuals);
  TensorPort degree3(&degree);
  
  done = 0;
  w_while(done == 0);
  {
    is_in_frontier.AddTileLevel(V);
    is_in_frontier2.AddTileLevel(V);
    is_in_frontier3.AddTileLevel(V);
    pagerank.AddTileLevel(V);
    residuals.AddTileLevel(V);
    residuals2.AddTileLevel(V);
    residuals3.AddTileLevel(V);
    old_residuals.AddTileLevel(V);
    old_residuals2.AddTileLevel(V);
    old_residuals3.AddTileLevel(V);
    degree.AddTileLevel(V);
    degree3.AddTileLevel(V);

    done = 1;
    t_for (s, 0, V);
    {
      w_if (is_in_frontier[s] != 0);
      {
        done = 0;
        pagerank[s] += old_residuals[s] * weight1;
        residuals[s] = residuals[s] * 0; // TEMPORARY
      }
      end();
    }
    end();
    w_if (done == 0);
    {
      t_for (s, 0, V);
      {
        w_if (is_in_frontier2[s] != 0);
        {
          update = (old_residuals2[s] * weight2) / degree[s];
          t_for (d, 0, V);
          {
            w_if (connections[s][d] != 0);
            {
              residuals2[d] = residuals2[d] + update;
            }
            end();
          }
          end();
        }
        end();
      }
      end();
      t_for (s, 0, V);
      {
        is_in_frontier3[s] = (residuals3[s] >= degree3[s] * epsilon) && (degree3[s] > 0) || (is_in_frontier3[s] && 0); // TEMPORARY
        old_residuals3[s] = residuals3[s] + (old_residuals3[s] * 0); // TEMPORARY
      }
      end();
    }
    end();
  }
  end();
  
  whoop::Run();

  whoop::Done();
}
