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
  Vec    seed_done("seed_done");
  
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
  seed_done.Resize(kNumSeeds);

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

  Var n;
  Var s;
  Var d;
  Var update;
  Var done; 
  
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
    is_in_frontier.AddTileLevel(N*V);
    is_in_frontier2.AddTileLevel(N*V);
    is_in_frontier3.AddTileLevel(N*V);
    pagerank.AddTileLevel(N*V);
    residuals.AddTileLevel(N*V);
    residuals2.AddTileLevel(N*V);
    residuals3.AddTileLevel(N*V);
    old_residuals.AddTileLevel(N*V);
    old_residuals2.AddTileLevel(N*V);
    old_residuals3.AddTileLevel(N*V);
    degree.AddTileLevel(V);
    degree3.AddTileLevel(V);

    done = 1;
    t_for (n, 0, N);
    {
      w_if (seed_done[n] == 0);
      {
        done = 0;
        seed_done[n] = 1;
        t_for (s, 0, V);
        {
          w_if (is_in_frontier[n][s] != 0);
          {
            seed_done[n] = seed_done[n] * 0; // TEMPORARY
            pagerank[n][s] += old_residuals[n][s] * weight1;
            residuals[n][s] = residuals[n][s] * 0; // TEMPORARY
          }
          end();
        }
        end();
        w_if (seed_done[n] == 0);
        {
          t_for (s, 0, V);
          {
            w_if (is_in_frontier2[n][s] != 0);
            {
              update = (old_residuals2[n][s] * weight2) / degree[s];
              t_for (d, 0, V);
              {
                w_if (connections[s][d] != 0);
                {
                  residuals2[n][d] = residuals2[n][d] + update;
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
            is_in_frontier3[n][s] = (residuals3[n][s] >= degree3[s] * epsilon) && (degree3[s] > 0) || (is_in_frontier3[n][s] && 0); // TEMPORARY
            old_residuals3[n][s] = residuals3[n][s] + (old_residuals3[n][s] * 0); // TEMPORARY
          }
          end();
        }
        end();
      }
      end();
    }
    end();
  }
  end();
  
  whoop::Run();

  whoop::Done();
}
