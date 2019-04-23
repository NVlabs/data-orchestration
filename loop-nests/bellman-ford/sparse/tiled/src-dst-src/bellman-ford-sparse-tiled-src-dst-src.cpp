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

#include "bellman-ford.hpp"


//Bellman-Ford algorithm w/o negative cycle check 
int main(int argc, char** argv)
{

  using namespace whoop;

  static const int starting_node = 0;

  VecIn connections_v("connections_v");
  VecIn connections_i("connections_i");
  TensorIn connections_j("connections_j");
  VecOut distances("distances");

  int kNumDstsL0 = 5;
  int kNumSrcsL0 = 5;

  whoop::AddOption(&kNumDstsL0, "num_dsts_0,d", "Size of L0 dst tile.");
  whoop::AddOption(&kNumSrcsL0, "num_srcs_0,s", "Size of L1 src tile.");
  
  whoop::Init(argc, argv);
  
  int kNumNodes = connections_i.Size() - 1;
  int kNumEdges = connections_v.Size();
  
  assert(connections_v.Size() == connections_j.Size(0));
  assert(kNumNodes % kNumDstsL0 == 0);
  assert(kNumNodes % kNumSrcsL0 == 0);

  const int kNumDstsL1 = kNumNodes / kNumDstsL0;
  const int kNumSrcsL1 = kNumNodes / kNumSrcsL0;
  
  distances.Resize(kNumNodes);
  // Initialize to MAX_INT.
  for (int x = 0; x < kNumNodes; x++)
  {
    distances.At(x) = std::numeric_limits<int>::max() / 2;
  }
  distances.At(starting_node) = 0;

  TensorPort distances2(&distances);
  
  whoop::T(0) << "Number of Nodes: " << kNumNodes << whoop::EndT;
  whoop::T(0) << "Number of Edges: " << kNumEdges << whoop::EndT;

  Tensor connections_i_d0({kNumNodes + 1, kNumDstsL1}, 0, "connections_i_d0");
  
  whoop::T(0) << "Beginning to tile the Index matrix into " << kNumDstsL1 << " tiles of max size: " << kNumDstsL0 << whoop::EndT;
  TileCompressedTensor2D(kNumDstsL0, kNumDstsL1, connections_i, connections_j, connections_i_d0);
  whoop::T(0) << "Done." << whoop::EndT;

  // Short-form variable names
  static const int V = kNumNodes;
  const int D0 = kNumDstsL0;
  const int D1 = kNumDstsL1;
  const int S0 = kNumSrcsL0;
  const int S1 = kNumSrcsL1;
  
  Var i("i");
  Var s("s");
  Var d("d");
  Var di0("di0");
  Var di0_start("di0_start");
  Var di0_end("di0_end");
  Var d1("d1");
  Var s0("s0");
  Var s1("s1");

  t_for(i, 0, V);
  {
    t_for(s1, 0, S1);
    {
      distances.AddTileLevel(D0);
      distances2.AddTileLevel(S0);

      t_for(d1, 0, D1);
      {
        t_for(s0, 0, S0);
        {
          connections_v.AddTileLevel(1);
          connections_j.AddTileLevel(1);
          connections_i_d0.AddTileLevel(2);
          distances.AddTileLevel(1);
          distances2.AddTileLevel(1);
          
          s = s1 * S0 + s0;

          di0_start = connections_i_d0[s][d1];
          w_if(d1 != D1 - 1);
          {
            di0_end = connections_i_d0[s][d1+1];
          }
          w_else();
          {
            di0_end = connections_i_d0[s+1][0];
          }
          end();
          
          t_for(di0, di0_start, di0_end);
          {
            d = connections_j[di0][0];
            w_if (distances[d] > (distances2[s] + connections_v[di0]));
            {
              distances[d] = distances2[s] + connections_v[di0];
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
  }
  end();

  std::cout << "RUNNING..." << std::endl;
  whoop::Run();
  std::cout << "DONE." << std::endl;

  for (int v = 0; v < V; v++)
  {
    whoop::T(1) << "Distances " << v << " = " << distances.At(v) << whoop::EndT;
  }
  
  whoop::Done();
}
