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

  TensorIn connections("connections");
  VecOut distances("distances");
  
  whoop::Init(argc, argv);
  
  int kNumSrcs = connections.Size(dims::S);
  int kNumDsts = connections.Size(dims::D);
  
  assert(kNumSrcs == kNumDsts);
  
  distances.Resize(kNumDsts);
  // Initialize to MAX_INT.
  for (int x = 0; x < kNumDsts; x++)
  {
    distances.At(x) = std::numeric_limits<int>::max() / 2;
  }
  distances.At(starting_node) = 0;
  
  whoop::T(0) << "Number of Nodes: " << kNumDsts << whoop::EndT;

  static const int V = kNumSrcs;
  
  Var i("i");
  Var s("s");
  Var d("d");

  t_for(i, 0, V);
  {
    t_for(s, 0, V);
    {
      t_for(d, 0, V);
      {
        w_if (connections[s][d] != 0);
        {
          w_if (distances[d] > (distances[s] + connections[s][d]));
          {
            distances[d] = distances[s] + connections[s][d];
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
