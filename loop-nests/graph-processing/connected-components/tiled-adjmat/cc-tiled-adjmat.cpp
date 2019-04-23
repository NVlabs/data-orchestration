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

#include "whoop.hpp"
#include <math.h>

int GetMin( int a, int b)
{
    if( a < b ) return a;
    return b;
}

int main(int argc, char** argv)
{

  using namespace whoop;

  VecIn  AdjacencyMatrix("adjmat");
  
  VecOut domain("domain");

  whoop::Init(argc, argv);

  int numVertices  = sqrt(AdjacencyMatrix.Size());

  // Short-form variable names
  const int V = numVertices;
  const int S = V;
  const int D = V;

  const int S_TILE_SIZE = 16;
  const int D_TILE_SIZE = 16;

  const int S0 = S_TILE_SIZE;
  const int D0 = D_TILE_SIZE;
  
  const int S1 = S / S0;
  const int D1 = D / D0;

  whoop::T(0) << "Number of Vertices:       " << numVertices  << whoop::EndT;
  whoop::T(0) << "Adjacency Matrix Size:    " << AdjacencyMatrix.Size() << whoop::EndT;
  whoop::T(0) << "Tiling with S0:           " << S0 << whoop::EndT;
  whoop::T(0) << "Tiling with S1:           " << S1 << whoop::EndT;
  whoop::T(0) << "Tiling with D0:           " << D0 << whoop::EndT;
  whoop::T(0) << "Tiling with D1:           " << D1 << whoop::EndT;
  
  whoop::T(0) << whoop::EndT;
  
  // Initialize domain
  domain.Resize( V );
  for(int v=0; v < V; v++) 
  {
      domain.At(v) = v;
  }

  Var m("m");
  Var s("s");
  Var d("d");
  
  Var s0("s0");
  Var d0("d0");
  Var s1("s1");
  Var d1("d1");
  
  t_for(m, 0, V); // replace with while later
  {
      t_for(s1, 0, S1);
      {
          t_for(d1, 0, D1);
          {
              t_for(s0, 0, S0);
              {
                  t_for(d0, 0, D0);
                  {
                      domain.AddTileLevel(D0);

                      s = s0 + s1*S0;
                      d = d0 + d1*D0;

                      w_if( (AdjacencyMatrix[ s*V + d ] == 1) && (domain[d] > domain[s]) );
                      {
                          domain[d] = domain[s];
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

  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::T(0) << "DONE." << whoop::EndT;

//   for (int v = 0; v < V; v++)
//   {
//     whoop::T(3) << "Domain " << v << " = " << domain.At(v) << whoop::EndT;
//   }
  
  whoop::Done();
}
