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

  VecIn  Offsets("offsets");
  VecIn  Sources("sources");
  
  VecOut domain("domain");

  whoop::Init(argc, argv);

  int numVertices  = Offsets.Size()-1;
  int numEdges     = Sources.Size();

  const int V      = numVertices;  
  const int S      = numVertices;  
  const int D      = numVertices;  

  whoop::T(0) << "Number of Vertices: " << V  << whoop::EndT;
  whoop::T(0) << "Number of Edges:    " << numEdges << whoop::EndT;

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
  Var i("i");

  Var o_s("offset_start");
  Var o_e("offset_end");

  // For undirected graphs, CSR and CSC formats are identical
  // for destination stationary, we are going to need CSC version

  t_for(m, 0, V);
  {
      t_for(d, 0, D);
      {
          o_s = Offsets[d];
          o_e = Offsets[d+1];
          t_for(i, o_s, o_e);
          {
              s = Sources[i];
              
              w_if( domain[d] > domain[s] );
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

  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::T(0) << "DONE." << whoop::EndT;

//   for (int v = 0; v < V; v++)
//   {
//     whoop::T(3) << "Domain " << v << " = " << domain.At(v) << whoop::EndT;
//   }
  
  whoop::Done();
}
