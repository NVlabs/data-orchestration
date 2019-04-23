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
  VecIn  Destinations("destinations");
  
  VecOut domain("domain");
  
  Vec    iters("iters");
  Vec    updates("updates");

  whoop::Init(argc, argv);

  int numVertices  = Offsets.Size()-1;
  int numEdges     = Destinations.Size();

  const int V      = numVertices;  
  const int S      = numVertices;  

  whoop::T(0) << "Number of Vertices: " << V  << whoop::EndT;
  whoop::T(0) << "Number of Edges:    " << numEdges << whoop::EndT;

  whoop::T(0) << whoop::EndT;
    
  // Initialize domain
  domain.Resize( V );
  for(int v=0; v < V; v++) 
  {
      domain.At(v) = v;
  }
  updates.Resize(1);
  iters.Resize(1);

  Var m("m");
  Var s("s");
  Var d("d");
  Var i("i");

  t_for(m, 0, V);
  {
      updates[0] = 0; 
      t_for(s, 0, S);
      {
          t_for(i, Offsets[s], Offsets[s+1]);
          {
//               d = Destinations[i];
              
              w_if( domain[ Destinations[i] ] > domain[s] );
              {
                  domain[ Destinations[i] ] = domain[s];
                  updates[0] = 1;
              }
              end();
          }
          end();
      }
      end();

      w_if (updates[0] != 1);
      {
          m = V; //workaround for a while loop
      }
      end();

      iters[0] += 1;
  }
  end();

  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::T(0) << "DONE." << whoop::EndT;

//   for (int v = 0; v < V; v++)
//   {
//     whoop::T(3) << "Domain " << v << " = " << domain.At(v) << whoop::EndT;
//   }
  
  // Finding number of components
  std::map<int, int> components;
  for (int v = 0; v < V; v++) 
  {
      int compID = domain.At(v);
      if (components.find(compID) == components.end()) 
          components.insert(std::pair<int, int>(compID, 0));
      components[compID]++;
  }
  whoop::T(0) << "Number of components = " << components.size() << whoop::EndT;
  
  whoop::T(0) << "Number of iterations until convergence = " << iters.At(0) << whoop::EndT;

  /* Sanity check - the above loop nest assumes that the outer loop can do a maximum of V iterations */
  assert(iters.At(0) < V && "[ERROR] Need to run components for more iterations");
  
  
  whoop::Done();
}
