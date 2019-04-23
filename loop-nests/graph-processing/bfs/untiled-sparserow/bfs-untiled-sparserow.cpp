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

  VecOut connected("connected");
  Vec    parent("parent");

  Vec    process("process");
  Vec    visited("visited");

  whoop::Init(argc, argv);

  int numVertices  = Offsets.Size()-1;
  int numEdges     = Destinations.Size();

  // Short-form variable names
  const int V = numVertices;

  whoop::T(0) << "Number of Vertices: " << V  << whoop::EndT;
  whoop::T(0) << "Number of Edges:    " << numEdges << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  
    
  // Initialize visited vector
  visited.Resize( V );
  process.Resize( V );
  connected.Resize( V );
  parent.Resize( V );

  for(int v=0; v < V; v++) 
  {
      visited.At(v)   = 0;
      parent.At(v)    = -1;
      connected.At(v) = 0;
      process.At(v)   = 0;
  }


  whoop::T(0) << "RUNNING..." << whoop::EndT;

  const int src = 0;
  process.At( src ) = 1;

  Var m("m");
  Var s("s");
  Var i("i");

  t_for(m, 0, V);
  {
      t_for(s, 0, V);
      {
          w_if( process[s] == 1 );
          {
              visited[s]   = 1;
              process[s]   = 0;
              connected[s] = 1;

              t_for(i, Offsets[s], Offsets[s+1]);
              {
                  w_if( (visited[ Destinations[i] ] == 0) && (process[ Destinations[i] ] == 0) );
                  {
                      parent[ Destinations[i] ]  = s;
                      process[ Destinations[i] ] = 1;
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

  for (int v = 0; v < V; v++)
  {
      whoop::T(3) << "connected " << v << " = " << connected.At(v) << " Parent: "<<parent.At(v)<<whoop::EndT;
  }
  
  whoop::Done();
}
