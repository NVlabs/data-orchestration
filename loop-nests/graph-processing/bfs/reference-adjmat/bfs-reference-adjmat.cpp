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

  VecOut connected("connected");
  Vec    parent("parent");

  Vec    process("process");
  Vec    visited("visited");

  whoop::Init(argc, argv);

  int numVertices  = sqrt(AdjacencyMatrix.Size());

  whoop::T(0) << "Number of Vertices:       " << numVertices  << whoop::EndT;
  whoop::T(0) << "Adjacency Matrix Size:    " << AdjacencyMatrix.Size() << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  
  // Short-form variable names
  const int V = numVertices;
    
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

  bool updates = true;
  
  while( updates ) 
  {
      updates = false;
      
      for( int s = 0; s < V; s++) 
      {
          if( process.At(s) == 1 ) 
          {
              visited.At(s) = 1;  
              process.At(s) = 0;
              
              connected.At(s) = 1;
              
              for (int d = 0; d <  V; d++)
              {
                  if( (AdjacencyMatrix.At(s*V + d) == 1) && (visited.At(d) == 0) && (process.At(d) == 0) ) 
                  {
                      updates = true;
                      parent.At(d) = s;
                      process.At(d) = 1;
                  }
              }
          }
      }
  }

  whoop::T(0) << "DONE." << whoop::EndT;

  for (int v = 0; v < V; v++)
  {
      whoop::T(3) << "connected " << v << " = " << connected.At(v) << " Parent: "<<parent.At(v)<<whoop::EndT;
  }
  
  whoop::Done();
}
