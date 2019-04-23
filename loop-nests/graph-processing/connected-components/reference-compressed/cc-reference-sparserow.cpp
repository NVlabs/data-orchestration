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
#include <map>

int main(int argc, char** argv)
{

  using namespace whoop;

  VecIn  Offsets("offsets");
  VecIn  Destinations("destinations");
  
  VecOut domain("domain");

  whoop::Init(argc, argv);

  int numVertices  = Offsets.Size()-1;
  int numEdges     = Destinations.Size();

  const int V      = numVertices;  

  whoop::T(0) << "Number of Vertices: " << V  << whoop::EndT;
  whoop::T(0) << "Number of Edges:    " << numEdges << whoop::EndT;

  whoop::T(0) << whoop::EndT;

    
  // Initialize domain
  domain.Resize( V );
  for(int v=0; v < V; v++) 
  {
      domain.At(v) = v;
  }

  whoop::T(0) << "RUNNING..." << whoop::EndT;

  bool updates = true;
  int iters     = 0;
  
  while( updates ) 
  {
      updates = false;
      
      for (int s = 0; s <  V; s++)
      {
          int d_start = Offsets.At( s );
          int d_end   = Offsets.At( s+1 );

          for (int i = d_start; i < d_end; i++)
          {
              int d = Destinations.At( i );
          
              if( domain.At(d) > domain.At(s) ) 
              {
                  updates = true;
                  //domain.At(d) = GetMin( domain.At(s), domain.At(d) );
                  domain.At(d) = domain.At(s);
              }
          }
      }
      ++iters;
  }

  whoop::T(0) << "DONE." << whoop::EndT;

  for (int v = 0; v < V; v++)
  {
    whoop::T(2) << "Domain " << v << " = " << domain.At(v) << whoop::EndT;
  }

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

  whoop::T(0) << "Number of iterations until convergence = " << iters << whoop::EndT;
  
  whoop::Done();
}
