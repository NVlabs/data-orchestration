/* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

int main(int argc, char** argv)
{

  using namespace whoop;

  VecIn  Offsets("offsets");
  VecIn  Destinations("destinations");
  VecOut Triangles("triangles");
  
  
  whoop::Init(argc, argv);

  int numVertices  = Offsets.Size()-1; 

  whoop::T(0) << "Number of Vertices:       " << Offsets.Size()-1  << whoop::EndT;
  whoop::T(0) << "Number of Edges:          " << Destinations.Size() << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  // Short-form variable names
  const int V = numVertices;
    
  // TODO: Find if Scalar outputs are supported?
  Triangles.Resize( 1 );

  whoop::T(0) << "RUNNING..." << whoop::EndT;

  for (int u = 0; u < V; u++)
  {
      int v_start = Offsets.At( u );
      int v_end   = Offsets.At( u+1 );
      for (int i = v_start; i < v_end; i++) 
      {
         int v = Destinations.At( i );
         if (v == u) continue;
         int w_start = Offsets.At( v ); 
         int w_end   = Offsets.At( v+1 );
         for (int j = w_start; j < w_end; j++) 
         {
            int w = Destinations.At( j );
            if (w == v) continue;
            int u_start = Offsets.At( w );
            int u_end   = Offsets.At( w+1 );
            for (int k = u_start; k < u_end; k++) {
                int ngh = Destinations.At( k );
                if (ngh == w) continue;
                if (ngh == u) ++Triangles.At(0);
            }
         }
      }
  }

  whoop::T(0) << "DONE." << whoop::EndT;
 
  whoop::T(0) << "Number of Triangles discovered = " << Triangles.At(0) << whoop::EndT;
  whoop::T(0) << "Number of Unique Triangles discovered = " << Triangles.At(0) / 6 << whoop::EndT;
  whoop::Done();
}
