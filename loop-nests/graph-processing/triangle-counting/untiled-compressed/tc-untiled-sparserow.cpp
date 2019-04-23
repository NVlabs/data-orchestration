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

  int numVertices = Offsets.Size()-1;

  whoop::T(0) << "Number of Vertices:       " << numVertices  << whoop::EndT;
  whoop::T(0) << "Number of Edges:          " << Destinations.Size() << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  // Short-form variable names
  const int V = numVertices;

  Var u("u");
  Var i("i"); //offset index of vertex u's neighbors 
  Var v("v");
  Var j("j"); //offset index of vertex v's neighbors
  Var w("w");
  Var k("k"); //offset index of vertex w's neighbors
  Var ngh("ngh");

  // TODO: Find if Scalar outputs are supported?
  Triangles.Resize( 1 );

  t_for (u, 0, V);
  {
      // int v_start = Offsets[u];
      // int v_end   = Offsets[u+1];
      t_for (i, Offsets[u], Offsets[u+1]); 
      {
         v = Destinations[i];
         w_if (v != u);
         {
             // int w_start = Offsets[v];
             // int w_end   = Offsets[v+1];
             t_for (j, Offsets[v], Offsets[v+1]);
             {
                 w = Destinations[j];
                 w_if (w != v);
                 {
                     // int u_start = Offsets[w];
                     // int u_end   = Offsets[w+1];
                     t_for (k, Offsets[w], Offsets[w+1]);
                     {
                         ngh = Destinations[k];
                         w_if ( (ngh != u) && (ngh == u) );
                         {
                             Triangles[0] += 1;
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
  }
  end();

  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::T(0) << "DONE." << whoop::EndT;

  whoop::T(0) << "Number of Triangles discovered = " << Triangles.At(0) << whoop::EndT;
  whoop::T(0) << "Number of Unique Triangles discovered = " << Triangles.At(0) / 6 << whoop::EndT;
  
  whoop::Done();
}
