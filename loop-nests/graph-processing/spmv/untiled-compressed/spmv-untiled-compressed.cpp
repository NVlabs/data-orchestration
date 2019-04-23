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
#include <cassert>

int main(int argc, char** argv)
{

  using namespace whoop;

  VecIn  InOffsets("inoffsets");
  VecIn  Sources("sources");
  VecOut InDegrees("indegrees");      //stores parent of each vertex
  Vec    vData("vData");              //App-specific data

  std::string StatsFileName = "stats.txt";

  AddOption( &StatsFileName, "stats", "Stats File Name");

  whoop::Init(argc, argv);

  int numVertices = InOffsets.Size() - 1;
  int numEdges    = Sources.Size();

  whoop::T(0) << "Number of Vertices:       " << numVertices  << whoop::EndT;
  whoop::T(0) << "Number of Edges:          " << numEdges << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  
  // Short-form variable names
  const int V = numVertices;
  const int E = numEdges;
  const int NUM_ITERS = 10;
    
  // Initialize visited vector
  InDegrees.Resize(V);
  vData.Resize(V);

  for(int v=0; v < V; v++) 
  {
      InDegrees.At(v) = 0;
      vData.At(v)     = 1;
  }


  Var d("d"); 
  Var i("i");       //index for finding source
  Var s("s");
  Var l("l");
  
  t_for (l, 0, NUM_ITERS);
  {
      vData.AddTileLevel(1);
      InDegrees.AddTileLevel(1);
      InOffsets.AddTileLevel(2);
      Sources.AddTileLevel(1);

      t_for (d, 0, V);
      {
          t_for (i, InOffsets[d], InOffsets[d+1]);
          {
              s = Sources[i];
              InDegrees[d] += vData[s];
          }
          end();
      }
      end();
  }
  end();


  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::T(0) << "DONE." << whoop::EndT;

  /* Verification */
  for (int v = 0; v < V; v++) 
  {
      int s_start = InOffsets.At(v);
      int s_stop  = InOffsets.At(v+1);
      int degree  = s_stop - s_start;
      if (InDegrees.At(v) != NUM_ITERS * degree)
      {
          std::cout << "SpMV is incorrect\n";
          std::exit(-1);
      }
  }
  
  whoop::Done(StatsFileName);
}
