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
#include <cassert>

int main(int argc, char** argv)
{

  using namespace whoop;

  VecIn  InOffsets("in-offsets");
  VecIn  Sources("sources");
  VecOut Parent("parent");      //stores parent of each vertex
  Vec    Frontier("frontier");  //stores level (in bfs-tree) of each vertexr"); 

  Vec    Stop("stop");          //stopping condition 
  
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
    
  // Initialize visited vector
  Parent.Resize(V);
  Frontier.Resize(V);
  Stop.Resize(1);

  for(int v=0; v < V; v++) 
  {
      Parent.At(v)   = -1; 
      Frontier.At(v) = -1; 
      Stop.At(0)     = 1;
  }

  const int src      = 0;   //TODO: Add support to find "good" sources
  Parent.At( src )   = src;
  Frontier.At( src ) = 0;   //source node is at root of bfs tree


  Var l("l");       //level of BFS tree
  Var d("d"); 
  Var i("i");       //index for finding source

  t_for (l, 0, V);
  {
      Frontier.AddTileLevel(1); 
      Parent.AddTileLevel(1);       //zero-reuse within a tile
      /* 
       * There is reuse in the TileInOffsets[][][d]. Since the 
       * second element TileInOffsets[][][d+1] will be reused
       * in the next iteration (when d changes).
       */
      InOffsets.AddTileLevel(2);
      Sources.AddTileLevel(1);  //zero-reuse within a tile

      t_for (d, 0, V);
      {
          w_if (Parent[d] == -1);
          {
              t_for (i, InOffsets[d], InOffsets[d+1]);
              {

                  w_if (Frontier[Sources[i]] == l);
                  {
                      Parent[d]   = Sources[i];
                      Frontier[d] = l + 1;
                      Stop[0]     = 0;
                      //i = InOffsets[d+1]; //work-around for break statement
                      i = E+1;  //TODO: the above statement caused a compile error
                  }
                  end();
              }
              end();
          }
          end();
      }
      end();

      w_if (Stop[0] == 1);
      {
          l = V; //work-around for break statement
      }
      end();
  }
  end();


  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::T(0) << "DONE." << whoop::EndT;

  // Stats (depth of BFS tree and number of connected nodes)
  int depth(0);
  long connections(0);
  for (int v = 0; v < V; v++)
  {
      if (Frontier.At(v) != -1)
      {
          ++connections;    
          if (Frontier.At(v) > depth) 
          {
              depth = Frontier.At(v);
          }
      }
  }
  whoop::T(3) << "Depth of BFS Tree =  " << depth << " and number of connected nodes = " << connections <<whoop::EndT;
  
  whoop::Done(StatsFileName);
}
