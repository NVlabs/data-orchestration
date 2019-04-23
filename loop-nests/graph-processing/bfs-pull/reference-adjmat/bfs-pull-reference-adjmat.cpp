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

int GetMin( int a, int b)
{
    if( a < b ) return a;
    return b;
}

int main(int argc, char** argv)
{

  using namespace whoop;

  VecIn  AdjacencyMatrix("adjmat");
  VecOut Parent("parent");      //stores parent of each vertex
  Vec    Frontier("frontier");  //stores level (in bfs-tree) of each vertexr"); 


  whoop::Init(argc, argv);

  int numVertices  = sqrt(AdjacencyMatrix.Size());

  whoop::T(0) << "Number of Vertices:       " << numVertices  << whoop::EndT;
  whoop::T(0) << "Adjacency Matrix Size:    " << AdjacencyMatrix.Size() << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  
  // Short-form variable names
  const int V = numVertices;
    
  // Initialize visited vector
  Parent.Resize( V );
  Frontier.Resize( V );

  for(int v=0; v < V; v++) 
  {
      Parent.At(v)   = -1; 
      Frontier.At(v) = -1; 
  }


  whoop::T(0) << "RUNNING..." << whoop::EndT;

  const int src      = 0;   //TODO: Add support to find "good" sources
  Parent.At( src )   = src;
  Frontier.At( src ) = 0;   //source node is at root of bfs tree

  bool stop = false; 

  int level (0);
  
  while( !stop ) 
  {
      stop = true;
      for (int d = 0; d < V; d++) 
      {
          if (Parent.At(d) == -1)
          {
              for (int s = 0; s < V; s++) 
              {
                  if (AdjacencyMatrix.At(s*V + d) == 0) continue;

                  if (Frontier.At(s) == level) 
                  {
                      Parent.At(d)   = s;
                      Frontier.At(d) = level + 1;
                      stop = false;
                      break;
                  }
              }
          }
      }
      ++level;
  }

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
  if (depth != --level) 
  {
    std::cout << "Depth (According to frontiers) = " << depth << std::endl;
    std::cout << "Depth (According to levels)    = " << level << std::endl;
  }
  
  whoop::Done();
}
