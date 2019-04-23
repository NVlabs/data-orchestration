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
#include <vector>
#include <random>

using namespace whoop;

const int PUSH {0};
const int PULL {1};

// Copy contents of front_src to front_dst. (src and dst are used in the 
// context of source and target of the copy command - not src and dst vertices)
// Destructive copy - resets front1 
//TODO: replace with library support for swapping frontiers
void copyFrontier(Vec& front_src, Vec& front_dst, int numVertices)
{
    //copying the contents
    for (int i = 0; i <= numVertices; ++i)
    {
        front_dst.At(i) = front_src.At(i);
        front_src.At(i) = 0;
    }
    return;
}

// Convert a bitmap representation of the frontier to a sparse 
// representation (list of vertices in the frontier)
void convertToSparseFrontier (Vec* frontier, int V) 
{
    if (frontier->At(V) < V)
      return; //already a sparse frontier
    
    std::vector<int> frontierElems;
    for (int i = 0; i < V; ++i)
    {
        if (frontier->At(i) == 1) 
            frontierElems.push_back(i);
    }
    
    int currPos = 0;
    for (auto &elem : frontierElems)
    {
        frontier->At(currPos) = elem;
        ++currPos;
        frontier->At(V) = currPos;
    }
}

// Convert a sparse representation of the frontier to a bitmap
// (with bits set for active vertices in the frontier)
void convertToDenseFrontier (Vec* frontier, int V)
{
    // Assumption: we will never have a sparse frontier with V elements
    if (frontier->At(V) == V)
        return; //already a dense frontier

    std::vector<int> frontierElems;
    for (int i = 0; i < frontier->At(V); ++i)
    {
         frontierElems.push_back(frontier->At(i));
    }

    int currPos = 0;
    for (auto &elem : frontierElems)
    {
        frontier->At(elem) = 1;
    }
    frontier->At(V) = V;
}

// Threshold values can be played around with to control if we should be 
// just be push (src-stn) in all iterations or pull (dst-stn) in all 
// iterations
int PushOrPull(Vec* frontier, VecIn& Offsets, int V, int numEdges, int threshold = -1)
{
    int frontierEdges = 0; // no. of edges emanating from sources in frontiers
    int frontierSize  = 0;

    if (frontier->At(V) < V) 
    {
        frontierSize = frontier->At(V);
        for (int i = 0; i < frontier->At(V); ++i)
        {
            int vtx = frontier->At(i);
            frontierEdges += (Offsets.At(vtx+1) - Offsets.At(vtx));
        }
    }
    else 
    {
        for (int i = 0; i < V; ++i)
        {
            if (frontier->At(i) == 1)
            {
                ++frontierSize;
                frontierEdges += (Offsets.At(i+1) - Offsets.At(i));
            }
        }
    }

    if (threshold == -1)
        threshold = numEdges / 20;

    if ((frontierEdges + frontierSize) > threshold)
        return PULL;
    else
        return PUSH;
}

int main(int argc, char** argv)
{
  VecIn  Offsets("offsets");
  VecIn  Destinations("destinations");
  VecIn  InOffsets("in-offsets");
  VecIn  Sources("sources");
  VecIn  Mapping("mapping");    //stores a mapping from the original ID to the reordered ID
  VecOut Parent("parent");      //stores parent of each vertex
  Vec    CurrFrontier("curr_frontier");  //stores level (in bfs-tree) of each vertexr"); 
  Vec    NextFrontier("next_frontier"); //frontier for the next iteration

  Vec    Stop("stop");          //stopping condition 
  
  std::string StatsFileName = "stats.txt";
  AddOption( &StatsFileName, "stats", "Stats File Name");
  
  int BURST_SIZE = 1;
  AddOption(&BURST_SIZE, "burst_size", "Data Transfer Granularity");


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
  /* for frontiers we are conservatively sizing it to V+1 because a completely
   * dense iteration would have a frontier with all the vertices in it.
   * The additional element at the end (index V) indicates the number of vertices
   * in the frontier (in case of a sparse frontier) */
  CurrFrontier.Resize(V+1);
  NextFrontier.Resize(V+1);
  Stop.Resize(1);

  for(int v=0; v < V; v++) 
  {
      Parent.At(v)   = -1; 
      CurrFrontier.At(v) = -1; 
      NextFrontier.At(v) = -1; 
      Stop.At(0)     = 1;
  }
  CurrFrontier.At(V) = 0; //no elements in the frontier
  NextFrontier.At(V) = 0;

  std::mt19937 rng(27491095);
  std::uniform_int_distribution<int> udist(0, V-1);
 
  // Find a source that has out_degree > 0
  int src;
  do {
    src = udist(rng);
  } while ((Offsets.At(src+1)-Offsets.At(src)) == 0);

  const int remapped_src    = Mapping.At(src);
  Parent.At( remapped_src ) = remapped_src; 
  /* we start with a sparse frontier containing only the (remapped) source vertex */
  CurrFrontier.At(0) = remapped_src;   //source node is at root of bfs tree
  CurrFrontier.At(V) = 1;   //one element in current frontier


  Var l("l");       //level of BFS tree
  Var d("d"); 
  Var s("s");
  Var i("i");       //index for finding source

  Var iterType("iterType"); //deciding between push/pull (src/dst-stn) iteration

  // push-phase variables
  Var s_start("s_start");
  Var frontSize("frontSize");
  Var currPos("currPos");

  t_for (l, 0, V);
  {
      CurrFrontier.AddTileLevel(BURST_SIZE, BURST_SIZE, BURST_SIZE); 
      NextFrontier.AddTileLevel(BURST_SIZE, BURST_SIZE, BURST_SIZE); 
      Parent.AddTileLevel(BURST_SIZE, BURST_SIZE, BURST_SIZE);       //zero-reuse within a tile
      /* 
       * There is reuse in the TileInOffsets[][][d]. Since the 
       * second element TileInOffsets[][][d+1] will be reused
       * in the next iteration (when d changes).
       */
      InOffsets.AddTileLevel(BURST_SIZE, BURST_SIZE, BURST_SIZE);
      Sources.AddTileLevel(BURST_SIZE, BURST_SIZE, BURST_SIZE);  //zero-reuse within a tile
      Offsets.AddTileLevel(BURST_SIZE, BURST_SIZE, BURST_SIZE);
      Destinations.AddTileLevel(BURST_SIZE, BURST_SIZE, BURST_SIZE);  //zero-reuse within a tile
      
      iterType = PushOrPull(&CurrFrontier, Offsets, V, E); 

      w_if (iterType == PUSH);
      {
          /* ~~~~~~~~~~~~~ Push Phase ~~~~~~~~~~~~~ */
          // during a push phase we use a sparse representation 
          // of the frontier (list of vertices in frontier)
          convertToSparseFrontier(&CurrFrontier, V);

          s_start   = 0;
          frontSize = CurrFrontier[V];
          t_for (s, s_start, frontSize);
          {
              //visit neighbors of each source in frontier
              t_for (i, Offsets[s], Offsets[s+1]);
              {
                  d = Destinations[i];
                  w_if (Parent[d] == -1);
                  {
                      Parent[d] = s;
                      currPos = NextFrontier[V];
                      NextFrontier[currPos] = d;  
                      currPos += 1;
                      NextFrontier[V] = currPos;
                  }
                  end();
              }
              end();
          }
          end();
          w_if (NextFrontier[V] == 0); 
          {
              // no vertex in the next frontier
              // visited all reachable vertices from the source
              l = V; //work-around for break statement
          }
          end();
          copyFrontier(NextFrontier, CurrFrontier, V);
      }
      end();

      w_if (iterType == PULL);
      {
          /* ~~~~~~~~~~~~~ Pull Phase ~~~~~~~~~~~~~ */
          // during a pull phase we use a dense representation 
          // of the frontier (bitmap; with bits set for vertices
          // in the frontier)
          convertToDenseFrontier(&CurrFrontier, V);

          t_for (d, 0, V);
          {
              w_if (Parent[d] == -1);
              {
                  t_for (i, InOffsets[d], InOffsets[d+1]);
                  {
                      w_if (CurrFrontier[Sources[i]] == 1);
                      {
                          Parent[d]   = Sources[i];
                          NextFrontier[d] = 1;
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

          copyFrontier(NextFrontier, CurrFrontier, V); 

          w_if (Stop[0] == 1);
          {
              l = V; //work-around for break statement
          }
          end();
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
      if (CurrFrontier.At(v) != -1)
      {
          ++connections;    
          if (CurrFrontier.At(v) > depth) 
          {
              depth = CurrFrontier.At(v);
          }
      }
  }
  whoop::T(3) << "Depth of BFS Tree =  " << depth << " and number of connected nodes = " << connections <<whoop::EndT;
  
  whoop::Done(StatsFileName);
}
