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
#include <map>
#include <algorithm>

using namespace whoop;

int MaxEdgesInTile( VecIn &InOffsets, VecIn &Sources, 
                     Tensor *OffsetsPerTile,
                     Tensor *TileEmpty, 
                     const int V, const int S0, const int D0, 
                     const int S1, const int D1 )
{
    int totalTiles = S1*D1;
    
    /* 
     * Keep track of number of edges per tile. Will be used for resizing SourcesPerTile later
     */
    int idx_wr[S1][D1]; 

    // Initialize the Per Tile Compressed Input
    for(int d1=0; d1<D1; d1++)
    {
        for(int s1=0; s1<S1; s1++)
        {
            TileEmpty->At({s1,d1}) = true;  // set that tile is empty
            idx_wr[s1][d1]         = 0;     // set the write pointer

            for(int d0=0; d0 < D0+1; d0++)  // initialize offsets to zero
            {
                
                OffsetsPerTile->At({s1,d1,d0}) = 0;
            }
        }
    }

    // First pass is to find the number of edges per tile  
    for( int d=0; d<V; d++) 
    {
        int o_start = InOffsets.At(d);
        int o_end   = InOffsets.At(d+1);
        
        for(int i=o_start; i<o_end; i++)
        {
            int s = Sources.At(i);
            
            int tile_s = s / S0;
            int tile_d = d / D0;

            TileEmpty->At({tile_s,tile_d}) = false;
            idx_wr[tile_s][tile_d]++;
         }
    }
   
    // finding the maximum number of edges in a tile
    int maxEdgesPerTile = 0;
    for (int d1 = 0; d1 < D1; d1++) 
    {
        for (int s1 = 0; s1 < S1; s1++) 
        {
            if (maxEdgesPerTile < idx_wr[s1][d1])
            {
                maxEdgesPerTile = idx_wr[s1][d1];
            }
        }
    }

    return maxEdgesPerTile;
}


void GenerateGraphInputsPerTile( VecIn &InOffsets, VecIn &Sources, 
                                 Tensor *OffsetsPerTile, Tensor *SourcesPerTile, 
                                 const int V, const int S0, const int D0, 
                                 const int S1, const int D1 )
{
    int totalTiles = S1*D1;
    

    //Populate the contets of the Per tile offsets and sources
    for (int d1 = 0; d1 < D1; d1++) 
    {
        for (int d0 = 0; d0 < D0; d0++) 
        {
            int degrees[S1] {0};
            int d = d1 * D0 + d0;
            if (d < V) 
            {
                int o_start = InOffsets.At(d);
                int o_end   = InOffsets.At(d+1);
        
                for (int i=o_start; i < o_end; i++) 
                {
                    int s = Sources.At(i);

                    int tile_s      = s / S0;
                    int curr_offset = OffsetsPerTile->At( {tile_s,d1,d0} );

                    SourcesPerTile->At( {tile_s,d1,curr_offset+degrees[tile_s]} )  = s; 

                    degrees[tile_s] += 1; 
                }

                for (int s1 = 0; s1 < S1; s1++) 
                {
                    OffsetsPerTile->At( {s1,d1,d0+1} ) = OffsetsPerTile->At( {s1,d1,d0} ) + degrees[s1];
                }
            }
        }
    }
    
}



int main(int argc, char** argv)
{

  VecIn  InOffsets("inoffsets");
  VecIn  Sources("sources");
  VecOut Parent("parent");              //stores parent of each vertex
  #if 0
  Vec    Frontier("frontier");          //stores level (in bfs-tree) of each vertex
  Vec    Frontier_SRC("frontier_src");  //stores Frontier[src] accesses 
  Vec    Frontier_DST("frontier_dst");  //stores Frontier[dst] accesses 
  #endif

  Vec Frontier("frontier");

  Var    Stop("Stop"); //stopping condition
  
  
  int S_TILE_SIZE = 32; //16;
  int D_TILE_SIZE = 32; //16;

  std::string StatsFileName = "stats.txt";

  AddOption( &StatsFileName, "stats", "Stats File Name");
  AddOption( &S_TILE_SIZE, "src_tile_size", "Source      Tile Size");
  AddOption( &D_TILE_SIZE, "dst_tile_size", "Destination Tile Size");

  whoop::Init(argc, argv);

  int numVertices  = InOffsets.Size()-1;
  int numEdges     = Sources.Size();

  const int V      = numVertices;  
  const int S      = numVertices;  
  const int D      = numVertices;  

  int S0 = S_TILE_SIZE; 
  int D0 = D_TILE_SIZE;

  if (S_TILE_SIZE == -1) 
  {
      S0 = V;
      S_TILE_SIZE = 0;
  }
  if (D_TILE_SIZE == -1)
  {
      D0 = V;
      D_TILE_SIZE = 0;

  }

  if( S0 > V ) S0 = V;
  if( D0 > V ) D0 = V;  

  const int S1 = (S % S0) ? (S/S0+1) : (S/S0);
  const int D1 = (D % D0) ? (D/D0+1) : (D/D0);

  const int TotalTiles = S1*D1;

  std::cout<<std::endl<<std::endl;
  std::cout<<"BFS Pull - Tiled Implementation (SRC-DST) "<<std::endl;
  std::cout<<"==========================================="<<std::endl;
  std::cout<<std::endl;
  std::cout<<"Graph Attributes:"<<std::endl;
  std::cout<<"\tGraph Name:                "<<InOffsets.GetFileName()<<std::endl;
  std::cout<<"\tNumber of Vertices:        "<<V<<std::endl;
  std::cout<<"\tNumber of Edges:           "<<numEdges<<std::endl;
  std::cout<<std::endl;
  std::cout<<"Tiling Parameters:"<<std::endl;
  std::cout<<"\tTile Size  (S0):           "<<S0<<std::endl;
  std::cout<<"\tTile Size  (D0):           "<<D0<<std::endl;
  std::cout<<std::endl;
  std::cout<<"\t# of Tiles (S1):           "<<S1<<std::endl;
  std::cout<<"\t# of Tiles (D1):           "<<D1<<std::endl;
  std::cout<<"\tTot # of Tiles:            "<<TotalTiles<<std::endl;
  std::cout<<std::endl;

  std::cout<<std::endl;

  Tensor TileEmpty("TileEmpty");
  Tensor TileInOffsets("TileInOffsets");
  Tensor TileSources("TileSources"); 

  TileEmpty.Resize({S1,D1});
  TileInOffsets.Resize({S1,D1,D0+1});               
  int maxEdges = MaxEdgesInTile (InOffsets, Sources, &TileInOffsets, &TileEmpty, V, S0, D0, S1, D1 );
  TileSources.Resize({S1,D1,maxEdges});  // While better than sizing each tile with |E| edges, this is still memory inefficient. 

  GenerateGraphInputsPerTile (InOffsets, Sources, &TileInOffsets, &TileSources, V, S0, D0, S1, D1); 
    
  // Initialize app data structures
  Parent.Resize(V);
  Frontier.Resize(V);
  #if 0 
  Frontier_SRC.Resize(S0);
  Frontier_DST.Resize(1);
  #endif
  for(int v=0; v < V; v++) 
  {
      Parent.At(v)   = -1; 
      Frontier.At(v) = -1; 
  }
  
  const int src      = 0;   //TODO: Add support to find "good" sources
  Parent.At( src )   = src;
  Frontier.At( src ) = 0;   //source node is at root of bfs tree

  TensorPort Frontier2(&Frontier);

#define REF 0
#if REF

  int quit = 1;

  for (int l = 0; l < V; l++) {
    for (int d1 = 0; d1 < D1; d1++) {
      for (int s1 = 0; s1 < S1; s1++) {
        if (TileEmpty.At({s1,d1}) == 0) {
          for (int d0 = 0; d0 < D0; d0++) {
            int d = d0 + d1 * D0;
            if (d < V && Parent.At(d) == -1) {
              int o_s = TileInOffsets.At({s1,d1,d0});
              int o_e = TileInOffsets.At({s1,d1,d0+1});
              for (int i = o_s; i < o_e; i++) {
                int s = TileSources.At({s1,d1,i});

                if (Frontier.At(s) == l) {
                  Parent.At(d)   = s;
                  Frontier.At(d) = l+1;
                  quit           = 0;
                  break;
                }
              }
            }
          }
        }
      }
    }
    if (quit == 1) {
      break;
    }
  }


#else

  Var x("x");
  Var n("n");
  Var s("s");
  Var d("d");
  Var i("i");
  Var l("l");       // depth of BFS tree

  Var s0("s0");
  Var d0("d0");
  Var s1("s1");
  Var d1("d1");
  Var s_ind("s_ind");

  Var o_s("offset_start");
  Var o_e("offset_end");

 
  t_for (l, 0, V);
  {
      Stop = 1;
      t_for (s1, 0, S1);
      {
          Frontier.AddTileLevel(std::max(S_TILE_SIZE,1));
          t_for (d1, 0, D1);
          {
              w_if (TileEmpty[s1][d1] == 0);
              {
                  // intra-tile iteration begins
                  
                  #if 0
                  /*
                   * DEPRECATED after TensorPort extension
                   *
                   * Explicit data orchestration. Copying the src indexed
                   * parts of Frontier to a different data type. 
                   *
                   * NOTE: no need to copy back the contents of Frontier_SRC
                   * to Frontier because bfs ensures that Frontier[s] and 
                   * Frontier[d] will never alias.
                   *
                   * NOTE2: The following data orchestration is excessive. 
                   * We are bringing in Frontier[src] that may not be accessed
                   */
                  Frontier_SRC.AddTileLevel(S_TILE_SIZE); 
                  t_for (s0, 0, S0);
                  {
                      w_if (s0 + s1*S0 < V);
                      {
                          Frontier_SRC[s0] = Frontier[s0 + s1*S0];
                      }
                      end();
                  }
                  end();
                  #endif
                    
                  Frontier2.AddTileLevel(std::max(D_TILE_SIZE,1));
                  Parent.AddTileLevel(1); //zero-reuse across tiles
                  /* 
                   * There is reuse in the TileInOffsets[][][d]. Since the 
                   * second element TileInOffsets[][][d+1] will be reused
                   * in the next iteration (when d changes).
                   */
                  TileInOffsets.AddTileLevel(2);
                  TileSources.AddTileLevel(1);  //zero-reuse within a tile

                  t_for (d0, 0, D0);
                  {
                      d = d0 + d1 * D0;
                      
                      #if 0
                      //explicitly copying contents of Frontier[dst] to different data type
                      Frontier_DST[0] = Frontier[d];
                      #endif

                      o_s = TileInOffsets[s1][d1][d0];
                      o_e = TileInOffsets[s1][d1][d0+1];

                      w_if (d < V && o_s != o_e);
                      {
                          w_if (Parent[d] == -1);
                          {
                              t_for (i, o_s, o_e);
                              {
                                  s = TileSources[s1][d1][i];
                                 
                                  #if 0
                                  s_ind = s - S0*s1;
                                  #endif
                                  //inner-loop: App logic
                                  w_if (Frontier[s] == l);
                                  {
                                      Parent[d]    = s;
                                      Frontier2[d] = l + 1;
                                      Stop         = 0;
                                      i = o_e; // break once parent is found
                                  }
                                  end();
                              }
                              end();
                          }
                          end();
                      }
                      end();
                      #if 0
                      //explicitly copying the contents of the datatype back to Frontier[dst]
                      Frontier[d] = Frontier_DST[0]; 
                      #endif
                  }
                  end();
              }
              end();
          }
          end();
      }
      end();

      // break out of the far loop if we did not do any updates this iteration
      // later we should be able to fix this with a while loop
      w_if (Stop == 1);
      {
          l = V; // work-around for break statement
      }
      end();      
  }
  end();

  std::cout<<std::endl;
  std::cout<< "RUNNING WHOOP..." <<std::endl;
  whoop::Run();
  std::cout<< "DONE WHOOP..." <<std::endl;
#endif
  

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
  std::cout << "Depth of BFS Tree =  " << depth << " and number of connected nodes = " << connections << std::endl;

  whoop::Done(StatsFileName);
}
