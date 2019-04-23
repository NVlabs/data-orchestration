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
#include <map>
#include <algorithm>

using namespace whoop;

int MaxEdgesInTile( VecIn &InOffsets, VecIn &Sources, 
                     Tensor *OffsetsPerTile, Tensor *ValidOffsetsPerTile,
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
                ValidOffsetsPerTile->At({s1,d1,d0}) = 0;
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
                                 Tensor *OffsetsPerTile, Tensor *ValidOffsetsPerTile, 
                                 Tensor *SourcesPerTile, 
                                 const int V, const int S0, const int D0, 
                                 const int S1, const int D1 )
{
    int totalTiles = S1*D1;
    

    //Populate the contents of the Per-tile offsets and sources
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
                    if (degrees[s1] != 0) 
                    {
                        int currPos = ValidOffsetsPerTile->At({s1, d1, D0}); //the last index holds the number of vertices with > 1 neighbors
                        ValidOffsetsPerTile->At({s1, d1, currPos}) = d0;
                        ++currPos;
                        ValidOffsetsPerTile->At({s1, d1, D0}) = currPos;
                    }
                }
            }
        }
    }
    
    #if 0
    std::cout <<"############################# BASELINE #####################"<< std::endl;
    std::cout <<" Offsets Array: ";
    for(int d=0; d<V+1; d++) 
    {
        auto x = InOffsets.At(d);
        std::cout << x <<" ";
    }
    std::cout << "\n";

    std::cout <<" Sources Array: ";
    for(int i=0; i<Sources.Size(); i++) 
    {
        auto x = Sources.At(i);
        std::cout << x <<" ";
    }
    std::cout << std::endl;


    // Print Out The Tiles
    for(int d1=0; d1<D1; d1++)
    {
        for(int s1=0; s1<S1; s1++)
        {
            std::cout <<"############################# TILE: ("<<s1<<","<<d1<<") #####################"<<std::endl;
            std::cout <<" Offsets Array: ";
            for(int d0 = 0; d0 < D0+1; d0++) 
            {
                std::cout <<OffsetsPerTile->At({s1,d1,d0})<<" ";
            }
       
            std::cout << std::endl;
        
            std::cout <<" Sources Array: ";
            for(int d0 = 0; d0 < D0; d0++) 
            {
                for(int i = OffsetsPerTile->At({s1,d1,d0}); i < OffsetsPerTile->At({s1,d1,d0+1}); i++) 
                {
                    std::cout <<SourcesPerTile->At({s1,d1,i})<<" ";
                }
            }
            std::cout << std::endl;
        }
    }
    #endif
}

long countZeroTiles (Tensor* tileEmpty, const int S1, const int D1) 
{
    long zeroTiles {0};
    for (int s1 = 0; s1 < S1; ++s1)
    {
        for (int d1 = 0; d1 < D1; ++d1)
        {
            zeroTiles += tileEmpty->At({s1,d1});
        }
    }
    return zeroTiles;
}



int main(int argc, char** argv)
{

  VecIn  InOffsets("inoffsets");
  VecIn  Sources("sources");
  VecOut InDegrees("indegrees");      //stores parent of each vertex
  Vec    vData("vData");  //stores level (in bfs-tree) of each vertex

  
  int S_TILE_SIZE = 32; //16;
  int D_TILE_SIZE = 32; //16;

  std::string StatsFileName = "stats.txt";

  AddOption( &StatsFileName, "stats", "Stats File Name");
  AddOption( &S_TILE_SIZE, "src_tile_size", "Source      Tile Size");
  AddOption( &D_TILE_SIZE, "dst_tile_size", "Destination Tile Size");
  
  int BURST_SIZE = 1;
  AddOption(&BURST_SIZE, "burst_size", "Data Transfer Granularity");

  whoop::Init(argc, argv);

  int numVertices  = InOffsets.Size()-1;
  int numEdges     = Sources.Size();

  const int V      = numVertices;  
  const int S      = numVertices;  
  const int D      = numVertices;  
  const int NUM_ITERS = 10;

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
  std::cout<<"SpMV - Tiled Implementation (DST-SRC) "<<std::endl;
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
  #if 0 //deprecated
  std::cout<<"Buffer Parameters:"<<std::endl;
  std::cout<<"\tBuffer Size For Tile:      "<<TILE_SIZE<<std::endl;
  #endif

  std::cout<<std::endl;

  Tensor TileEmpty("TileEmpty");
  Tensor TileInOffsets("TileInOffsets");
  Tensor TileValidOffsets("TileValidOffsets"); //per-tile compression on the offsets array
  Tensor TileSources("TileSources"); 

  TileEmpty.Resize({S1,D1});
  TileInOffsets.Resize({S1,D1,D0+1});               
  TileValidOffsets.Resize({S1,D1,D0+1}); // conservatively sized for maximum D0 valid indices. 
                                         // Last index to track number of valid indices in the tile
  int maxEdges = MaxEdgesInTile (InOffsets, Sources, &TileInOffsets, &TileValidOffsets, &TileEmpty, V, S0, D0, S1, D1 );
  TileSources.Resize({S1,D1,maxEdges});  // While better than sizing each tile with |E| edges, this is still memory inefficient. 

  GenerateGraphInputsPerTile (InOffsets, Sources, &TileInOffsets, &TileValidOffsets, &TileSources, V, S0, D0, S1, D1); 

  std::cout << "Number of zero-tiles: " << countZeroTiles(&TileEmpty, S1, D1) << std::endl;
    
  // Initialize app data structures
  InDegrees.Resize(V);
  vData.Resize(V);
  for(int v=0; v < V; v++) 
  {
      InDegrees.At(v) = 0; 
      vData.At(v)     = 1; 
  }
  

  Var s0("s0");
  Var d0("d0");
  Var wn("dn");  //iterator if there is offset compression
  Var ws("ws");
  Var numValid("numValid"); //number of vertices in tile with >1 neighbors
  Var s1("s1");
  Var d1("d1");

  Var l("l");
  Var d("d");
  Var i("i");
  Var s("s");
  Var o_s("offset_start");
  Var o_e("offset_end"); 
  

  t_for (l, 0, NUM_ITERS);
  {
      t_for (d1, 0, D1);
      {
          InDegrees.AddTileLevel(std::max(D_TILE_SIZE, D_TILE_SIZE, BURST_SIZE), BURST_SIZE);
          t_for (s1, 0, S1);
          {
              w_if (TileEmpty[s1][d1] == 0);
              {
                  vData.AddTileLevel(std::max(S_TILE_SIZE, S_TILE_SIZE, BURST_SIZE), BURST_SIZE);  //load once reuse across tiles in a row

                  /* 
                   * There is reuse in the TileInOffsets[][][d]. Since the 
                   * second element TileInOffsets[][][d+1] will be reused
                   * in the next iteration (when d changes).
                   */
                  TileInOffsets.AddTileLevel(BURST_SIZE, BURST_SIZE, BURST_SIZE);
                  TileSources.AddTileLevel(BURST_SIZE, BURST_SIZE, BURST_SIZE);  //zero-reuse within a tile
                  TileValidOffsets.AddTileLevel(BURST_SIZE, BURST_SIZE, BURST_SIZE);


#define USE_VWN 1
#ifdef USE_VWN
                  ws = 0;
                  numValid = TileValidOffsets[s1][d1][D0];
                  t_for (wn, ws, numValid);
                  {
                      d0 = TileValidOffsets[s1][d1][wn];
#else
                  t_for (d0, 0, D0);
                  {
#endif
                      d = d0 + d1*D0;

                      w_if (d < V);
                      {
                          o_s = TileInOffsets[s1][d1][d0];
                          o_e = TileInOffsets[s1][d1][d0+1];

                          t_for (i, o_s, o_e);
                          {
                              s = TileSources[s1][d1][i];
                              InDegrees[d] += vData[s];
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

  std::cout<<std::endl;
  std::cout<< "RUNNING WHOOP..." <<std::endl;
  whoop::Run();
  std::cout<< "DONE WHOOP..." <<std::endl;
  
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
