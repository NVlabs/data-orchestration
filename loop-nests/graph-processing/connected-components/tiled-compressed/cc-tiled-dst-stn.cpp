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

using namespace whoop;

int MaxEdgesInTile( VecIn &InOffsets, VecIn &Sources, 
                    Tensor *TileOffsets,
                    Tensor *TileEmpty, Tensor *TileWrPtr, double &avgDegree,
                    const int V, const int S0, const int D0, 
                    const int S1, const int D1 )
{
    long degree_sum = 0;
    
    int totalTiles = S1*D1;
    
    /* 
     * Keep track of number of edges per tile. Will be used for resizing TileSources later
     */

    // Initialize the Per Tile Compressed Input
    TileWrPtr->FillVal( 0 );
    TileEmpty->FillVal( 1 );
    TileOffsets->FillVal( 0 );

    // First pass is to find the number of edges per tile  
    for( int d=0; d<V; d++) 
    {
        int o_start = InOffsets.At(d);
        int o_end   = InOffsets.At(d+1);

        degree_sum += o_end - o_start;
        
        for(int i=o_start; i<o_end; i++)
        {
            int s = Sources.At(i);
            
            int tile_s = s / S0;
            int tile_d = d / D0;

            TileEmpty->At({tile_s,tile_d}) = false;
            TileWrPtr->At({tile_s,tile_d})++;
         }
    }
    
    // finding the maximum number of edges in a tile
    int maxEdgesPerTile = 0;
    for (int d1 = 0; d1 < D1; d1++) 
    {
        for (int s1 = 0; s1 < S1; s1++) 
        {
            if (maxEdgesPerTile < TileWrPtr->At({s1,d1}))
            {
                maxEdgesPerTile = TileWrPtr->At({s1,d1});
            }
        }
    }

    avgDegree = (double)degree_sum/(double)V;

    return maxEdgesPerTile;
}

// This is what a Pattern Generator Would Perhaps Wanna Do
void GetTile( VecIn &InOffsets,  
              VecIn &Sources, 
              Tensor *TileOffsets, 
              Tensor *TileSources, 
              Tensor *TileEmpty, 
              Tensor *TileWrPtr, 
              Tensor *TileVWNcnt, 
              Tensor *TileVWN, 
              int V, 
              int S0, 
              int D0, 
              int S1, 
              int D1,
              int s1,
              int d1 )
{

    // We basically want the connectivity information for tile s1,d1
    if( s1 >= S1 || d1 >= D1 ) 
    {
        std::cout<<"You have provided an invalid tile:  (s1="<<s1<<",d1="<<d1<<")"<<std::endl;
        exit(0);
    }

    // Since we are CSC, let's determine where we are going
    for(int d0=0; d0 < D0; d0++) 
    {
        int d = d1*D0 + d0;

        if (d < V) 
        {
            int o_start = InOffsets.At(d);
            int o_end   = InOffsets.At(d+1);

            if( o_start != o_end ) 
            {
                int curr_offset = TileOffsets->At( {s1,d1,d0} );

                // look at all the neighbors
                for (int i=o_start; i < o_end; i++) 
                {
                    int s = Sources.At(i);
                    
                    int tile_s      = s / S0;
                    
                    // Is the neighbor in the requested tile?  If so take it
                    if( tile_s == s1 ) 
                    {
                        // Either create a new compressed format, or instead
                        // just read the contents and send it to the CT

                    }
                    else if( i != o_start )
                    {
                         InOffsets.At(d) = i;

                        // break out of the find neighbors loop
                        i = o_end;
                    }
                }
            }
        }
    }
}

void GenerateGraphInputsPerTile( VecIn &InOffsets,  
                                 VecIn &Sources, 
                                 Tensor *TileOffsets, 
                                 Tensor *TileSources, 
                                 Tensor *TileEmpty, 
                                 Tensor *TileWrPtr, 
                                 Tensor *TileVWNcnt, 
                                 Tensor *TileVWN, 
                                 int V, int S0, int D0, int S1, int D1 )
{

#if 1    
    
    TileVWNcnt->FillVal(0);
    
    //Populate the contents of the Per tile offsets and sources
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
                    int curr_offset = TileOffsets->At( {tile_s,d1,d0} );

                    TileSources->At( {tile_s,d1,curr_offset+degrees[tile_s]} )  = s; 

                    degrees[tile_s] += 1; 
                }

                for (int s1 = 0; s1 < S1; s1++) 
                {
                    TileOffsets->At( {s1,d1,d0+1} ) = TileOffsets->At( {s1,d1,d0} ) + degrees[s1];

                    // track the vertices that have neighbors so we can skip
                    // acccesses to those that have no neighbors

                    if( degrees[s1] != 0 ) 
                    {
                        int wr_ptr = TileVWNcnt->At( {s1,d1} );
                        TileVWN->At( {s1,d1,wr_ptr} ) = d0;
                        TileVWNcnt->At( {s1,d1} ) += 1;
                    }
                }
            }
        }
    }
#else
    TileWrPtr->FillVal( 0 );
    TileEmpty->FillVal( 1 );
    TileOffsets->FillVal( 0 );


    // Make the tiles
    for( int d=0; d<V; d++) 
    {
        int o_start = InOffsets.At(d);
        int o_end   = InOffsets.At(d+1);
        
        for(int i=o_start; i<o_end; i++)
        {
            int s = Sources.At(i);
            
            int tile_s = s / S0;
            int tile_d = d / D0;

            int tile_num = tile_s*D1 + tile_d;
        
            int wi = TileWrPtr->At({tile_s,tile_d});

//             std::cout<<"Tile Num: "<<tile_num<<" s: "<<s<<" d: "<<d<<" tile_s: "<<tile_s<<" tile_d: "<<tile_d<<" write: "<<wi<<std::endl;

            TileEmpty->At({tile_s,tile_d}) = false;

            TileSources->At({tile_s,tile_d,wi}) = s;

            int d_idx = d % D0;
            
            // If this is the first time I am recording for this vertex, make
            // sure start and end bounds are the same
            if( TileOffsets->At({tile_s,tile_d,d_idx}) == TileOffsets->At({tile_s,tile_d,d_idx+1}) )
            {
                TileOffsets->At({tile_s,tile_d,d_idx}) = wi;

                // a corner case happens if some vertices do not exist in the tile, we need
                // to ensure they are padded from '0' default to current write-index so we don't
                // incorrectly use the neighbors of vertex that already started at 0
                int sup_i = d_idx-1;
                while( sup_i && TileOffsets->At({tile_s,tile_d,sup_i}) < wi )
                {
                    TileOffsets->At({tile_s,tile_d,sup_i}) = wi;
                    sup_i--;
                }
            }
            
            TileOffsets->At({tile_s,tile_d,d_idx+1}) = wi+1;
            TileWrPtr->At({tile_s,tile_d})++;
        }
    }

#endif

    whoop::T(4)<<"############################# BASELINE #####################"<<whoop::EndT;
    whoop::T(4)<<"Offsets_Array: ";
    for(int d=0; d<V+1; d++) 
    {
        whoop::T(4)<<InOffsets.At(d)<<" ";
    }
    whoop::T(4)<<whoop::EndT;

    whoop::T(4)<<"Sources_Array: ";
    for(int i=0; i<Sources.Size(); i++) 
    {
        whoop::T(4)<<Sources.At(i)<<" ";
    }
    whoop::T(4)<<whoop::EndT;
    whoop::T(4)<<whoop::EndT;


    long src_accesses;
    long src_misses;
    long dest_accesses;
    long dest_misses;
    
    std::map<int,long> uniq;
    // Print Out The Tiles
    for(int d1=0; d1<D1; d1++)
    {
        for(int s1=0; s1<S1; s1++)
        {
            dest_accesses = 0;
            dest_misses   = 0;
            
            whoop::T(4)<<"Offsets_Array: ";
            for(int d_idx=0; d_idx<D0+1; d_idx++) 
            {
                if( d_idx<D0 && (TileOffsets->At({s1,d1,d_idx}) != TileOffsets->At({s1,d1,d_idx+1})) )
                {
                    dest_accesses++;

//                     if( uniq.find(d_idx) == uniq.end() ) 
//                     {
//                         dest_misses++;
//                         uniq.insert(std::pair<int,long>(d_idx,0));
//                     }
//                     uniq[d_idx]++;
                    
                }
                    
                whoop::T(4)<<TileOffsets->At({s1,d1,d_idx})<<" ";
            }
       
            whoop::T(4)<<whoop::EndT;

            src_accesses = 0;
            src_misses   = 0;
        
            whoop::T(4)<<"Sources_Array: ";
            for(int i=0; i<TileWrPtr->At({s1,d1}); i++) 
            {
                int src = TileSources->At({s1,d1,i});

                whoop::T(4)<<src<<" ";

                src_accesses++;
                if( uniq.find(src) == uniq.end() ) 
                {
                    src_misses++;
                    uniq.insert(std::pair<int,long>(src,0));
                }
                uniq[src]++;
            }

            whoop::T(4)<<whoop::EndT;
            whoop::T(4)<<whoop::EndT;
            std::cout<<"############################# TILE: ("<<s1<<","<<d1<<") NNZ: "<<TileWrPtr->At({s1,d1})
                     <<" domain[s]_access: "<<src_accesses<<" domain[s]_first_miss: "<<src_misses
                     <<" domain[d]_access: "<<dest_accesses
                     <<" domain[d]_first_miss: "<<dest_misses
                     <<" #####################"<<std::endl;

            uniq.clear();
        }
    }
}



int main(int argc, char** argv)
{

  VecIn  InOffsets("inoffsets");
  VecIn  Sources("sources");
  Vec    OffsetsCopy;
  
  VecOut     domain("domain");
  TensorPort domain_dst( &domain );

  int GRANULARITY        = 1;  // buffet granularity
  int S_TILE_SIZE        = 32; 
  int D_TILE_SIZE        = 32; 
  int TILE_SIZE_OVERRIDE = 0;
  
  std::string StatsFileName = "stats.txt";

  AddOption( &StatsFileName, "stats", "Stats File Name");
  AddOption( &S_TILE_SIZE, "src_tile_size", "Source      Tile Size");
  AddOption( &D_TILE_SIZE, "dst_tile_size", "Destination Tile Size");
  AddOption( &TILE_SIZE_OVERRIDE, "override_tile_size", "Override The Tile Size");
  AddOption( &GRANULARITY, "granularity", "fetch granularity (in item count) from next level of memory");

  whoop::Init(argc, argv);

  int numVertices  = InOffsets.Size()-1;
  int numEdges     = Sources.Size();

  const int V      = numVertices;  
  const int S      = numVertices;  
  const int D      = numVertices;  

  bool amTiled = true;
  
  int S0 = (S_TILE_SIZE == 0) ? V : S_TILE_SIZE;
  int D0 = (D_TILE_SIZE == 0) ? V : D_TILE_SIZE;

  if( (S_TILE_SIZE==0 ) && (D_TILE_SIZE == 0) ) 
  {
      amTiled = false;
  }

  if( S0 > V ) S0 = V;
  if( D0 > V ) D0 = V;  

  const int S1 = (S % S0) ? (S/S0+1) : (S/S0);
  const int D1 = (D % D0) ? (D/D0+1) : (D/D0);

  const int TotalTiles = S1*D1;

  int TILE_SIZE         = ((S0==V)?16:S0*2)+((D0==V)?16:D0*2);
  int OFFSETS_TILE_SIZE = 16;
  int SOURCES_TILE_SIZE = 16;

  if( !amTiled ) 
  {
      TILE_SIZE             = 16;
      OFFSETS_TILE_SIZE     = 16;
      SOURCES_TILE_SIZE     = 16;
  }

  if( TILE_SIZE_OVERRIDE ) 
  {
      TILE_SIZE = 2*TILE_SIZE_OVERRIDE;
  }

  Tensor TileEmpty("TileEmpty");
  Tensor TileWrPtr("TileWrPtr");

  Tensor TileVWN("TileVWN");    //TileVerticesWithNeighbors
  Tensor TileVWNcnt("TileVWNcnt"); //TileVerticesWithNeighbors

  Tensor TileOffsets("TileOffsets");
  Tensor TileSources("TileSources");

  TileEmpty.Resize({S1,D1});
  TileWrPtr.Resize({S1,D1});
  TileOffsets.Resize( {S1,D1,D0+1});

  TileVWN.Resize( {S1,D1,D0});
  TileVWNcnt.Resize( {S1,D1});

  // Make A Copy Of The Original Offsets In Case We Need It Back
  OffsetsCopy.Resize(V+1);
  for(int v=0; v<V+1; v++)
  {
      OffsetsCopy.At(v) = InOffsets.At(v);
  }


  double avgDegree;
  int maxTileEdges = MaxEdgesInTile (InOffsets, Sources, &TileOffsets, &TileEmpty, &TileWrPtr, avgDegree, V, S0, D0, S1, D1 );
  
  TileSources.Resize( {S1,D1,maxTileEdges} );

  GenerateGraphInputsPerTile( InOffsets, Sources, &TileOffsets, &TileSources, &TileEmpty, &TileWrPtr, &TileVWNcnt, &TileVWN, V, S0, D0, S1, D1 ); 
    
  // Initialize domain
  domain.Resize( V );
  for(int v=0; v < V; v++) 
  {
      domain.At(v) = v;
  }


  std::cout<<std::endl<<std::endl;
  std::cout<<"Connected Components - Tiled Implementation"<<std::endl;
  std::cout<<"==========================================="<<std::endl;
  std::cout<<std::endl;
  std::cout<<"Graph Attributes:"<<std::endl;
  std::cout<<"\tGraph Name:                "<<InOffsets.GetFileName()<<std::endl;
  std::cout<<"\tNumber of Vertices:        "<<V<<std::endl;
  std::cout<<"\tNumber of Edges:           "<<numEdges<<std::endl;
  std::cout<<"\tAverage Degree:            "<<avgDegree<<std::endl;
  std::cout<<std::endl;
  std::cout<<"Tiling Parameters:"<<std::endl;
  std::cout<<"\tAm Tiled?                  "<<(amTiled?"YES":"NO")<<std::endl;
  std::cout<<std::endl;
  std::cout<<"\tTile Size  (S0):           "<<S0<<std::endl;
  std::cout<<"\tTile Size  (D0):           "<<D0<<std::endl;
  std::cout<<std::endl;
  std::cout<<"\t# of Tiles (S1):           "<<S1<<std::endl;
  std::cout<<"\t# of Tiles (D1):           "<<D1<<std::endl;
  std::cout<<"\tTot # of Tiles:            "<<TotalTiles<<std::endl;
  std::cout<<std::endl;
  std::cout<<"Buffer Parameters:"<<std::endl;
  std::cout<<"\tBuffer Size For Tile:      "<<TILE_SIZE<<std::endl;
  std::cout<<"\tBuffer Size Granularity:   "<<GRANULARITY<<std::endl;

  std::cout<<std::endl;


#define REF 0
#if REF

  int indir_lds_sum = 0;
  int offset_refs = 0;
  
  for(int m=0; m<V; m++)
  {
      for(int d1=0; d1<D1; d1++)
      {
          for(int s1=0; s1<S1; s1++)
          {
              if( (TileEmpty.At({s1,d1}) == false) ) 
              {
                  for(int d0=0; d0<D0; d0++)
                  {
                      int d = d0 + d1*D0;
                      if( d < V ) 
                      {
                          offset_refs += 2;

                          int o_s = TileOffsets.At({s1,d1,d0});
                          int o_e = TileOffsets.At({s1,d1,d0+1});

                          int indir_loads = ((o_e-o_s)>0?(o_e-o_s):(0));
                          indir_lds_sum += indir_loads;
                          
                          for(int i=o_s; i<o_e; i++) 
                          {
                              int s = TileSources.At({s1,d1,i}); 
                              if( domain.At(d) > domain.At(s) ) 
                              {
                                  domain.At(d) = domain.At(s);
//                               std::cout<<"Tile: "<<tilenum<<" Updated: dst_domain["<<d<<"] To That Of src_domain["<<s<<"] Value: "<<domain.At(s)<<std::endl;
                              }
                      
                          }
                      }
                  }
              }
          }
      }
  }

  std::cout<<"Indirect Loads Sum: "<<indir_lds_sum<<std::endl;
  std::cout<<"Offset Refs: "<<offset_refs<<std::endl;

#else

  // For undirected graphs, CSR and CSC formats are identical
  // for destination stationary, we are going to need CSC version

  Var x("x");
  Var n("n");
  Var m("m");
  Var s("s");
  Var d("d");
  Var i("i");

  Var s0("s0");
  Var d0("d0");
  Var s1("s1");
  Var d1("d1");

  Var o_s("offset_start");
  Var o_e("offset_end");

  Var tilenum("tilenum");

  Var d_val("d_val");
  Var s_val("s_val");  

  Var    iters("iters");
  Var    did_updates_this_iter("did_updates_this_iter");
  Var    update_dval("update_dval");

  Var numVwithneighbors("numVwithneighbors");
  Var wn("wn");
  Var ws("ws");
  
  iters = 0;
  
  t_for(m, 0, V); // replace with while later
  {
      // track iteration count for measuring covergence speed
      iters += 1;

      // in case we want to break out of outer loop if no work done in an iteration
      did_updates_this_iter = 0;

      t_for(d1, 0, D1);
      {
<<<<<<< HEAD
          domain.AddTileLevel(TILE_SIZE, GRANULARITY);
          domain_dst.AddTileLevel(16, GRANULARITY);

          TileOffsets.AddTileLevel(OFFSETS_TILE_SIZE, GRANULARITY);
          TileVWN.AddTileLevel(OFFSETS_TILE_SIZE, GRANULARITY);
          TileSources.AddTileLevel(SOURCES_TILE_SIZE, GRANULARITY);
=======
          t_for(s1, 0, S1);
          {
              domain.AddTileLevel(TILE_SIZE, TILE_SIZE, GRANULARITY);

              TileVWN.AddTileLevel(OFFSETS_TILE_SIZE, OFFSETS_TILE_SIZE, GRANULARITY);
              TileOffsets.AddTileLevel(OFFSETS_TILE_SIZE, OFFSETS_TILE_SIZE, GRANULARITY);
              TileSources.AddTileLevel(SOURCES_TILE_SIZE, SOURCES_TILE_SIZE, GRANULARITY);
>>>>>>> b79b077b01a8166d5fbc8d78f9c7a4c77703709b

          t_for(s1, 0, S1);
          {
              w_if( TileEmpty[s1][d1]==0 );
              {

#define USE_VWN 1
#if USE_VWN
                  ws = 0;
                  numVwithneighbors = TileVWNcnt[s1][d1];
                  t_for(wn, ws, numVwithneighbors); 
                  {
                      
                      d0 = TileVWN[s1][d1][wn];
#else                  
                  
                  t_for(d0, 0, D0);
                  {                      
#endif

                      domain_dst.AddTileLevel(16, 16, GRANULARITY);

                      // Analytical Analysis Of Inner Loop
                      // We are destination stationary so we have:
                      //     1 read for domain[d]
                      //     1 write to domain[d]
                      //     1 read for each neighbor of d (i.e. domain[s])

                      o_s = TileOffsets[s1][d1][d0];
                      o_e = TileOffsets[s1][d1][d0+1];

//                       w_if( (d < V) && (o_s != o_e) );
                      w_if( o_s != o_e );
                      {
                          d           = d0 + d1*D0;
                          d_val       = domain_dst[d];
                          update_dval = 0;

                          t_for(i, o_s, o_e);
                          {
                              s = TileSources[s1][d1][i];

                              s_val = domain[s];

                              w_if( (d_val > s_val ) );
                              {
                                  d_val       = s_val;
                                  update_dval = 1;
                              }
                              end();
                          }
                          end();

                          w_if( update_dval == 1 );
                          {
                              did_updates_this_iter = 1;
                              domain_dst[d]         = d_val;
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

      // break out of the far loop if we did not do any updates this iteration
      // later we should be able to fix this with a while loop
      w_if (did_updates_this_iter != 1);
      {
          m = V; 
      }
      end();      
  }
  end();

  std::cout<<std::endl;
  std::cout<< "RUNNING WHOOP..." <<std::endl;
  whoop::Run();
  std::cout<< "DONE WHOOP..." <<std::endl;

#endif

  // Finding number of components
  std::map<int, int> components;
  for (int v = 0; v < V; v++) 
  {
      int compID = domain.At(v);
      if (components.find(compID) == components.end()) 
          components.insert(std::pair<int, int>(compID, 0));
      components[compID]++;
  }
  std::cout<<std::endl;
  std::cout<<std::endl;
  std::cout<<"Connected Components Output:"<<std::endl;
  std::cout<<"\tTotal number of components = " << components.size() <<std::endl;
  std::cout<<"\tIterations until convergence = " << iters.Access() << std::endl;
  std::cout<<std::endl;

  /* Sanity check - the above loop nest assumes that the outer loop can do a maximum of V iterations */
  assert(iters.Access() < V && "[ERROR] Need to run components for more iterations");

  whoop::Done(StatsFileName);
}
