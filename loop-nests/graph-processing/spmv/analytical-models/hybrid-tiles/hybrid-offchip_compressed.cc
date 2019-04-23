// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <numeric>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include <cassert>


/*
GAP Benchmark Suite
Kernel: Square Tiles access count
Author: Vignesh Balaji 

For square 2d tiles, count the number of accesses to the two 
datatypes and also the adjacency matrix.

Note, the access pattern is 2d-dst-tile-stationary. 
After executing a tile, we move to the tile vertically below
the current tile.
Within each tile, we execute the vertices in a destination
stationary manner.

*/

using namespace std;

typedef float ScoreT;
typedef uint64_t stat_t;
const float kDamp = 0.85;

const int BURST_1  = 0;
const int BURST_16 = 1;

//round-up division
template <class T>
T ceilDivide(T num, T den)
{
    return (num + (den-1)) / den;
}

NodeID getHubs (const Graph &g)
{
  NodeID numHubs {0};
  NodeID degreeCutoff = g.num_edges_directed() / g.num_nodes();
  #pragma omp parallel for reduction (+ : numHubs)
  for (NodeID n = 0; n < g.num_nodes(); ++n) 
  {
    if (g.out_degree(n) > degreeCutoff)
    {
      ++numHubs;
    }
  }
  return numHubs;
}

/* 
 * Tile size parameter is the size of one dimension in a square tile setup
 * Aspect ratio parameter is the ratio of the dst tile / src tile
 */
pvector<ScoreT> hybridTiles(const Graph &g, int tile_size, int aspectRatio)
{
  NodeID numVertices = g.num_nodes();
  //tiles sizes for hub vertices (wide tiles)
  NodeID totalBufferSz = 2 * tile_size;
  NodeID hub_sTile_size = totalBufferSz - 16; 
  NodeID hub_dTile_size = g.num_nodes(); 

  //tiles sizes for tail vertices (narrow tiles)
  NodeID tail_sTile_size = (aspectRatio * totalBufferSz) / (1 + aspectRatio);
  NodeID tail_dTile_size = (totalBufferSz) - tail_sTile_size;
  #if 0
  NodeID tail_sTile_size = tile_size; 
  NodeID tail_dTile_size = tile_size; 
  #endif

  //caculate number of tiles
  #if 0
  NodeID numHubs      = (10 * g.num_nodes()) / 100;  //10% of the vertices cover a majority of edges
  #endif
  NodeID numHubs      = getHubs(g); 
  NodeID hub_sTiles   = numHubs / hub_sTile_size;
  if (hub_sTiles == 0) 
  {
    hub_sTiles = 1;
    hub_sTile_size = numHubs;
  }
  NodeID hub_dTiles   = ceilDivide(numVertices, hub_dTile_size);
  assert(hub_dTiles == 1); 
  NodeID numHubVertices = (hub_sTiles * hub_sTile_size);
  NodeID tail_sTiles  = ceilDivide(numVertices - numHubVertices, tail_sTile_size);
  NodeID tail_dTiles  = ceilDivide(numVertices, tail_dTile_size);

  // Tile dimensions
  std::cout << "Hub.sTiles  = " << hub_sTiles << " (" << hub_sTile_size << ")" << std::endl;
  std::cout << "Hub.dTiles  = " << hub_dTiles << " (" << hub_dTile_size << ")" << std::endl;
  std::cout << "Tail.sTiles = " << tail_sTiles << " (" << tail_sTile_size << ")" <<  std::endl;
  std::cout << "Tail.dTiles = " << tail_dTiles << " (" << tail_dTile_size << ")" <<   std::endl;
  std::cout << "**************\n";


  //Tile stats
  stat_t numZeroTiles {0};

  /* Offchip access counts for different data types - for burst size of 1 and 16 */
  stat_t srcData[2] = {0};
  stat_t dstData[2] = {0};
  stat_t tileEmpty[2]       = {0}; 
  stat_t tileOffsetsList[2] = {0}; //compressed representation of vertices with in-degree > 0
  stat_t tileInOffsets[2]   = {0};
  stat_t tileSources[2]     = {0};

  const int chunkSz = std::max(g.num_nodes() / 50, 16l);

  /* tile bounds */
  NodeID dTiles;
  NodeID sTiles;
  NodeID dTile_size;
  NodeID sTile_size;

  /* Pass 1 - executing in 1d tile fashion */
  dTiles     = hub_dTiles;
  dTile_size = hub_dTile_size;
  sTiles     = hub_sTiles;
  sTile_size = hub_sTile_size;

  #pragma omp parallel for schedule (dynamic, chunkSz)
  for (int sTile = 0; sTile < sTiles; ++sTile)
  {
    #pragma omp atomic
    ++tileEmpty[BURST_1];

    // keep track of number of consecutive dst vertices with in_degree > 0
    bool nonNullNeighbor {false};
    stat_t contiguousVertices {0};
    std::set<NodeID> inOffsetWindow;

    // populate all sources in the tile
    std::vector<NodeID> allSources;
    stat_t numNullVtx = 0; //number of empty columns in a tile (wasted adjacency read)
    for (NodeID dst = 0; dst < g.num_nodes(); ++dst)
    {
      NodeID srcStart = sTile * sTile_size;
      NodeID srcEnd   = (sTile + 1) * sTile_size;
      auto sources    = g.in_neighs_in_range(dst, srcStart, srcEnd);
      if (sources.size() == 0)
      {
        ++numNullVtx;
        nonNullNeighbor = false;
      }
      else
      {
        #pragma omp atomic
        ++tileOffsetsList[BURST_1]; //we access the list only when vtx.in_degree > 0
        if ((dst > 0) && (nonNullNeighbor == true))
        {
          ++contiguousVertices; 
        }
        inOffsetWindow.insert(dst / 16);
        nonNullNeighbor = true;
      }
      allSources.insert(allSources.end(), sources.begin(), sources.end());
    }

    // Modeling the cost in-offsets access (BURST_1)
    #pragma omp atomic
    tileInOffsets[BURST_1] += (2 * (dTile_size - numNullVtx)) - contiguousVertices;
    #pragma omp atomic
    tileInOffsets[BURST_16] += inOffsetWindow.size();

    // dst data is accessed similar as above (no reuse of dst data across tiles)
    #pragma omp atomic
    dstData[BURST_1] += (dTile_size - numNullVtx);
    #pragma omp atomic
    dstData[BURST_16] += inOffsetWindow.size();
  
    // find the unique sources in the tile
    std::set<NodeID> uniqSources (allSources.begin(), allSources.end());

    // This tile was not empty
    if (uniqSources.size() != 0) 
    {
      #pragma omp atomic
      srcData[BURST_1] += uniqSources.size();
      #pragma omp atomic
      tileSources[BURST_1] += allSources.size();

      // find number of BURST-16 windows required for uniqSources
      std::set<NodeID> uniqWindows;
      for (auto src : uniqSources)
      {
        uniqWindows.insert(src / 16); 
      }
      #pragma omp atomic
      srcData[BURST_16] += uniqWindows.size();
    }
  }

  /* pass 2 - square tiles for non-hub vertices */
  dTiles     = tail_dTiles;
  dTile_size = tail_dTile_size;
  sTiles     = tail_sTiles;
  sTile_size = tail_sTile_size;
  /* Access counts - BURST_SZ = 1 and BURST_SZ = 16 are mixed */
  #pragma omp parallel for schedule(dynamic, chunkSz)
  for (int dTile = 0; dTile < dTiles; ++dTile) 
  {
    for (int sTile = 0; sTile < sTiles; ++sTile)
    {
      #pragma omp atomic
      ++tileEmpty[BURST_1]; //checked for each tile
      
      // keep track of number of consecutive dst vertices with in_degree > 0
      bool nonNullNeighbor {false};
      stat_t contiguousVertices {0};
      std::set<NodeID> inOffsetWindow;

      // populate all sources in the tile
      std::vector<NodeID> allSources;
      stat_t numNullVtx = 0; //number of empty columns in a tile (wasted adjacency read)
      for (NodeID dst = dTile * dTile_size; dst < (dTile + 1) * dTile_size; ++dst)
      {
        if (dst >= g.num_nodes())
          break;

        NodeID srcStart = sTile * sTile_size;
        NodeID srcEnd   = (sTile + 1) * sTile_size;
        srcStart += numHubVertices; 
        srcEnd += numHubVertices; 
        auto sources    = g.in_neighs_in_range(dst, srcStart, srcEnd);
        if (sources.size() == 0)
        {
          ++numNullVtx;
          nonNullNeighbor = false;
        }
        else
        {
          #pragma omp atomic
          ++tileOffsetsList[BURST_1]; //we access the list only when vtx.in_degree > 0
          if (nonNullNeighbor == true)
          {
            ++contiguousVertices; 
          }
          inOffsetWindow.insert(dst / 16);
          nonNullNeighbor = true;
        }
        allSources.insert(allSources.end(), sources.begin(), sources.end());
      }

      // Modeling the cost in-offsets access (BURST_1)
      #pragma omp critical
      {
        tileInOffsets[BURST_1] += (2 * (dTile_size - numNullVtx)) - contiguousVertices;
        tileInOffsets[BURST_16] += inOffsetWindow.size();
      }

      // find the unique sources in the tile
      std::set<NodeID> uniqSources (allSources.begin(), allSources.end());

      // This tile was not empty
      if (uniqSources.size() != 0) 
      {
        #pragma omp critical 
        {
          srcData[BURST_1] += uniqSources.size();
          tileSources[BURST_1] += allSources.size();
        }

        // find number of BURST-16 windows required for uniqSources
        std::set<NodeID> uniqWindows;
        for (auto src : uniqSources)
        {
          uniqWindows.insert(src / 16); 
        }
        #pragma omp atomic
        srcData[BURST_16] += uniqWindows.size();
      }
      else
      {
        #pragma omp atomic
        ++numZeroTiles;
      }
    }
    
    /* Counting reuse of destination vertices  */
    for (NodeID dst = dTile * dTile_size; dst < (dTile+1) * dTile_size; ++dst)
    {
      if (dst >= g.num_nodes())
        break;
      if (g.in_degree(dst) != 0) 
      {
        #pragma omp atomic
        ++dstData[BURST_1];
      }
    }
    /* Counting reuse of destination vertices. Accesses in terms of burst windows */
    stat_t numWindows = (dTile_size + 15) / 16;
    for (stat_t window = 0; window < numWindows; ++window)
    {
      bool containsElem {false}; //check if we have a single valid dst in burst window
      for (stat_t winID = window * 16; winID < (window+1) * 16; ++winID)
      {
        NodeID dstID = (dTile * dTile_size) + winID;
        if (dstID >= g.num_nodes())
          break;
        if (g.in_degree(dstID) != 0)
        {
          containsElem = true;
          break;
        }
      }
      if (containsElem) 
      {
        #pragma omp atomic
        ++dstData[BURST_16];
      }
    }
  }
  
  /* simple computations for streaming data types */
  tileEmpty[BURST_16]       = (tileEmpty[BURST_1] + 15) /  16;
  tileSources[BURST_16]     = (tileSources[BURST_1] + 15) / 16;
  tileOffsetsList[BURST_16] = (tileOffsetsList[BURST_1] + 15) / 16;

  /* Account for destination data being read and written */
  dstData[BURST_1]  *= 2;
  dstData[BURST_16] *= 2;
  

  /* summarize stats */
  stat_t dataAccesses[2]  = {0};
  stat_t adjAccesses[2]   = {0};
  stat_t totalAccesses[2] = {0}; 
  for (int burst = 0; burst < 2; ++burst)
  {
    dataAccesses[burst]   = (srcData[burst] + dstData[burst]);
    adjAccesses[burst]    = (tileEmpty[burst] + tileInOffsets[burst] \
                             + tileSources[burst] + tileOffsetsList[burst]);
    totalAccesses[burst]  = dataAccesses[burst] + adjAccesses[burst];
  }

  /* print stats */
  std::cout << "------- STATS BEGIN ------\n";
  std::cout <<"*** Tile Stats ***\n"; 
  std::cout << "Total number of tiles = " << (hub_sTiles * hub_dTiles) + (tail_sTiles * tail_dTiles) << std::endl;
  std::cout << "Number of zero tiles  = " << numZeroTiles << std::endl;
  std::cout <<"*** Tile Stats ***\n"; 
  std::cout <<"*** BURST - 1 ***\n"; 
  std::cout << "SRC data (vData) offchip reads         = " << srcData[BURST_1] << std::endl;
  std::cout << "DST data (inDegrees) offchip accesses  = " << dstData[BURST_1] << std::endl;
  std::cout << "TileEmpty offchip accesses             = " << tileEmpty[BURST_1] << std::endl;
  std::cout << "TileInOffsets offchip accesses         = " << tileInOffsets[BURST_1] << std::endl;
  std::cout << "TileOffsetsList accesses (compression) = " << tileOffsetsList[BURST_1] << std::endl;
  std::cout << "TileSources offchip accesses           = " << tileSources[BURST_1] << std::endl;
  std::cout << "~ SUMMARY ~\n";
  std::cout << "Vertex Data Accesses = " << dataAccesses[BURST_1] << std::endl;
  std::cout << "Ajacency Accesses    = " << adjAccesses[BURST_1] << std::endl;
  std::cout << "Total Accesses       = " << totalAccesses[BURST_1] << std::endl;
  std::cout <<"*** BURST - 1 ***\n"; 
  std::cout << "\n";
  std::cout <<"*** BURST - 16 ***\n"; 
  std::cout << "SRC data (vData) offchip reads         = " << srcData[BURST_16] << std::endl;
  std::cout << "DST data (inDegrees) offchip accesses  = " << dstData[BURST_16] << std::endl;
  std::cout << "TileEmpty offchip accesses             = " << tileEmpty[BURST_16] << std::endl;
  std::cout << "TileInOffsets offchip accesses         = " << tileInOffsets[BURST_16] << std::endl;
  std::cout << "TileOffsetsList accesses (compression) = " << tileOffsetsList[BURST_16] << std::endl;
  std::cout << "TileSources offchip accesses           = " << tileSources[BURST_16] << std::endl;
  std::cout << "~ SUMMARY ~\n";
  std::cout << "Vertex Data Accesses = " << dataAccesses[BURST_16] << std::endl;
  std::cout << "Ajacency Accesses    = " << adjAccesses[BURST_16] << std::endl;
  std::cout << "Total Accesses       = " << totalAccesses[BURST_16] << std::endl;
  std::cout <<"*** BURST - 16 ***\n"; 
  std::cout << "\n";
  std::cout << "------- STATS END --------\n";

  pvector<ScoreT> scores(g.num_nodes());
  return scores;
}


void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = g.num_nodes();
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  k = min(k, static_cast<int>(top_k.size()));
  cout << "[START] Scores \n";
  for (auto kvp : top_k)
    cout << kvp.second << " : " << kvp.first << "\n";
  cout << "[END] Scores \n";
}


// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool PRVerifier(const Graph &g, const pvector<ScoreT> &scores,
                        double target_error) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> incomming_sums(g.num_nodes(), 0);
  double error = 0;
  for (NodeID u : g.vertices()) {
    ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
    for (NodeID v : g.out_neigh(u))
      incomming_sums[v] += outgoing_contrib;
  }
  for (NodeID n : g.vertices()) {
    error += fabs(base_score + kDamp * incomming_sums[n] - scores[n]);
    incomming_sums[n] = 0;
  }
  PrintTime("Total Error", error);
  return error < target_error;
}

int main(int argc, char* argv[]) {
  CLPageRank cli(argc, argv, "square-tiles", 262144, 1);
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  assert(cli.aspect_ratio() >= 1 && "aspect ratio for wide tile incorrectly specified\n");
  auto analyticalCount = [&cli] (const Graph &g) {
    return hybridTiles(g, cli.tile_sz(), cli.aspect_ratio());
  };
  auto VerifierBound = [&cli] (const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tile_sz());
  };
  BenchmarkKernel(cli, g, analyticalCount, PrintTopScores, VerifierBound);
  return 0;
}
