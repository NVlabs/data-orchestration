// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <set>

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

This version does not compress the row-offsets list and, hence, 
does many unnecessary reads (for vertices with no in-neighbors
in a tile). 

The main version now accounts for a list that maintains a list
of vertices that have atleast one in-neighbor 

*/


using namespace std;

typedef float ScoreT;
typedef uint64_t stat_t;
const float kDamp = 0.85;

const int BURST_1  = 0;
const int BURST_16 = 1;

pvector<ScoreT> squareTiles(const Graph &g, int s_tile_size,
                            int d_tile_size) 
{
  if (s_tile_size == -1) s_tile_size = g.num_nodes(); //s_tile_size == S0
  if (d_tile_size == -1) d_tile_size = g.num_nodes(); //d_tile_size == D0

  const int num_sTiles = (g.num_nodes() + s_tile_size-1) / s_tile_size;
  const int num_dTiles = (g.num_nodes() + d_tile_size-1) / d_tile_size;

  //Tile stats
  std::cout << "S_TILES = " << num_sTiles << std::endl;
  std::cout << "D_TILES = " << num_dTiles << std::endl;

  /* Offchip access counts for different data types - for burst size of 1 and 16 */
  stat_t srcData[2] = {0};
  stat_t dstData[2] = {0};
  stat_t tileEmpty[2]     = {0};
  stat_t tileInOffsets[2] = {0};
  stat_t tileInOffsets_redundant[2] = {0}; //number of zero in-deg vertices in tiled sub-graph
  stat_t tileSources[2]   = {0};
  const int chunkSz = std::max(g.num_nodes() / 50, 16l);
  

  /* Access counts - BURST_SZ = 1 and BURST_SZ = 16 are mixed */
  #pragma omp parallel for schedule(dynamic, chunkSz) reduction(+ : srcData, dstData, \
                                                                tileEmpty, tileInOffsets, \
                                                                tileInOffsets_redundant, \
                                                                tileSources)
  for (int dTile = 0; dTile < num_dTiles; ++dTile) 
  {
    for (int sTile = 0; sTile < num_sTiles; ++sTile)
    {
      ++tileEmpty[BURST_1]; //checked for each tile

      // populate all sources in the tile
      std::vector<NodeID> allSources;
      stat_t numNullVtx = 0; //number of empty columns in a tile (wasted adjacency read)

      for (NodeID dst = dTile * d_tile_size; dst < (dTile + 1) * d_tile_size; ++dst)
      {
        if (dst >= g.num_nodes())
          break;

        ++tileInOffsets[BURST_1]; 

        NodeID srcStart = sTile * s_tile_size;
        NodeID srcEnd   = (sTile + 1) * s_tile_size;
        auto sources    = g.in_neighs_in_range(dst, srcStart, srcEnd);
        if (sources.size() == 0)
          ++numNullVtx;
        allSources.insert(allSources.end(), sources.begin(), sources.end());
      }
    
      // find the unique sources in the tile
      std::set<NodeID> uniqSources (allSources.begin(), allSources.end());

      // This tile was not empty
      if (uniqSources.size() != 0) 
      {
        srcData[BURST_1] += uniqSources.size();
        tileSources[BURST_1] += allSources.size();
        tileInOffsets_redundant[BURST_1] += numNullVtx;

        // find number of BURST-16 windows required for uniqSources
        std::set<NodeID> uniqWindows;
        for (auto src : uniqSources)
        {
          uniqWindows.insert(src / 16); 
        }
        srcData[BURST_16] += uniqWindows.size();
      }
    }
    
    /* Counting reuse of destination vertices  */
    for (NodeID dst = dTile * d_tile_size; dst < (dTile+1) * d_tile_size; ++dst)
    {
      if (dst >= g.num_nodes())
        break;
      if (g.in_degree(dst) != 0) 
      {
        ++dstData[BURST_1];
      }
    }
    /* Counting reuse of destination vertices. Accesses in terms of burst windows */
    stat_t numWindows = (d_tile_size + 15) / 16;
    for (stat_t window = 0; window < numWindows; ++window)
    {
      bool containsElem {false}; //check if we have a single valid dst in burst window
      for (stat_t winID = window * 16; winID < (window+1) * 16; ++winID)
      {
        NodeID dstID = (dTile * d_tile_size) + winID;
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
        ++dstData[BURST_16];
      }
    }
  }
  
  /* simple computations for streaming data types */
  tileEmpty[BURST_16]     = (tileEmpty[BURST_1] + 15) /  16;
  tileSources[BURST_16]   = (tileSources[BURST_1] + 15) / 16;
  tileInOffsets[BURST_16] = (tileInOffsets[BURST_1] + 15) / 16;

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
                              + tileSources[burst]);
    totalAccesses[burst]  = dataAccesses[burst] + adjAccesses[burst];
  }

 
  /* print stats */
  std::cout << "------- STATS BEGIN ------\n";
  std::cout <<"*** BURST - 1 ***\n"; 
  std::cout << "SRC data (vData) offchip reads         = " << srcData[BURST_1] << std::endl;
  std::cout << "DST data (inDegrees) offchip accesses  = " << dstData[BURST_1] << std::endl;
  std::cout << "TileEmpty offchip accesses             = " << tileEmpty[BURST_1] << std::endl;
  std::cout << "TileInOffsets offchip accesses         = " << tileInOffsets[BURST_1] << std::endl;
  std::cout << "TileInOffsets (Redundant) accesses     = " << tileInOffsets_redundant[BURST_1] << std::endl;
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
  std::cout << "TileInOffsets (Redundant) accesses     = " << tileInOffsets_redundant[BURST_16] << std::endl;
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
  CLPageRank cli(argc, argv, "square-tiles", 262144, 262144);
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  assert(cli.s_tile_sz() == cli.d_tile_sz() && "Src and Dst tiles must be same size\n");
  auto analyticalCount = [&cli] (const Graph &g) {
    return squareTiles(g, cli.s_tile_sz(), cli.d_tile_sz());
  };
  auto VerifierBound = [&cli] (const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.d_tile_sz());
  };
  BenchmarkKernel(cli, g, analyticalCount, PrintTopScores, VerifierBound);
  return 0;
}
