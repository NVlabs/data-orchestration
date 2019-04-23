// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <numeric>
#include <cmath>

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

pvector<ScoreT> squareTiles(const Graph &g, int s_tile_size,
                            int d_tile_size) 
{
  if (s_tile_size == -1) s_tile_size = g.num_nodes(); //s_tile_size == S0
  if (d_tile_size == -1) d_tile_size = g.num_nodes(); //d_tile_size == D0

  const int num_sTiles = (g.num_nodes() + s_tile_size-1) / s_tile_size;
  const int num_dTiles = (g.num_nodes() + d_tile_size-1) / d_tile_size;
  
  std::cout << "S_TILES = " << num_sTiles << std::endl;
  std::cout << "D_TILES = " << num_dTiles << std::endl;

  //Tile stats
  stat_t numZeroTiles {0};
  //std::vector<double> nnzsPerTile (num_sTiles * num_dTiles); 
  pvector<double> nnzsPerTile (num_sTiles * num_dTiles, 0);


  /* Offchip access counts for different data types - for burst size of 1 and 16 */
  stat_t srcData[2] = {0};
  stat_t dstData[2] = {0};
  stat_t tileEmpty[2]       = {0}; 
  stat_t tileOffsetsList[2] = {0}; //compressed representation of vertices with in-degree > 0
  stat_t tileInOffsets[2]   = {0};
  stat_t tileSources[2]     = {0};

  const int chunkSz = std::max(g.num_nodes() / 50, 16l);
  

  /* Access counts - BURST_SZ = 1 and BURST_SZ = 16 are mixed */
  #pragma omp parallel for schedule(dynamic, chunkSz) reduction(+ : srcData, dstData, \
                                                                tileEmpty, tileInOffsets, \
                                                                tileOffsetsList, \
                                                                tileSources, numZeroTiles)
  for (int dTile = 0; dTile < num_dTiles; ++dTile) 
  {
    for (int sTile = 0; sTile < num_sTiles; ++sTile)
    {
      ++tileEmpty[BURST_1]; //checked for each tile
      
      // keep track of number of consecutive dst vertices with in_degree > 0
      bool nonNullNeighbor {false};
      stat_t contiguousVertices {0};
      std::set<NodeID> inOffsetWindow;

      // populate all sources in the tile
      std::vector<NodeID> allSources;
      stat_t numNullVtx = 0; //number of empty columns in a tile (wasted adjacency read)
      for (NodeID dst = dTile * d_tile_size; dst < (dTile + 1) * d_tile_size; ++dst)
      {
        if (dst >= g.num_nodes())
          break;

        NodeID srcStart = sTile * s_tile_size;
        NodeID srcEnd   = (sTile + 1) * s_tile_size;
        auto sources    = g.in_neighs_in_range(dst, srcStart, srcEnd);
        if (sources.size() == 0)
        {
          ++numNullVtx;
          nonNullNeighbor = false;
        }
        else
        {
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
      tileInOffsets[BURST_1] += (2 * (d_tile_size - numNullVtx)) - contiguousVertices;
      tileInOffsets[BURST_16] += inOffsetWindow.size();

      // Number of NNZs in this tile
      //nnzsPerTile[(sTile * s_tile_size) + dTile] = static_cast<double>(allSources.size());

      
    
      // find the unique sources in the tile
      std::set<NodeID> uniqSources (allSources.begin(), allSources.end());

      // This tile was not empty
      if (uniqSources.size() != 0) 
      {
        srcData[BURST_1] += uniqSources.size();
        tileSources[BURST_1] += allSources.size();

        // find number of BURST-16 windows required for uniqSources
        std::set<NodeID> uniqWindows;
        for (auto src : uniqSources)
        {
          uniqWindows.insert(src / 16); 
        }
        srcData[BURST_16] += uniqWindows.size();
      }
      else
      {
        ++numZeroTiles;
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

  /* Compute variance in NNZ distribution across tiles */
  #if 0
  double sum      = std::accumulate(nnzsPerTile.begin(), nnzsPerTile.end(), 0.0); 
  double meanNNZs = sum / nnzsPerTile.size();

  std::vector<double> diff (nnzsPerTile.size());
  std::transform(nnzsPerTile.begin(), nnzsPerTile.end(), diff.begin(), [meanNNZs](double x) {return x - meanNNZs; });
  double sq_sum     = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double stddevNNZs = std::sqrt(sq_sum / nnzsPerTile.size());
  #endif

  // mean
  double sum {0.0};
  for (auto &elem : nnzsPerTile)
    sum += elem;
  double meanNNZs = sum / nnzsPerTile.size();
  #if 0
  std::cout << "Sum of NNZs = " << sum << std::endl;
  std::cout << "Num. of edges = " << g.num_edges_directed() << std::endl;
  assert(sum == g.num_edges_directed()); 
  #endif
  assert(nnzsPerTile.size() == static_cast<size_t>(num_sTiles * num_dTiles));

  // std-dev
  double stdSum {0.0};
  for (auto &elem : nnzsPerTile)
    stdSum += std::pow(elem - meanNNZs, 2);
  double stddevNNZs = std::sqrt(stdSum / nnzsPerTile.size());


 
  /* print stats */
  std::cout << "------- STATS BEGIN ------\n";
  std::cout <<"*** Tile Stats ***\n"; 
  std::cout << "Total number of tiles = " << num_sTiles * num_dTiles << std::endl;
  std::cout << "Number of zero tiles  = " << numZeroTiles << std::endl;
  std::cout << "Avg NNZs per tile     = " << meanNNZs << std::endl;
  std::cout << "std-dev of NNZs       = " << stddevNNZs << std::endl;
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
