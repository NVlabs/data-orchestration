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

For 1d tiles, count the number of accesses to the two 
datatypes and also the adjacency matrix.

*/


using namespace std;

typedef float ScoreT;
typedef uint64_t stat_t;
const float kDamp = 0.85;

const int BURST_1  = 0;
const int BURST_16 = 1;

pvector<ScoreT> UnTiles(const Graph &g, int s_tile_size,
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
  stat_t InOffsets[2] = {0};
  stat_t Sources[2]   = {0};

  /* Access counts - BURST_SZ = 1 and BURST_SZ = 16 are mixed */
  #pragma omp parallel for schedule(dynamic, 64) reduction(+ : srcData, dstData, \
                                                               InOffsets, \
                                                               Sources)
  for (NodeID dst = 0; dst < g.num_nodes(); ++dst) 
  {
    std::vector <NodeID> allWindows;
    for (NodeID src : g.in_neigh(dst)) 
    {
        allWindows.push_back(src / 16);
    }
    std::set<NodeID> uniqWindows (allWindows.begin(), allWindows.end());
    srcData[BURST_16] += uniqWindows.size();
  }

  /* simple computation for destination data */
  dstData[BURST_1] += g.num_nodes();
  srcData[BURST_1] += g.num_edges_directed(); 
  
  /* simple computations for streaming data types */
  InOffsets[BURST_1]  = g.num_nodes() + 1;
  Sources[BURST_1]    = g.num_edges_directed();
  InOffsets[BURST_16] = (InOffsets[BURST_1] + 15) / 16;
  Sources[BURST_16]   = (Sources[BURST_1] + 15) / 16;
  dstData[BURST_16]   = (dstData[BURST_1] + 15) / 16;

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
    adjAccesses[burst]    = (InOffsets[burst] + Sources[burst]);
    totalAccesses[burst]  = dataAccesses[burst] + adjAccesses[burst];
  }

 
  /* print stats */
  std::cout << "------- STATS BEGIN ------\n";
  std::cout <<"*** BURST - 1 ***\n"; 
  std::cout << "SRC data (vData) offchip reads         = " << srcData[BURST_1] << std::endl;
  std::cout << "DST data (inDegrees) offchip accesses  = " << dstData[BURST_1] << std::endl;
  std::cout << "InOffsets offchip accesses             = " << InOffsets[BURST_1] << std::endl;
  std::cout << "Sources offchip accesses               = " << Sources[BURST_1] << std::endl;
  std::cout << "~ SUMMARY ~\n";
  std::cout << "Vertex Data Accesses = " << dataAccesses[BURST_1] << std::endl;
  std::cout << "Ajacency Accesses    = " << adjAccesses[BURST_1] << std::endl;
  std::cout << "Total Accesses       = " << totalAccesses[BURST_1] << std::endl;
  std::cout <<"*** BURST - 1 ***\n"; 
  std::cout << "\n";
  std::cout <<"*** BURST - 16 ***\n"; 
  std::cout << "SRC data (vData) offchip reads         = " << srcData[BURST_16] << std::endl;
  std::cout << "DST data (inDegrees) offchip accesses  = " << dstData[BURST_16] << std::endl;
  std::cout << "InOffsets offchip accesses             = " << InOffsets[BURST_16] << std::endl;
  std::cout << "Sources offchip accesses               = " << Sources[BURST_16] << std::endl;
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
  CLPageRank cli(argc, argv, "untiled", -1, -1);
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  assert(cli.d_tile_sz() == -1 && cli.s_tile_sz() == -1 && "Should not tile in dst dimension\n");
  auto analyticalCount = [&cli] (const Graph &g) {
    return UnTiles(g, cli.s_tile_sz(), cli.d_tile_sz());
  };
  auto VerifierBound = [&cli] (const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.d_tile_sz());
  };
  BenchmarkKernel(cli, g, analyticalCount, PrintTopScores, VerifierBound);
  return 0;
}
