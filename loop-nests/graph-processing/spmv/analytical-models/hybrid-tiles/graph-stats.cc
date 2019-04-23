// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cmath>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include <cassert>


/*
GAP Benchmark Suite
Kernel: Wide-Aspect Tiles access count
Author: Vignesh Balaji 

General graph statistics.

Reports the number of hub vertices, medium degree vertices, 
and tail vertices in the graph (using out-degree because we 
are running dst-stationary code)
*/


using namespace std;

typedef float ScoreT;
typedef uint64_t stat_t;
const float kDamp = 0.85;

const int maxElements = std::pow(2,19);

pvector<ScoreT> graphStats(const Graph &g)
{
  /* report following stats */
  stat_t hubVertices  {0};
  stat_t tailVertices {0};
  NodeID maxDegree    {0};
  NodeID degreeCutoff = g.num_edges_directed() / g.num_nodes();
  
  #pragma omp parallel for reduction(+ : hubVertices)
  for (NodeID n = 0; n < g.num_nodes(); ++n)
  {
    if (g.out_degree(n) > degreeCutoff)
      ++hubVertices;
  }

  #pragma omp parallel for reduction(max : maxDegree) 
  for (NodeID n = 0; n < g.num_nodes(); ++n) 
  {
    if (g.out_degree(n) > maxDegree)
      maxDegree = g.out_degree(n);
  }

  tailVertices = g.num_nodes() - hubVertices;

  /* collect tile dimension stats. Index 0 is src tiles and 1 is dst tiles */
  stat_t numHubTiles[2]  {0};
  stat_t numTailTiles[2] {0};

  NodeID s_tile_size = (1 * pow(2,18)) / 4;
  numHubTiles[0] = (hubVertices + s_tile_size-1) / s_tile_size;
  NodeID numRemaining = g.num_nodes() - (s_tile_size * numHubTiles[0]);
  numHubTiles[1] = (numRemaining + (maxElements-s_tile_size-1)) / (maxElements-s_tile_size);

  s_tile_size = (3 * pow(2,18)) / 4;
  numTailTiles[0] = (hubVertices + s_tile_size-1) / s_tile_size;
  numRemaining = g.num_nodes() - (s_tile_size * numTailTiles[0]);
  numTailTiles[1] = (numRemaining + (maxElements-s_tile_size-1)) / (maxElements-s_tile_size);

  /* print out stats */
  std::cout << "********* General Stats ********\n";
  std::cout << "Total number of vertices = " << g.num_nodes() << std::endl;
  std::cout << "Max. Degree              = " << maxDegree     << std::endl;
  std::cout << "Num. of hub vertices     = " << hubVertices   << std::endl;
  std::cout << "Num. of non-hub vertices = " << tailVertices   << std::endl;
  std::cout << std::endl;
  std::cout << "********* Tile dimensions ********\n";
  std::cout << "Hub.src_tiles  = " << numHubTiles[0]  << " & Hub.dst_tiles  = " << numHubTiles[1] << std::endl;
  std::cout << "Tail.src_tiles = " << numTailTiles[0] << " & Tail.dst_tiles = " << numTailTiles[1] << std::endl;
  std::cout << std::endl;
  std::cout << "********* Num. Tiles ********\n";
  std::cout << "Number of tiles with wide-aspect ratio = " << (numHubTiles[0]*numHubTiles[1]) + (numTailTiles[0]*numTailTiles[1]) << std::endl;
  NodeID numSqTiles = (g.num_nodes() + pow(2,18)-1) / pow(2,18);
  std::cout << "Number of tiles with square-aspect ratio = " << numSqTiles * numSqTiles << std::endl;
  std::cout << std::endl;

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
  auto analyticalCount = [&cli] (const Graph &g) {
    return graphStats(g);
  };
  auto VerifierBound = [&cli] (const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tile_sz());
  };
  BenchmarkKernel(cli, g, analyticalCount, PrintTopScores, VerifierBound);
  return 0;
}
