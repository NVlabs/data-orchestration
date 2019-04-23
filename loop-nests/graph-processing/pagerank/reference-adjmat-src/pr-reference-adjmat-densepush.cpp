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
 
 // a reference implementation of pagerank-delta
// push-based, no frontier, always iterate all nodes


#include "whoop.hpp"
#include <math.h>


int main(int argc, char** argv)
{

  using namespace whoop;

  // TODO: Need to make input parameters alpha and e configurable
  // Currently hardcode them.
  // [trick]: everything x100 to support integer
  const int alpha = 85;//0.85;
  const int e = 1;//0.01;
  const int big_const = 1000000;

  VecIn  AdjacencyMatrix("adjmat");
  // Rank needs to be floating point
  // TODO: floating point support, currently integer
  VecOut Rank("rank"); 

  whoop::Init(argc, argv);

  int numVertices  = sqrt(AdjacencyMatrix.Size());

  whoop::T(0) << "Number of Vertices:       " << numVertices  << whoop::EndT;
  whoop::T(0) << "Adjacency Matrix Size:    " << AdjacencyMatrix.Size() << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  // Short-form variable names
  const int V = numVertices;

  //TODO: DeltaSum, Delta: need to be floating point 
  Vec DeltaSum(V, "deltasum");
  // [trick]: everything times V^2 to support integer
  Vec Delta(V, "delta");

  // Frontier need to be bool (single bit). storage efficiency
  Vec Frontier(V, "frontier");
  Vec OutDegree(V, "outdegree");
  

  // preprocess the adjcent matrix to obtain outdegree for each node
  // this step should be skipped for CSR format input
  Rank.Resize(V);
  for(int s=0; s < V; s++) 
  {
      OutDegree.At(s) = 0;
      for(int d=0; d < V; d++){
          if( AdjacencyMatrix.At(s * V + d) != 0){ 
              OutDegree.At(s) += 1;
          }
      }
      Rank.At(s) = 0;
      DeltaSum.At(s) = 0;
      Delta.At(s) = big_const/V;
      Frontier.At(s) = 1;
  }

  whoop::T(0) << "RUNNING..." << whoop::EndT;

  bool updates = true;
  int round = 1;
      
  for (int v = 0; v < V; v++)
  {
      whoop::T(3) << "Rank " << v << " = " << Rank.At(v) 
        << "; Delta = " << Delta.At(v)
        << "; Frontier = " << Frontier.At(v) 
        << "; OutDegree = " << OutDegree.At(v) << whoop::EndT;
  }

  while( updates ) 
  {
      updates = false;

      whoop::T(1) << "Push Round " << round << whoop::EndT;
      // iterate every node. Later need to make it a frontier
      for (int s = 0; s <  V; s++)
      {
          if (Frontier.At(s) == 0 ){
              continue;
          }
          // push to neighbors
          for (int d = 0; d <  V; d++)
          {
              // skip non-neighbor nodes
              if( AdjacencyMatrix.At(s*V + d) == 0 ) continue;
          
              DeltaSum.At(d) += Delta.At(s) / OutDegree.At(s) ;
          }
      }

      
      for (int v = 0; v < V; v++)
      {
          whoop::T(3) << "DeltaSum = " << v << " = " << DeltaSum.At(v) 
            << "; Delta = " << Delta.At(v)
            << "; Frontier = " << Frontier.At(v) << whoop::EndT;
      }
      
      whoop::T(1) << "Compute and Filter Round " << round << whoop::EndT;
      
      for (int v=0; v < V; v++){
          if (round == 1){
              Delta.At(v) = (DeltaSum.At(v)*alpha*V - alpha*big_const)/ (100*V);
          } else {
              Delta.At(v) = alpha * DeltaSum.At(v) / 100;
          }
          Rank.At(v) += Delta.At(v);
          DeltaSum.At(v) = 0;
          // test whether converges
          if (Rank.At(v) > 0 && (Delta.At(v) > e * Rank.At(v)/100
              || Delta.At(v) < -e*Rank.At(v)/100)){
              updates = true;
              Frontier.At(v) = 1;
          } else {
              Frontier.At(v) = 0;
          }
      }
 
      whoop::T(0) << "Finish Round ..." << round 
        << ". Verifying ... "<< whoop::EndT;
      
      for (int v = 0; v < V; v++)
      {
          whoop::T(3) << "Rank " << v << " = " << Rank.At(v) 
            << "; Delta = " << Delta.At(v)
            << "; Frontier = " << Frontier.At(v) << whoop::EndT;
      }
      round += 1;
  }

  whoop::T(0) << "DONE." << whoop::EndT;

  for (int v = 0; v < V; v++)
  {
    whoop::T(3) << "Rank " << v << " = " << Rank.At(v) << whoop::EndT;
  }
  
  whoop::Done();
}
