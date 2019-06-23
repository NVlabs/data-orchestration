/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdlib>

#include "whoop.hpp"

int main(int argc, char** argv)
{
  using namespace whoop;

  const int kGraphVertices = 64;
  const int kFeatureSize = 8;
  const int kHiddenLayerSize = 16;
  const int kBatchSize = 8;
  
  // Input graph.
  TensorIn is_connected("adj_matrix.64.64");
  TensorIn is_in_neighbor_sample("neib_sample.64.64");
  TensorIn features("features.64.8");

  // Trained model.
  TensorIn W_in("W_in.16.8");
  TensorIn W_1("W_1.16.32"), b_1("b_1.16");
  TensorIn W_2("W_2.16.16"), b_2("b_2.16");
  TensorIn W_out("W_out.16");

  // Output predictions.
  TensorOut prediction("prediction.64");
  prediction.Resize({ kGraphVertices });

  whoop::Init(argc, argv);

  // Intermediate tensors.
  Tensor is_in_batch("batch");
  Tensor n1_sums("n1_sums"), n1_count("n1_count");
  Tensor src_sums("src_sums"), src_count("src_count");
  Tensor activation("activation");

  is_in_batch.Resize({ kGraphVertices });
  n1_sums.Resize({ kGraphVertices, kHiddenLayerSize });
  n1_count.Resize({ kGraphVertices });
  src_sums.Resize({ kGraphVertices, kHiddenLayerSize });
  src_count.Resize({ kGraphVertices });
  activation.Resize({ 2*kHiddenLayerSize });
  
  for (int vertex_id = 0; vertex_id < kGraphVertices; vertex_id++)
  {
    is_in_batch.At({ vertex_id }) = 0;
    n1_count.At({ vertex_id }) = 0;
    src_count.At({ vertex_id }) = 0;    
    for (int h = 0; h < kHiddenLayerSize; h++)
    {
      n1_sums.At({ vertex_id, h }) = 0;
    }
  }

  for (int h = 0; h < 2*kHiddenLayerSize; h++)
  {
    activation.At({ h }) = 0;
  }
  
  // Choose random nodes to place in batch.

  for (int vertex_id = 0; vertex_id < kGraphVertices; vertex_id++)
  {
    is_in_batch.At({ vertex_id }) = 0;
  }
  
  int chosen = 0;
  while (chosen < kBatchSize)
  {
    int vertex_id = rand() % kGraphVertices;
    if (is_in_batch.At({ vertex_id }) == 0)
    {
      is_in_batch.At({ vertex_id }) = 1;
      chosen++;      
    }    
  }
  
  // Scalar data variables.
  int pred;
  int temp;
  int mean;
  
  // Short-form variable names
  const int V = kGraphVertices;
  const int F = kFeatureSize;
  const int H = kHiddenLayerSize;
  
  whoop::T(0) << "RUNNING..." << whoop::EndT;
  
  // The algorithm.
  for (int s = 0; s < V; s++)
  {
    if (is_in_batch.At({ s }) == 1)
    {
      // Look for n1 vertices.
      for (int n1 = 0; n1 < V; n1++)
      {
        if (is_connected.At({ s, n1 }) == 1)
        {
          if (is_in_neighbor_sample.At({ s, n1 }) == 1)
          {
            // "Recurse" into n2 vertices.
            for (int n2 = 0; n2 < V; n2++)
            {
              if (is_connected.At({ n1, n2 }) == 1)
              {
                if (is_in_neighbor_sample.At({ n1, n2 }) == 1)
                {
                  for (int h = 0; h < H; h++)
                  {
                    temp = 0;
                    for (int f = 0; f < F; f++)
                    {
                      temp += (W_in.At({ h, f }) * features.At({ n2, f }));
                    }
                    // ReLU.
                    if (temp < 0)
                    {
                      temp = 0;
                    }
                    n1_sums.At({ n1, h }) += temp;
                  }
                  n1_count.At({ n1 }) = n1_count.At({ n1 }) + 1;
                }
              }
            }

            // Digest n1 vertices, rendezvous the results with those from n2 digestion above,
            // and apply the W_1 NN layer.
            for (int h = 0; h < H; h++)
            {
              temp = 0;
              for (int f = 0; f < F; f++)
              {
                temp += (W_in.At({ h, f }) * features.At({ n1, f }));
              }
              // ReLU.
              if (temp < 0)
              {
                temp = 0;
              }
              activation.At({ h }) = temp;
              activation.At({ h+H }) = n1_sums.At({ n1, h }) / n1_count.At({ n1 }); // mean of n2 results.
            }

            for (int h = 0; h < H; h++)
            {
              temp = 0;
              for (int f = 0; f < 2*H; f++) // ** note! **
              {
                temp += (W_1.At({ h, f }) * activation.At({ f }));
              }
              temp += b_1.At({ h });
              // ReLU.
              if (temp < 0)
              {
                temp = 0;
              }
              src_sums.At({ s, h }) += temp;
            }
            src_count.At({ s }) = src_count.At({ s }) + 1;
          }
        }
      }

      // // Calculate src means.
      // for (int h = 0; h < H; h++)
      // {
      //   mean.At({ h] = src_sums.At({ s, h }) / src_count[s }); // mean of n2 results.
      // }

      // Calculate final prediction.
      pred = 0;
      for (int h = 0; h < H; h++)
      {
        temp = 0;
        for (int f = 0; f < H; f++) // ** note! **
        {
          mean = src_sums.At({ s, f }) / src_count.At({ s });
          temp += (W_2.At({ h, f }) * mean);
        }
        temp += b_2.At({ h });
        // ReLU.
        if (temp < 0)
        {
          temp = 0;
        }
        pred += W_out.At({ h }) * temp;
      }
      
      prediction.At({ s }) = pred;
    }
  }

  whoop::T(0) << "DONE." << whoop::EndT;

  for (int v = 0; v < kGraphVertices; v++)
  {
    if (is_in_batch.At({ v }))
    {
      whoop::T(3) << "prediction[" << v << "] = " << prediction.At({ v }) << whoop::EndT;
    }
  }
  
  whoop::Done();
}
