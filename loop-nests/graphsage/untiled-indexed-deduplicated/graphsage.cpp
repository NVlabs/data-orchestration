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

  // Input graph.
  TensorIn is_connected("adj_matrix");
  TensorIn features("features");

  // Trained model.
  TensorIn W_in("W_in");
  TensorIn W_1("W_1"), b_1("b_1");
  TensorIn W_2("W_2"), b_2("b_2");
  TensorIn W_out("W_out");

  // Output predictions.
  TensorOut prediction("prediction");

  // Intermediate tensors.

  // -- dense --
  Tensor batch("batch");
  Tensor n1_sample("n1_sample");
  Tensor n2_sample("n2_sample");

  // -- sparse --
  Tensor n1_dedup("n1_dedup");
  Tensor n1_parent("n1_parent");
  Tensor n2_dedup("n2_dedup");
  Tensor n2_parent("n2_parent");

  // -- dense --
  Tensor n2_encoded("n2_encoded");
  Tensor n1_encoded("n1_encoded");
  Tensor n1_sums("n1_sums");
  Tensor src_sums("src_sums");

  // Temporaries.
  Tensor activation("activation");
  Tensor encoded("encoded");

  // Un-simulated tensors.
  Tensor adj_lookup("adj_lookup");
  Tensor adj_count("adj_count");

  // Init.
  whoop::Init(argc, argv);

  // Sizes.
  int num_vertices = is_connected.Size(0);
  int feature_size = features.Size(0);
  int hidden_layer_size = W_in.Size(1);
  
  int batch_size = 8;
  int num_sampled_neighbors = 3;

  // Short-form variable names
  int V = num_vertices;
  int F = feature_size;
  int B = batch_size;
  int H = hidden_layer_size;
  int N = num_sampled_neighbors;

  assert(is_connected.Size(1) == V);
  assert(features.Size(1) == V);
  assert(W_in.Size(0) == F);
  assert(W_1.Size(0) == 2*H && W_1.Size(1) == H);
  assert(b_1.Size(0) == H);
  assert(W_2.Size(0) == H && W_2.Size(1) == H);
  assert(b_2.Size(0) == H);
  assert(W_out.Size(0) == H);

  // -- dense --
  batch.Resize({ B });
  n1_sample.Resize({ B, N });
  n2_sample.Resize({ B, N, N });

  // -- sparse --
  n1_dedup.Resize({ V });
  n1_parent.Resize({ V, B, N });
  n2_dedup.Resize({ V });
  n2_parent.Resize({ V, B, N, N });

  // -- dense --
  n2_encoded.Resize({ B, N, N, H });
  n1_encoded.Resize({ B, N, H });
  n1_sums.Resize({ B, N, H });
  src_sums.Resize({ B, H });
  prediction.Resize({ B });
  
  activation.Resize({ 2*H });
  encoded.Resize({ H });

  adj_lookup.Resize({ V, V });
  adj_count.Resize({ V });  
  
  // Pre-generate some useful graph meta-data (essentially an adjacency list).
  for (int s = 0; s < V; s++)
  {
    int count = 0;
    for (int d = 0; d < V; d++)
    {
      if (is_connected.At({ s, d }) == 1)
      {
        adj_lookup.At({ s, count++ }) = d;
      }
    }
    adj_count.At({ s }) = count;
  }

  // Pre-generate (in un-simulated code) batch, n1 and n2 samples.
  for (int b = 0; b < B; b++)
  {
    int s = rand() % V;
    batch.At({ s, b }) = 1;

    for (int n1_pos = 0; n1_pos < N; n1_pos++)
    {
      int n1_idx = rand() % adj_count.At({ s });
      int n1 = adj_lookup.At({ s, n1_idx });
      n1_sample.At({ b_pos, n1_pos }) = n1;

      for (int n2_pos = 0; n2_pos < N; n2_pos++)
      {
        int n2_idx = rand() % adj_count.At({ n1 });
        int n2 = adj_lookup.At({ n1, n2_idx });
        n2_sample.At({ b_pos, n1_pos, n2_pos }) = n2;
      }
    }
  }
  
  // Iteration indices.
  Var b_pos("b_pos");
  Var n1_pos("n1_pos");
  Var n2_pos("n2_pos");
  Var h("h");
  Var f("f");

  // Scalar data/index variables.
  Var n1("n1");
  Var n2("n2");
  Var pred("pred");
  Var temp("temp");
  Var mean("mean");

  //
  // The algorithm.
  //

  // Dedup n1 neighbors and setup parent links.
  t_for(b_pos, 0, B); {
    t_for(n1_pos, 0, N); {
      n1 = n1_sample[b_pos][n1_pos];
      n1_dedup[n1] = 1;
      n1_parent[n1][b_pos][n1_pos] = 1;
    } end();
  } end();
  
  // Dedup n2 neighbors and setup parent links.
  // We could have inlined this with the n1 dedup nest above.
  t_for(b_pos, 0, B); {
    t_for(n1_pos, 0, N); {
      t_for(n2_pos, 0, N); {
        n2 = n2_sample[b_pos][n1_pos][n2_pos];
        n2_dedup[n2] = 1;
        n2_parent[n2][b_pos][n1_pos][n2_pos] = 1;
      } end();
    } end();
  } end();

  // Encode de-duplicated n2 vertices and distribute/copy the results into the dense
  // n2_encoded tensor.
  t_for(n2, 0, V); {
    w_if(n2_dedup[n2] == 1); {
      
      // Encode each unique n2 vertex.
      t_for(h, 0, H); {
        temp = 0;
        t_for(f, 0, F); {
          temp += (W_in[h][f] * features[n2][f]);
        } end();
        w_if(temp < 0); {
          temp = 0; // ReLU.
        } end();
        encoded[h] = temp;
      } end();

      // I have an encoded value for each distinct n2. Now
      // scatter this value to the n2_encoded tensor.
      t_for(b_pos, 0, B); {
        t_for(n1_pos, 0, N); {
          t_for(n2_pos, 0, N); {
            w_if(n2_parent[n2][b_pos][n1_pos][n2_pos] == 1); {
              
              t_for(h, 0, H); {
                n2_encoded[b_pos][n1_pos][n2_pos][h] = encoded[h];
              } end();
                      
            } end();
          } end();
        } end();
      } end();
      
    } end();
  } end();
   
  // Encode de-duplicated n1 vertices and distribute/copy the results into the dense
  // n1_encoded tensor.
  t_for(n1, 0, V); {
    w_if(n1_dedup[n1] == 1); {

      // Encode each unique n1 vertex.
      t_for(h, 0, H); {
        temp = 0;
        t_for(f, 0, F); {
          temp += (W_in[h][f] * features[n1][f]);
        } end();
        w_if(temp < 0); {
          temp = 0; // ReLU.
        } end();
        encoded[h] = temp;
      } end();

      // I have an encoded value for each distinct n1. Now
      // scatter this value to the n1_encoded tensor.
      t_for(b_pos, 0, B); {
        t_for(n1_pos, 0, N); {
          w_if(n1_parent[n1][b_pos][n1_pos] == 1); {
            t_for(h, 0, H); {
              n1_encoded[b_pos][n1_pos][h] = encoded[h];
            } end();
          } end();
        } end();
      } end();

    } end();
  } end();

  // -- dense --

  // (1) Work on B*N*N cuboid.
  t_for(b_pos, 0, B); {
    t_for(n1_pos, 0, N); {
      t_for(n2_pos, 0, N); {

        // Reduce the n2_encoded tensor into n1_sums.
        t_for(h, 0, H); {
          n1_sums[b_pos][n1_pos][h] += n2_encoded[b_pos][n1_pos][n2_pos][h];
        } end();
        
      } end();
    } end();
  } end();
  
  // (2) Work on B*N rectangle.
  t_for(b_pos, 0, B); {
    t_for(n1_pos, 0, N); {

      // Concatenate n1_sums with n1_encoded.
      t_for(h, 0, H); {
        activation[h] = n1_encoded[b_pos][n1_pos][h];
        activation[h+H] = n1_sums[b_pos][n1_pos][h] / N;
      } end();
      
      // Apply the W_1 NN layer.
      t_for(h, 0, H); {
        temp = 0;
        t_for(f, 0, 2*H); { // ** note! **
          temp += (W_1[h][f] * activation[f]);
        } end();
        temp += b_1[h];                  
        w_if(temp < 0); {
          temp = 0; // ReLU.
        } end();

        // Reduce the results to src_sums.
        src_sums[s][b_pos][h] += temp;        
      } end();
  
    } end();
  } end();

  // (3) Work on B vector.
  t_for(b_pos, 0, B); {

    // Calculate src means.
    t_for(h, 0, H); {
      mean[h] = src_sums[b_pos][h] / N;
    } end();

    // Apply the W_2 and W_out layers.
    pred = 0;
    t_for(h, 0, H); {

      // W_2.
      temp = 0;
      t_for(f, 0, H); { // ** note! **
        temp += (W_2[h][f] * mean[f]);
      } end();
      temp += b_2[h];
      w_if(temp < 0); {
        temp = 0; // ReLU.
      } end();

      // W_out.
      pred += W_out[h] * temp;
    } end();

    w_if(pred < 0); {
      pred = 0; // ReLU.
    }
    
    // This is it.
    prediction[b_pos] = pred;
        
  } end();

  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::T(0) << "DONE." << whoop::EndT;

  for (int b = 0; b < B; b++)
  {
    whoop::T(3) << "prediction[" << b << "] = " << prediction.At({ b }) << whoop::EndT;
  }
    
  whoop::Done();
}
