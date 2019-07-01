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
  Tensor batch("batch");
  Tensor n1_sample("n1_sample");
  Tensor n2_sample("n2_sample");
  
  Tensor n1_sums("n1_sums"), n1_count("n1_count");
  Tensor src_sums("src_sums"), src_count("src_count");
  Tensor activation("activation");

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
  
  prediction.Resize({ V, B });
  
  batch.Resize({ V, B });
  n1_sample.Resize({ V, B, V, N });
  n2_sample.Resize({ V, B, V, N, V, N });
  
  n1_sums.Resize({ V, B, V, N, H });
  n1_count.Resize({ V, B, V, N });
  src_sums.Resize({ V, B, H });
  src_count.Resize({ V, B });
  
  activation.Resize({ 2*H });

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
      n1_sample.At({ s, b, n1, n1_pos }) = 1;

      for (int n2_pos = 0; n2_pos < N; n2_pos++)
      {
        int n2_idx = rand() % adj_count.At({ n1 });
        int n2 = adj_lookup.At({ n1, n2_idx });
        n2_sample.At({ s, b, n1, n1_pos, n2, n2_pos }) = 1;
      }
    }
  }
  
  // Iteration indices.
  Var s("s");
  Var b("b");
  Var n1("n1");
  Var n1_pos("n1_pos");
  Var n2("n2");
  Var n2_pos("n2_pos");
  Var h("h");
  Var f("f");

  // Scalar data variables.
  Var pred("pred");
  Var temp("temp");
  Var mean("mean");
  
  // The algorithm.
  t_for(s, 0, V);
  {
    t_for(b, 0, B);
    {
      w_if(batch[s][b] == 1);
      {
        // Look for n1 vertices.
        t_for(n1, 0, V);
        {
          t_for(n1_pos, 0, N);
          {
            w_if(n1_sample[s][b][n1][n1_pos] == 1);
            {
              // "Recurse" into n2 vertices.
              t_for(n2, 0, V);
              {
                t_for(n2_pos, 0, N);
                {
                  w_if(n2_sample[s][b][n1][n1_pos][n2][n2_pos] == 1);
                  {
                    t_for(h, 0, H);
                    {
                      temp = 0;
                      t_for(f, 0, F);
                      {
                        temp += (W_in[h][f] * features[n2][f]);
                      }
                      end(); // t_for(f, 0, F);
                      // ReLU.
                      w_if(temp < 0);
                      {
                        temp = 0;
                      }
                      end();
                      n1_sums[s][b][n1][n1_pos][h] += temp;
                    }
                    end(); // t_for(h, 0, H);
                    n1_count[s][b][n1][n1_pos] += 1;
                  }
                  end(); // w_if(n2_sample[s][b][n1][n1_pos][n2][n2_pos] == 1);
                }
                end(); // t_for(n2_pos, 0, N);
              }
              end(); // t_for(n2, 0, V);

              // Digest n1 vertices, rendezvous the results with those from n2 digestion above,
              // and apply the W_1 NN layer.
              t_for(h, 0, H);
              {
                temp = 0;
                t_for(f, 0, F);
                {
                  temp += (W_in[h][f] * features[n1][f]);
                }
                end(); // t_for(f, 0, F);
                // ReLU.
                w_if(temp < 0);
                {
                  temp = 0;
                }
                end();
                activation[h] = temp;
                activation[h+H] = n1_sums[s][b][n1][n1_pos][h] /
                  n1_count[s][b][n1][n1_pos]; // mean of n2 results.
              }
              end(); // t_for(h, 0, H);

              t_for(h, 0, H);
              {
                temp = 0;
                t_for(f, 0, 2*H); // ** note! **
                {
                  temp += (W_1[h][f] * activation[f]);
                }
                end(); // t_for(f, 0, 2H);
                temp += b_1[h];
                // ReLU.
                w_if(temp < 0);
                {
                  temp = 0;
                }
                end();
                src_sums[s][b][h] += temp;
              }
              end(); // t_for(h, 0, H);
              src_count[s][b] += 1;
            }
            end(); // w_if(n1_sample[s][b][n1][n1_pos] == 1);
          }
          end(); // t_for(n1_pos, 0, N);
        }
        end(); // t_for(n1, 0, V);

        // // Calculate src means.
        // t_for(h, 0, H);
        // {
        //   mean[h] = src_sums[s][h] / src_count[s]; // mean of n2 results.
        // }
        // end(); // t_for(h, 0, H);

        // Calculate final prediction.
        pred = 0;
        t_for(h, 0, H);
        {
          temp = 0;
          t_for(f, 0, H); // ** note! **
          {
            mean = src_sums[s][b][f] / src_count[s][b];
            temp += (W_2[h][f] * mean);
          }
          end(); // t_for(f, 0, H);
          temp += b_2[h];
          // ReLU.
          w_if(temp < 0);
          {
            temp = 0;
          }
          end();
          pred += W_out[h] * temp;
        }
        end(); // t_for(h, 0, H);
      
        prediction[s][b] = pred;
      }
      end(); // w_if(batch[s][b] == 1);
    }
    end(); // t_for(b, 0, B);
  }
  end(); // t_for(s, 0, V);

  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::T(0) << "DONE." << whoop::EndT;

  for (int s = 0; s < V; s++)
  {
    for (int b = 0; b < B; b++)
    {
      if (batch.At({ s, b }))
      {
        whoop::T(3) << "prediction[" << s << "][" << b << "] = " << prediction.At({ s, b }) << whoop::EndT;
      }
    }
  }
    
  whoop::Done();
}
