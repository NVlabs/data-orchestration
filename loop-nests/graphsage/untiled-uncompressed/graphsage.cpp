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
  TensorIn is_in_neighbor_sample("neib_sample");
  TensorIn features("features");

  // Trained model.
  TensorIn W_in("W_in");
  TensorIn W_1("W_1"), b_1("b_1");
  TensorIn W_2("W_2"), b_2("b_2");
  TensorIn W_out("W_out");

  // Output predictions.
  TensorOut prediction("prediction");

  // Intermediate tensors.
  Tensor is_in_batch("batch");
  Tensor n1_sums("n1_sums"), n1_count("n1_count");
  Tensor src_sums("src_sums"), src_count("src_count");
  Tensor activation("activation");

  // Init.
  whoop::Init(argc, argv);

  // Sizes.
  int num_vertices = is_connected.Size(0);
  int feature_size = features.Size(0);
  int hidden_layer_size = W_in.Size(1);
  
  int batch_size = 8;

  assert(is_connected.Size(1) == num_vertices);
  assert(features.Size(1) == num_vertices);
  assert(W_in.Size(0) == feature_size);
  assert(W_1.Size(0) == 2*hidden_layer_size && W_1.Size(1) == hidden_layer_size);
  assert(b_1.Size(0) == hidden_layer_size);
  assert(W_2.Size(0) == hidden_layer_size && W_2.Size(1) == hidden_layer_size);
  assert(b_2.Size(0) == hidden_layer_size);
  assert(W_out.Size(0) == hidden_layer_size);
  
  prediction.Resize({ num_vertices });
  is_in_batch.Resize({ num_vertices });
  n1_sums.Resize({ num_vertices, hidden_layer_size });
  n1_count.Resize({ num_vertices });
  src_sums.Resize({ num_vertices, hidden_layer_size });
  src_count.Resize({ num_vertices });
  activation.Resize({ 2*hidden_layer_size });
  
  for (int vertex_id = 0; vertex_id < num_vertices; vertex_id++)
  {
    is_in_batch.At({ vertex_id }) = 0;
    n1_count.At({ vertex_id }) = 0;
    src_count.At({ vertex_id }) = 0;    
    for (int h = 0; h < hidden_layer_size; h++)
    {
      n1_sums.At({ vertex_id, h }) = 0;
    }
  }

  for (int h = 0; h < 2*hidden_layer_size; h++)
  {
    activation.At({ h }) = 0;
  }
  
  // Choose random nodes to place in batch.

  for (int vertex_id = 0; vertex_id < num_vertices; vertex_id++)
  {
    is_in_batch.At({ vertex_id }) = 0;
  }
  
  int chosen = 0;
  while (chosen < batch_size)
  {
    int vertex_id = rand() % num_vertices;
    if (is_in_batch.At({ vertex_id }) == 0)
    {
      is_in_batch.At({ vertex_id }) = 1;
      chosen++;      
    }    
  }
  
  // Iteration indices.
  Var s("s");
  Var d("d");
  Var n1("n1");
  Var n2("n2");
  Var h("h");
  Var f("f");

  // Scalar data variables.
  Var pred("pred");
  Var temp("temp");
  Var mean("mean");
  
  // Short-form variable names
  int V = num_vertices;
  int F = feature_size;
  int H = hidden_layer_size;
  
  // The algorithm.
  t_for(s, 0, V);
  {
    w_if(is_in_batch[s] == 1);
    {
      // Look for n1 vertices.
      t_for(n1, 0, V);
      {
        w_if(is_connected[s][n1] == 1);
        {
          w_if(is_in_neighbor_sample[s][n1] == 1);
          {
            // "Recurse" into n2 vertices.
            t_for(n2, 0, V);
            {
              w_if(is_connected[n1][n2] == 1);
              {
                w_if(is_in_neighbor_sample[n1][n2] == 1);
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
                    n1_sums[n1][h] += temp;
                  }
                  end(); // t_for(h, 0, H);
                  n1_count[n1] = n1_count[n1] + 1;
                }
                end(); // w_if(is_in_neighbor_sample[n1][n2] == 1);
              }
              end(); // w_if(is_connected[n1][n2]);
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
              activation[h+H] = n1_sums[n1][h] / n1_count[n1]; // mean of n2 results.
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
              src_sums[s][h] += temp;
            }
            end(); // t_for(h, 0, H);
            src_count[s] = src_count[s] + 1;
          }
          end(); // w_if(is_in_neighbor_sample[s][n1] == 1);
        }
        end(); // w_if(is_connected[s][n1] == 1);
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
          mean = src_sums[s][f] / src_count[s];
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
      
      prediction[s] = pred;
    }
    end(); // w_if(is_in_batch[s]);
  }
  end(); // t_for(s, 0, V);

  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::T(0) << "DONE." << whoop::EndT;

  for (int v = 0; v < num_vertices; v++)
  {
    if (is_in_batch.At({ v }))
    {
      whoop::T(3) << "prediction[" << v << "] = " << prediction.At({ v }) << whoop::EndT;
    }
  }
  
  whoop::Done();
}
