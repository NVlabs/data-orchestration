/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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


#include "conv6d.hpp"

#include "whoop.hpp"

#include "timewhoop.hpp"
#include "yaml-writer.hpp"

int main(int argc, char** argv)
{

  using namespace whoop;

  TensorIn  inputs("inputs");
  TensorIn  weights("weights");
  TensorOut outputs("outputs");

  int hTileSz = 4;
  int wTileSz = 4;

  AddOption(&hTileSz, "tile_size_h", "The height of input 2D tile");
  AddOption(&wTileSz, "tile_size_w", "The width of input 2D tile");

  whoop::Init(argc, argv);

  const int kInputWidth = inputs.Size(static_cast<int>(dims::Inputs::W));
  const int kInputHeight = inputs.Size(static_cast<int>(dims::Inputs::H));
  const int kInputChannel = inputs.Size(static_cast<int>(dims::Inputs::C));
  const int kWeightHeight = weights.Size(static_cast<int>(dims::Weights::S));
  const int kWeightWidth = weights.Size(static_cast<int>(dims::Weights::R));
  const int kWeightChannel = weights.Size(static_cast<int>(dims::Weights::C));
  const int kOutputWidth = kInputWidth - kWeightWidth + 1;
  const int kOutputHeight = kInputHeight - kWeightHeight + 1;
  const int kOutputChannel = weights.Size(static_cast<int>(dims::Weights::K));

  assert(kWeightChannel == kInputChannel);
  
  // Short-form variable names
  const int K = kOutputChannel;
  const int C = kInputChannel;
  const int R = kWeightWidth;
  const int S = kWeightHeight;
  const int W = kInputWidth;
    const int W0 = wTileSz;
    const int W1 = (W % W0 == 0)? W/W0 : W/W0 + 1;
  const int H = kInputHeight;
    const int H0 = hTileSz; 
    const int H1 = (H % H0 == 0)? H/H0 : H/H0 + 1; 
  const int P = kOutputWidth;
    const int P0 = W0+R;
  const int Q = kOutputHeight;
    const int Q0 = H0+S;

 
  outputs.Resize({kOutputChannel, kOutputHeight, kOutputWidth});

  whoop::T(0) << "6D Convolution dimension description" << whoop::EndT;
  whoop::T(0) << "K (Output Channel): " << K << whoop::EndT;
  whoop::T(0) << "C (Input Channel): " << C << whoop::EndT;
  whoop::T(0) << "R (Weight Width): " << R << whoop::EndT;
  whoop::T(0) << "S (Weight Height): " << S << whoop::EndT;
  whoop::T(0) << "W (Input Width): " << W << whoop::EndT;
  whoop::T(0) << "W0 (Input Tile Width): " << W0 << whoop::EndT;
  whoop::T(0) << "W1 (Num Input Tiles in a 2D-tile column): " << W1 << whoop::EndT;
  whoop::T(0) << "H (Input Height): " << H << whoop::EndT;
  whoop::T(0) << "H0 (Input Tile Height): " << H0 << whoop::EndT;
  whoop::T(0) << "H1 (Num Input Tiles in a 2D-tile row): " << H1 << whoop::EndT;
  whoop::T(0) << whoop::EndT;


  Var k("k");
  Var c("c");
  Var w1("w1");
  Var w0("w0");
  Var w("w");
  Var h1("h1");
  Var h0("h0");
  Var h("h");
  Var r("r");
  Var s("s");

  t_for(k, 0, K);
  {
    t_for(s, 0, S);
    {
      t_for(r, 0, R);
      {
        t_for(c, 0 , C);
        {
          t_for(h1, 0, H1);
          {
            t_for(w1, 0, W1);
            {
              inputs.AddTileLevel(C*H0*W0);
              weights.AddTileLevel(1);
              outputs.AddTileLevel(Q0*P0);
              t_for(h0, 0, H0);
              {
                t_for(w0, 0, W0);
                {
                  h = H0 * h1 + h0;
                  w = W0 * w1 + w0;

                  w_if( (h-s < Q) && (w-r < P) && (h-s >= 0) && (w-r >= 0) );
                  {
                    outputs[k][h-s][w-r] += inputs[c][h][w] * weights[k][c][s][r];
                  }
                  end(); //w_if
                }
                end(); //t_for(w0)
              }
              end(); //t_for(h0)
            }
            end(); //t_for(w1)
          }
          end(); //t_for(h1)
        }
        end(); //t_for(c)
      }
      end(); //t_for(r)
    }
    end(); //t_for(s)
  }
  end(); //t_for(k)
 
  std::cout << "RUNNING..." << std::endl;
  whoop::Run();
  std::cout << "DONE." << std::endl;

  for (int ki = 0; ki < kOutputChannel; ki++)
  {  
    for (int ci = 0; ci < kWeightChannel; ci++)
    {
      for (int si = 0; si < kWeightHeight; si++)
      {  
        for (int ri = 0; ri < kWeightWidth; ri++)
        {
          whoop::T(2) << "W [" << ki << "][" << ci << "][" << si << "][" << ri << "] = " << weights.At({ki,ci,si,ri}) << whoop::EndT;
        }
      }
    }
  }

  for (int ci = 0; ci < kInputChannel; ci++)
  {
    for (int hi = 0; hi < kInputHeight; hi++)
    {  
      for (int wi = 0; wi < kInputWidth; wi++)
      {
        whoop::T(2) << "I [" << ci << "][" << hi << "][" << wi << "] = " << inputs.At({ci,hi,wi}) << whoop::EndT;
      }
    }
  }

  for (int ki = 0; ki < kOutputChannel; ki++)
  {
    for (int qi = 0; qi < kOutputHeight; qi++)
    {  
      for (int pi = 0; pi < kOutputWidth; pi++)
      {
        whoop::T(2) << "O [" << ki << "][" << qi << "][" << pi << "] = " << outputs.At({ki,qi,pi}) << whoop::EndT;
      }
    }
  }

  whoop::Done();
}
