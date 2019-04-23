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

int main(int argc, char** argv)
{

  using namespace whoop;

  TensorIn  inputs("inputs");
  TensorIn  weights("weights");
  TensorOut outputs("outputs");

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
  const int H = kInputHeight;
  const int P = kOutputWidth;
  const int Q = kOutputHeight;
  
  outputs.Resize({kOutputChannel, kOutputHeight, kOutputWidth});

  whoop::T(0) << "6D Convolution dimension description" << whoop::EndT;
  whoop::T(0) << "K (Output Channel): " << K << whoop::EndT;
  whoop::T(0) << "C (Input Channel): " << C << whoop::EndT;
  whoop::T(0) << "R (Weight Width): " << R << whoop::EndT;
  whoop::T(0) << "S (Weight Height): " << S << whoop::EndT;
  whoop::T(0) << "W (Input Width): " << W << whoop::EndT;
  whoop::T(0) << "H (Input Height): " << H << whoop::EndT;
  whoop::T(0) << whoop::EndT;


  Var k("k");
  Var c("c");
  Var p("p");
  Var q("q");
  Var r("r");
  Var s("s");

  // (K -> Q -> P) -> (C -> S -> R) : Output-stationary
  t_for(k, 0, K);
  {
    t_for(q, 0, Q);
    {
      t_for(p, 0, P);
      {
        inputs.AddTileLevel(C*R*S, S);
        weights.AddTileLevel(C*R*S);
        outputs.AddTileLevel(1);
        t_for(c, 0 , C);
        {
          t_for(s, 0, S);
          {
            t_for(r, 0, R);
            {
             outputs[k][q][p] += inputs[c][q+s][p+r]  * weights[k][c][s][r];
            }
            end();
          }
          end();
        }
        end();
      }
      end();
    }
    end();
  }
  end();

 
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
