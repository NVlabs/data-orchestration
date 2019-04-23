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

#include "whoop.hpp"
#include "loop-variable.hpp" 

int main(int argc, char** argv)
{

  using namespace whoop;

  VecIn  inputs("inputs");
  VecIn  weights("weights");
  VecIn  dimensions("dimensions");
  VecOut outputs("outputs");

  whoop::Init(argc, argv);


  assert(dimensions.Size() == 6);

  int kInputWidth = inputs.Size();
  int kWeightWidth = weights.Size();

  // Short-form variable names
  const int K = dimensions.At(conv6dVar::K);
  const int C = dimensions.At(conv6dVar::C);
  const int R = dimensions.At(conv6dVar::R);
  const int S = dimensions.At(conv6dVar::S);
  const int W = dimensions.At(conv6dVar::W);
  const int H = dimensions.At(conv6dVar::H);
  const int P = W - R + 1;
  const int Q = H - S + 1;
  
  assert(kInputWidth == C * W * H);
  assert(kWeightWidth == K * C * R * S);

  const int kOutputWidth = K * P * Q;
  outputs.Resize(kOutputWidth);

  whoop::T(0) << "6D Convolution dimension description" << whoop::EndT;
  whoop::T(0) << "K (Output Channel): " << K << whoop::EndT;
  whoop::T(0) << "C (Input Channel): " << C << whoop::EndT;
  whoop::T(0) << "R (Weight Width): " << R << whoop::EndT;
  whoop::T(0) << "S (Weight Height): " << S << whoop::EndT;
  whoop::T(0) << "W (Input Width): " << W << whoop::EndT;
  whoop::T(0) << "H (Input Height): " << H << whoop::EndT;
  whoop::T(0) << "Input Width (Number of input pixels): " << kInputWidth << whoop::EndT;
  whoop::T(0) << "Weight Width (Number of weight pixels): " << kWeightWidth << whoop::EndT;
  whoop::T(0) << "Output Width (Number of output pixels): " << kOutputWidth << whoop::EndT;
  whoop::T(0) << whoop::EndT;


  Var k("k");
  Var c("c");
  Var p("p");
  Var q("q");
  Var r("r");
  Var s("s");

  // (K -> Q -> P) -> (C -> S -> R) : Output-stationary
  t_for(p, 0, P);
  {
    t_for(q, 0, Q);
    {
      t_for(k, 0, K);
      {
        inputs.AddTileLevel(C*R*S);
        weights.AddTileLevel(K*C*R*S);
        outputs.AddTileLevel(1);
        t_for(c, 0 , C);
        {
          t_for(s, 0, S);
          {
            t_for(r, 0, R);
            {
             outputs[P*Q*k + P*q + p] +=
               inputs[W*H*c + W*(q+s) + (p+r)] * weights[C*R*S*k + R*S*c + R*s + r];
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
  for (int x = 0; x < kWeightWidth; x++)
  {
    whoop::T(1) << "W " << x << " = " << weights.At(x) << whoop::EndT;
  }
  for (int x = 0; x < kInputWidth; x++)
  {
    whoop::T(1) << "I " << x << " = " << inputs.At(x) << whoop::EndT;
  }
  for (int x = 0; x < kOutputWidth; x++)
  {
    whoop::T(1) << "O " << x << " = " << outputs.At(x) << whoop::EndT;
  }

  whoop::Done(); 
}
