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


int main(int argc, char** argv)
{

  using namespace whoop;
  
  VecIn  inputs("inputs");
  VecIn  weights("weights");
  VecOut outputs("outputs");

  int kWeightWidthL0 = 3;
  int kWeightWidthL1 = 1;
  int kOutputWidthL0 = 3;
  int kOutputWidthL1 = 5;

  whoop::AddOption(&kWeightWidthL0, "weight_width_0,w", "Length of L0 weight tile.");
  whoop::AddOption(&kWeightWidthL1, "weight_width_1,w", "Length of L1 weight tile.");
  whoop::AddOption(&kOutputWidthL0, "output_width_0,o", "Length of L0 output tile.");
  whoop::AddOption(&kOutputWidthL1, "output_width_1,o", "Length of L1 output tile.");

  whoop::Init(argc, argv);

  int kInputWidth = inputs.Size();
  int kWeightWidth = weights.Size();
  
  const int kOutputWidth = kInputWidth - kWeightWidth + 1;
  outputs.Resize(kOutputWidth);

  const int kWeightWidthL2 = kWeightWidth / (kWeightWidthL1 * kWeightWidthL0);
  const int kOutputWidthL2 = kOutputWidth / (kOutputWidthL1 * kOutputWidthL0);

  whoop::T(0) << "Input Width: " << kInputWidth << whoop::EndT;
  whoop::T(0) << "Weight Width: " << kWeightWidth << whoop::EndT;
  whoop::T(0) << "Output Width: " << kOutputWidth << whoop::EndT;
  whoop::T(0) << whoop::EndT;
  whoop::T(0) << "WeightL2 Width: " << kWeightWidthL2 << whoop::EndT;
  whoop::T(0) << "WeightL1 Width: " << kWeightWidthL1 << whoop::EndT;
  whoop::T(0) << "WeightL0 Width: " << kWeightWidthL0 << whoop::EndT;
  whoop::T(0) << whoop::EndT;
  whoop::T(0) << "OutputL2 Width: " << kOutputWidthL2 << whoop::EndT;
  whoop::T(0) << "OutputL1 Width: " << kOutputWidthL1 << whoop::EndT;
  whoop::T(0) << "OutputL0 Width: " << kOutputWidthL0 << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  assert(kWeightWidth % (kWeightWidthL0 * kWeightWidthL1) == 0);
  assert(kOutputWidth % (kOutputWidthL0 * kOutputWidthL1) == 0);

  // Short-form variable names
  const int W = kInputWidth;
  const int S = kWeightWidth;
  const int S0 = kWeightWidthL0;
  const int S1 = kWeightWidthL1;
  const int S2 = kWeightWidthL2;
  const int Q = kOutputWidth;
  const int Q0 = kOutputWidthL0;
  const int Q1 = kOutputWidthL1;
  const int Q2 = kOutputWidthL2;

  Var s2("s2");
  Var s1("s1");
  Var s0("s0");
  Var s("s");
  Var q2("q2");
  Var q1("q1");
  Var q0("q0");
  Var q("q");

  t_for(q2, 0, Q2);
  {
    t_for(s2, 0, S2);
    {
      t_for(q1, 0, Q1);
      {      
        inputs.AddTileLevel(Q0 + S0);
        weights.AddTileLevel(S1 * S0);
        outputs.AddTileLevel(Q0);

        t_for(s1, 0, S1);
        {
          t_for(q0, 0, Q0);
          {
            inputs.AddTileLevel(S0, 1);
            weights.AddTileLevel(S0);
            outputs.AddTileLevel(1);

            t_for(s0, 0, S0);
            {
              q = q2 * Q1 * Q0 + q1 * Q0 + q0;
              s = s2 * S1 * S0 + s1 * S0 + s0;
              outputs[q] += inputs[q + s] * weights[s];
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

  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::T(0) << "DONE." << whoop::EndT;

  whoop::Done();
}
