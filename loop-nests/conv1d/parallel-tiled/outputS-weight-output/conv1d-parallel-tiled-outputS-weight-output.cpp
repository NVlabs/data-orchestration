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

  int kWeightWidthL0 = 4;
  int kOutputWidthL1 = 64;

  whoop::AddOption(&kWeightWidthL0, "weight_width_0,w", "Length of L0 weight tile.");
  whoop::AddOption(&kOutputWidthL1, "output_width_1,o", "Length of L1 output tile.");

  whoop::Init(argc, argv);

  int kInputWidth = inputs.Size();
  int kWeightWidth = weights.Size();
  
  const int kOutputWidth = kInputWidth - kWeightWidth + 1;
  outputs.Resize(kOutputWidth);

  const int kWeightWidthL1 = kWeightWidth / kWeightWidthL0;
  const int kOutputWidthL0 = kOutputWidth / kOutputWidthL1;

  whoop::T(0) << "Input Width: " << kInputWidth << whoop::EndT;
  whoop::T(0) << "Weight Width: " << kWeightWidth << whoop::EndT;
  whoop::T(0) << "Output Width: " << kOutputWidth << whoop::EndT;
  whoop::T(0) << whoop::EndT;
  whoop::T(0) << "WeightL1 Width: " << kWeightWidthL1 << whoop::EndT;
  whoop::T(0) << "WeightL0 Width: " << kWeightWidthL0 << whoop::EndT;
  whoop::T(0) << whoop::EndT;
  whoop::T(0) << "OutputL1 Width: " << kOutputWidthL1 << whoop::EndT;
  whoop::T(0) << "OutputL0 Width: " << kOutputWidthL0 << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  assert(kWeightWidth % kWeightWidthL0 == 0);
  assert(kOutputWidth % kOutputWidthL1 == 0);

  // Short-form variable names
  const int W = kInputWidth;
  const int R = kWeightWidth;
  const int R0 = kWeightWidthL0;
  const int R1 = kWeightWidthL1;
  const int P = kOutputWidth;
  const int P0 = kOutputWidthL0;
  const int P1 = kOutputWidthL1;

  Var r1("r1");
  Var r0("r0");
  Var p1("p1");
  Var p0("p0");

  t_for(p1, 0, P1);
  {
    inputs.AddTileLevel(P0 + R0, 1);
    outputs.AddTileLevel(P0);
    t_for(r1, 0, R1);
    {
      t_for(p0, 0, P0);
      {
        inputs.AddTileLevel(R0, 1);
        weights.AddTileLevel(R0);
        outputs.AddTileLevel(1);

        s_for(r0, 0, R0);
        {
          outputs[p1 * kOutputWidthL0 + p0] += 
           inputs[p1 * kOutputWidthL0 + p0 + r1 * kWeightWidthL0 + r0] * 
                                     weights[r1 * kWeightWidthL0 + r0];
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

  return 0;
}
