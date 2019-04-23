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

  int kOutputWidthL0 = 4;

  whoop::AddOption(&kOutputWidthL0, "output_width_0,o", "Length of L0 output tile.");

  whoop::Init(argc, argv);

  int kInputWidth = inputs.Size();
  int kWeightWidth = weights.Size();
  
  const int kOutputWidth = kInputWidth - kWeightWidth + 1;
  outputs.Resize(kOutputWidth);

  const int kOutputWidthL1 = kOutputWidth / kOutputWidthL0;

  whoop::T(0) << "Input Width: " << kInputWidth << whoop::EndT;
  whoop::T(0) << "Weight Width: " << kWeightWidth << whoop::EndT;
  whoop::T(0) << "Output Width: " << kOutputWidth << whoop::EndT;
  whoop::T(0) << whoop::EndT;
  whoop::T(0) << "OutputL1 Width: " << kOutputWidthL1 << whoop::EndT;
  whoop::T(0) << "OutputL0 Width: " << kOutputWidthL0 << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  assert(kOutputWidth % kOutputWidthL0 == 0);

  // Short-form variable names
  const int W = kInputWidth;
  const int R = kWeightWidth;
  const int P = kOutputWidth;

  const int P0 = kOutputWidthL0;
  const int P1 = kOutputWidthL1;

  Var p0("p0");
  Var p1("p1");

  Var r("r");
  Var p("p");

  t_for(p1, 0, P1);
  {
    t_for(r, 0, R);
    {
      inputs.AddTileLevel(P0, 1);
      weights.AddTileLevel(1);
      outputs.AddTileLevel(P0);

      t_for(p0, 0, P0);
      {
        p = p1 * P0 + p0;
        outputs[p] += inputs[p + r] * weights[r];
      }
      end();
    }
    end();
  }
  end();
  
    
  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::T(0) << "DONE." << whoop::EndT;

  for (int x = 0; x < W; x++)
  {
    whoop::T(2) << "I " << x << " = " << inputs.At(x) << whoop::EndT;
  }
  for (int x = 0; x < P; x++)
  {
    whoop::T(2) << "O " << x << " = " << outputs.At(x) << whoop::EndT;
  }

  whoop::Done();
}
