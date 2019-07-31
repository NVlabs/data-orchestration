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

#include "assert.h"

#include "whoop.hpp"


int main(int argc, char** argv)
{

  using namespace whoop;

  VecIn  inputs("inputs");
  VecIn  weights("weights");
  VecOut outputs("outputs");

  int kNumPEs = 1;

  whoop::AddOption(&kNumPEs, "num_PEs", "Number of PEs");

  whoop::Init(argc, argv);

  int kInputWidth = inputs.Size();
  int kWeightWidth = weights.Size();
  const int kHaloWidth = kWeightWidth - 1;
  
  const int kOutputWidth = kInputWidth - kWeightWidth + 1;
  outputs.Resize(kOutputWidth);
  
  whoop::ASSERT(kWeightWidth % kNumPEs == 0) << "Number of PEs: " << kNumPEs << " does not divide weight width: " << kWeightWidth << whoop::EndT;
 
  whoop::T(0) << "Input Width: " << kInputWidth << whoop::EndT;
  whoop::T(0) << "Weight Width: " << kWeightWidth << whoop::EndT;
  whoop::T(0) << "Output Width: " << kOutputWidth << whoop::EndT;
  whoop::T(0) << whoop::EndT;
  whoop::T(0) << "NumPEs: " << kNumPEs << whoop::EndT;
  whoop::T(0) << "Weights per PE: " << kWeightWidth / kNumPEs << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  // Short-form variable names
  const int W = kInputWidth;
  const int S = kWeightWidth;
  const int S1 = kWeightWidth / kNumPEs;
  const int S0 = kNumPEs;
  const int Q = kOutputWidth;

  Var s1("s1");
  Var s0("s0");
  Var s("s");
  Var q("q");

  t_for(s1, 0, S1);
  {
    t_for(q, 0, Q);
    {
      inputs.AddTileLevel(W);
      outputs.AddTileLevel(Q);
      weights.BypassTileLevel();
      s_for(s0, 0, S0);
      {
        inputs.BypassTileLevel();
        weights.AddTileLevel(1);
        outputs.BypassTileLevel();
        s = s1 * S0 + s0;
        outputs[q] += inputs[q + s] * weights[s];
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
