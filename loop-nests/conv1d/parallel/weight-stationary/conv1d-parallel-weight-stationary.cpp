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
  
  assert(kWeightWidth >= kNumPEs );

  if(kWeightWidth % kNumPEs != 0)
  {
    std::cout << "[" << argv[0] << "] Number of PEs does not divide weight width. Number of PEs will be updated to weight width ( " << kWeightWidth << ")" << std::endl;
    kNumPEs = kWeightWidth;
  }

  const int kPartitionSize = kWeightWidth / kNumPEs;

  whoop::T(0) << "Input Width: " << kInputWidth << whoop::EndT;
  whoop::T(0) << "Weight Width: " << kWeightWidth << whoop::EndT;
  whoop::T(0) << "Output Width: " << kOutputWidth << whoop::EndT;
  whoop::T(0) << whoop::EndT;
  whoop::T(0) << "NumPEs: " << kNumPEs << whoop::EndT;
  whoop::T(0) << "Partition size: " << kPartitionSize << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  // Short-form variable names
  const int E = kNumPEs;
  const int W = kInputWidth;
  const int R = kWeightWidth;
  const int R1 = kNumPEs;
  const int R0 = kPartitionSize;
  const int P = kOutputWidth;

  Var r1("r1");
  Var w("w");
  Var r0("r0");
  Var p("p");

  s_for(r1, 0, R1);
  {
    t_for(r0, 0, R0);
    {
      inputs.AddTileLevel(P, 1);
      weights.AddTileLevel(1);
      outputs.AddTileLevel(P);

      t_for(p, 0, P);
      {
        outputs[p] += inputs[p + r1*R0 + r0] * weights[r1*R0 + r0];
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
