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

  int kNumPEs = 10;
 
  whoop::AddOption(&kNumPEs, "num_PEs", "Number of PEs");

  whoop::Init(argc, argv);

  int kInputWidth = inputs.Size();
  int kWeightWidth = weights.Size();

  assert(kInputWidth == kWeightWidth);
  
  const int kOutputWidth = 1;
  outputs.Resize(kOutputWidth);

  if(kWeightWidth % kNumPEs != 0)
  {
    whoop::T(0) << "Warning: data set size is not dividable by the number of PEs. Setting number of PEs = 1" << whoop::EndT;
    whoop::T(0) << "  Data set size: " << kWeightWidth << ", given number of PEs: " << kNumPEs << whoop::EndT;
    kNumPEs = 1;
  }
  assert(kWeightWidth % kNumPEs == 0);

  const int kPartitionSize = kWeightWidth / kNumPEs;
  Vec partialSums(kNumPEs, 0, "psums");

  // Short-form variable names
  const int P = kNumPEs;
  const int R = kWeightWidth;
  const int R0 = kPartitionSize;
  

  Var p("p");
  Var r("r");

  s_for(p, 0, P);
  {
    inputs.AddTileLevel(R0, 1);
    weights.AddTileLevel(R0);
    partialSums.AddTileLevel(1);
    
    t_for(r, 0, R0);
    {
      partialSums[p] += inputs[R0*p + r] * weights[R0*p + r];
    }
    end();
  }
  end();  

  t_for(p, 0, P);
  {
    outputs.AddTileLevel(1);
    outputs[0] += partialSums[p];
  }
  end();


  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::T(0) << "DONE." << whoop::EndT;
  for (int x = 0; x < kInputWidth; x++)
  {
    whoop::T(2) << "I " << x << " = " << inputs.At(x) << whoop::EndT;
  }
  for (int x = 0; x < kInputWidth; x++)
  {
    whoop::T(2) << "W " << x << " = " << weights.At(x) << whoop::EndT;
  }
  for (int x = 0; x < kOutputWidth; x++)
  {
    whoop::T(2) << "O " << x << " = " << outputs.At(x) << whoop::EndT;
  }

  whoop::Done();  
}
