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

#include "conv2d.hpp"


int main(int argc, char** argv)
{

  using namespace whoop;
  
  TensorIn  inputs("inputs");
  TensorIn  weights("weights");
  TensorOut outputs("outputs");

  whoop::Init(argc, argv);

  int kInputWidth   = inputs.Size(dims::W);
  int kInputHeight  = inputs.Size(dims::H);
  int kWeightWidth  = weights.Size(dims::R);
  int kWeightHeight = weights.Size(dims::S);
  
  const int kOutputWidth = kInputWidth - kWeightWidth + 1;
  const int kOutputHeight = kInputHeight - kWeightHeight + 1;
  outputs.Resize({kOutputHeight, kOutputWidth});

  whoop::T(0) << "Input Width: " << kInputWidth << whoop::EndT;
  whoop::T(0) << "Input Height: " << kInputWidth << whoop::EndT;
  whoop::T(0) << whoop::EndT;
  whoop::T(0) << "Weight Width: " << kWeightWidth << whoop::EndT;
  whoop::T(0) << "Weight Height: " << kWeightWidth << whoop::EndT;
  whoop::T(0) << whoop::EndT;
  whoop::T(0) << "Output Width: " << kOutputWidth << whoop::EndT;
  whoop::T(0) << "Output Height: " << kOutputWidth << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  // Short-form variable names
  const int W = kInputWidth;
  const int H = kInputHeight;
  const int R = kWeightHeight;
  const int S = kWeightWidth;
  const int P = kOutputHeight;
  const int Q = kOutputWidth;

  Var q("q");
  Var p("p");
  Var r("r");
  Var s("s");

  t_for(p, 0, P);
  {
    t_for(q, 0, Q);
    {
      t_for(r, 0, R);
      {
        t_for(s, 0, S);
        {
          outputs[p][q] += inputs[p + r][q + s] * weights[r][s];
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

  for (int y = 0; y < H; y++)
  {
    for (int x = 0; x < W; x++)
    {
      whoop::T(2) << "I [" << y << "][" << x << "] = " << inputs.At({y, x}) << whoop::EndT;
    }
  }
  for (int y = 0; y < Q; y++)
  {
    for (int x = 0; x < P; x++)
    {
      whoop::T(2) << "O [" << y << "][" << x << "] = " << outputs.At({y, x}) << whoop::EndT;
    }
  }
  
  whoop::Done();
}
