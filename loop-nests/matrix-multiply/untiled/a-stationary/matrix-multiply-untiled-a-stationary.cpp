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

//An output-stationary inner product 
int main(int argc, char** argv)
{

  using namespace whoop;

  TensorIn  input_a("input_a");
  TensorIn  input_b("input_b");
  TensorOut outputs("outputs");

  whoop::Init(argc, argv);

  int kInputAWidth = input_a.Size(0);
  int kInputBWidth = input_b.Size(0);
  int kInputAHeight = input_a.Size(1);
  int kInputBHeight = input_b.Size(1);
  
  const int kOutputWidth = kInputBWidth;
  const int kOutputHeight = kInputAHeight;
  outputs.Resize({kOutputHeight, kOutputWidth});

  whoop::T(0) << "Input A Width: "  << kInputAWidth << whoop::EndT;
  whoop::T(0) << "Input A Height: " << kInputAHeight << whoop::EndT;
  whoop::T(0) << "Input B Width: "  << kInputBWidth << whoop::EndT;
  whoop::T(0) << "Input B Height: " << kInputBHeight << whoop::EndT;
  whoop::T(0) << "Output Width: "   << kOutputWidth << whoop::EndT;
  whoop::T(0) << "Output Height: "  << kOutputHeight << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  whoop::ASSERT(kInputAWidth == kInputBHeight) << "The width of Matrix A must equal the height of Matrix B. A width: " << kInputAWidth << ", B height: " << kInputBHeight << whoop::EndT;
  
  // Short-form variable names
  const int M = kInputAHeight;
  const int K = kInputAWidth;
  const int N = kInputBWidth;

  Var m("m");
  Var k("k");
  Var n("n");

  // A-stationary, row-major.
  t_for(m, 0, M);
  {
    input_a.AddTileLevel(1);
    input_b.AddTileLevel(K*N);
    outputs.AddTileLevel(N);
    t_for(k, 0, K);
    {
      t_for(n, 0, N);
      {
        outputs[m][n] += input_a[m][k] * input_b[k][n];
      }
      end();
    }
    end();
  }
  end();
  
  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::Done();
}
