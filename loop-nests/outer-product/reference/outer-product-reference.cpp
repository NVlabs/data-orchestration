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

  VecIn  input_a("input_a");
  VecIn  input_b("input_b");
  VecOut outputs("outputs");

  whoop::Init(argc, argv);

  int kInputAWidth = input_a.Size();
  int kInputBWidth = input_b.Size();
  
  const int kOutputWidth = kInputAWidth * kInputBWidth;
  outputs.Resize(kOutputWidth);

  whoop::T(0) << "Input A Width: " << kInputAWidth << whoop::EndT;
  whoop::T(0) << "Input B Width: " << kInputBWidth << whoop::EndT;
  whoop::T(0) << "Output Width: " << kOutputWidth << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  // Short-form variable names
  const int A = kInputAWidth;
  const int B = kInputBWidth;
  const int P = kOutputWidth;

  whoop::T(0) << "RUNNING..." << whoop::EndT;

  for (int a = 0; a < A; a++)
  {
    for (int b = 0; b < B; b++)
    {
      outputs.At(a * B + b) += input_a.At(a) * input_b.At(b);
    }
  }

  for (int x = 0; x < A; x++)
  {
    whoop::T(2) << "I_A " << x << " = " << input_a.At(x) << whoop::EndT;
  }
  for (int x = 0; x < B; x++)
  {
    whoop::T(2) << "I_B " << x << " = " << input_b.At(x) << whoop::EndT;
  }
  for (int x = 0; x < P; x++)
  {
    whoop::T(2) << "O " << x << " = " << outputs.At(x) << whoop::EndT;
  }
  
  whoop::Done();
}
