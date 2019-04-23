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


#include "conv6d.hpp"

#include "assert.h"

#include "whoop.hpp"


int main(int argc, char** argv)
{

  using namespace whoop;
  
  TensorIn  inputs("inputs");
  TensorIn  weights("weights");
  TensorOut outputs("outputs");

  whoop::Init(argc, argv);
  
  const int kInputWidth = inputs.Size(dims::inputs::W);
  const int kInputHeight = inputs.Size(dims::inputs::H);
  const int kInputChannel = inputs.Size(dims::inputs::C);
  const int kWeightHeight = weights.Size(dims::weights::S);
  const int kWeightWidth = weights.Size(dims::weights::R);
  const int kWeightChannel = weights.Size(dims::weights::C);
  const int kOutputWidth = kInputWidth - kWeightWidth + 1;
  const int kOutputHeight = kInputHeight - kWeightHeight + 1;
  const int kOutputChannel = weights.Size(dims::weights::K);

  assert(kWeightChannel == kInputChannel);
  
  // Short-form variable names
  const int K = kOutputChannel;
  const int C = kInputChannel;
  const int R = kWeightWidth;
  const int S = kWeightHeight;
  const int W = kInputWidth;
  const int H = kInputHeight;
  const int P = kOutputWidth;
  const int Q = kOutputHeight;
  
  outputs.Resize({kOutputChannel, kOutputHeight, kOutputWidth});
 
  whoop::T(0) << "RUNNING..." << whoop::EndT;

  for(int k=0; k<K; k++)
  {
    for(int q=0; q<Q; q++)
    {
      for(int p=0; p<P; p++)
      {
        for(int c=0; c<C; c++)
        {
          for(int r=0; r<R; r++)
          {
            for(int s=0; s<S; s++)
            {
              outputs.At({k, q, p}) +=
                inputs.At({c, q+s, p+r}) * weights.At({k, c, s, r});
            }
          }
        }
      }
    }
  }

  whoop::T(0) << "DONE..." << whoop::EndT;

  for(int k=0; k<K; k++)
  {
    for(int q=0; q<Q; q++)
    {
      for(int p=0; p<P; p++)
      {
        whoop::T(2) << "O [" << k << "][" << q << "][" << p << "] = " << outputs.At({k,q,p}) << whoop::EndT;
      }
    }
  }

  whoop::Done();
}
