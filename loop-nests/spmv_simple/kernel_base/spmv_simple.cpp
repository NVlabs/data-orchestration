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
#include "compressed-tensor.hpp"

//An output-stationary inner product 
int main(int argc, char** argv)
{

  using namespace whoop;
    
  CompressedTensorIn sparse("input", 2);
  Tensor input("vector_in");         
  Tensor output("vector_out"); 
  
  whoop::Init(argc, argv);

  const int SIZE_H = sparse.Size(0); 
  const int SIZE_W = sparse.Size(1); 

  //set vectors
  output.Resize({SIZE_H});
  input.Resize({SIZE_W});
  for (int x = 0; x < SIZE_W; x++)
  {
    input.PrimAt(x) = 1;
  }

  whoop::T(0) << "SPARSE MATRIX  W:"  << SIZE_W << "  H:" << SIZE_H << whoop::EndT;
  whoop::T(0) << whoop::EndT;


  //sparse matrix vector multiply
  Var m("m");
  Var n("n");

  Var m_pos("m_pos");
  Var n_pos("n_pos");

  Var val("val");
  Var temp("temp");

  // XXX Do a Spare matrix - vector multiply
  //

  t_for(m_pos, sparse.GetSegmentBeginAt(0,0), sparse.GetSegmentEndAt(0,0)); //SIZE_H max
  {
      temp = 0*val;

      m = sparse.GetCoordinate(0, m_pos); //get m coordinate (row from sparse)

      //get the nonzeroes for each row via scan
      t_for(n_pos, sparse.GetSegmentBegin(1,m_pos), sparse.GetSegmentEnd(1,m_pos)); //SIZE_W max
      {
          n = sparse.GetCoordinate (1,n_pos);
          val = sparse.GetValue(n_pos);

          temp += input[n] * val; 
      }
      end();

      output[m] = temp;

  }
  end();


           
  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();

  whoop::T(0) << "Complete..." << whoop::EndT;
  sparse.PrintCompressedTensor(true);

  whoop::T(0) << "OUTPUT" << whoop::EndT;
  for (int x = 0; x < SIZE_H; x++)
  {
      std::cout << "  " << output.PrimAt(x);
  }
  std::cout << std::endl;

  whoop::Done();
}
