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

  CompressedTensorIn input_A("input_a"); //input vector a
  CompressedTensorIn input_B("input_b"); //input vector b

  whoop::Init(argc, argv);
 
  whoop::ASSERT(input_A.Size(0) == input_B.Size(0)) << "The input vectors are of different sizes! A size: " << input_A.Size(0) << ", B size: " << input_B.Size(0) << whoop::EndT;
  const int SIZE = input_A.Size(0); // vector size (shared

  whoop::T(0) << "Vector Sizer: "  << SIZE << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  Var a_pos("apos");
  Var a_k("ak");

  Var b_pos("bpos");
  Var b_k("bk");

  Var output("output");
  /*
k2_pa = a.GetSegStart(“K2”, 0);
k2_pb = b.GetSegStart(“K2”, n2_p);
w_while (k2_pa < a.GetSegEnd(“K2”, 0) &&
         k2_pb < b.GetSegEnd(“K2”, n2_p));
  k2_a = a.GetCoord(“K2”, k2_pa);
  k2_b = b.GetCoord(“K2”, k2_pb);
  w_if (k2_a < k2_b);
    k2_pa++;
  w_else_if (k2_b < k2_a);
    k2_pb++;
  w_else();
    <BODY>
*/
  //perform intersection on A and B
  //intialize 
  a_pos = input_A.GetSegmentBeginAt(0,0);
  b_pos = input_B.GetSegmentBeginAt(0,0);

  //scan through vector A and vector B simultaneous
  w_while(a_pos < input_A.GetSegmentEndAt(0,0) && b_pos < input_B.GetSegmentEndAt(0,0)); //DBSIZE max
  {
      a_coord = input_A.GetCoordinate(0, a_pos); //get first element coordinate from vector A
      b_coord = input_B.GetCoordinate(0, b_pos); //get first element coordinate from vector B

      //check to see which way we advance
      w_if(a_coord < b_coord);
      {
          a_pos++;
      }
      w_else_if(a_coord > b_coord);
      {
          b_pos++;
      }
      w_else();
      {
          output += input_A.GetValue(a_pos)*input_B.GetValue(b_pos); 
          a_pos++;
          b_pos++;
      }
      end();
  }
  end();
           
  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();


  whoop::Done();
}
