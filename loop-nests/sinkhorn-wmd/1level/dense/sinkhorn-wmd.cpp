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
    
  int p_rnz = 8; //R query nonzeroes
  int p_dbsize = 16; //Number of docs in Database
  int p_vocabsize = 32; //Number of words in dictionary

  whoop::AddOption(&p_rnz, "rnz,r", "R query nonzeroes.");
  whoop::AddOption(&p_dbsize, "dbsize,d", "Number of docs in Database.");
  whoop::AddOption(&p_vocabsize, "vocabsize,v", "Number of words in dictionary.");

  whoop::Init(argc, argv);

  Tensor input_X("X"); //"X iteration solution in");
  Tensor input_K("K"); // dictionary");
  Tensor input_KT("K.T"); // dictionary");
  Tensor input_R("R"); // R query");
  Tensor input_C("C"); // database of documents to compare against");
  Tensor output_X("Xp"); // X iteration solution out");

  Tensor PR4("PR4"); //PR3 ElementMul C partial result");
  Tensor PR5("PR5"); //K @ PR4 partial result");

  
  const int RNZ = p_rnz; //R query nonzeroes
  const int DBSIZE = p_dbsize; //Number of docs in Database
  const int VOCABSIZE = p_vocabsize; //Number of words in dictionary

  //set sizes (reversed ordering for whoop c*r)
  input_X.Resize({RNZ, DBSIZE});
  input_K.Resize({RNZ, VOCABSIZE});
  input_KT.Resize({VOCABSIZE, RNZ});
  input_R.Resize({RNZ});
  output_X.Resize({RNZ, DBSIZE});
  PR4.Resize({VOCABSIZE, DBSIZE});
  input_C.Resize({VOCABSIZE, DBSIZE});
  PR5.Resize({RNZ, DBSIZE});
  

  whoop::T(0) << "R Query Nonzeroes: "  << RNZ << whoop::EndT;
  whoop::T(0) << "Num Docs in Database: " << DBSIZE << whoop::EndT;
  whoop::T(0) << "Num Words in word2vec Dictionary: "  << VOCABSIZE << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  whoop::ASSERT(RNZ == RNZ) << "The width of Matrix A must equal the height of Matrix B. A width: " << RNZ << ", B height: " << RNZ << whoop::EndT;

  //matrix multiply and elewise
  Var m("m");
  Var k("k");
  Var n("n");



  t_for(m, 0, VOCABSIZE);
  {
      t_for(n, 0, DBSIZE);
      {
          //PR1: 1/X
          //PR2: MatMul (mnk)

          t_for(k, 0, RNZ);
          {
              PR4[m][n] += input_KT[m][k] * input_X[k][n]; //XXX * should be /
          }
          end();

          //PR3: 1/PR2
          //PR4: PR3 * C
          PR4[m][n] = input_C[m][n] * PR4[m][n]; //XXX * should be /
      }
      end();
  }
  end();


  t_for(m, 0, RNZ);
  {
      t_for(n, 0, DBSIZE);
      {
          //PR5: K @ PR4
          t_for(k, 0, VOCABSIZE);
          {
              output_X[m][n] += input_K[m][k] * PR4[k][n];
          }
          end();

          //Output X: PR5 * 1/R
          output_X[m][n] = output_X[m][n]*input_R[m];

      }
      end();
  }
  end();

           
  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::T(0) << output_X.Size(0) << whoop::EndT;  // K


  whoop::Done();
}
