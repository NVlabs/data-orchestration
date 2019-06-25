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

  Tensor PR1("PR1"); //1/X partial result");
  Tensor PR2("PR2"); //K.T @ 1/X partial result");
  Tensor PR3("PR3"); //1/MatMul of PR2 partial result");
  Tensor PR4("PR4"); //PR3 ElementMul C partial result");
  Tensor PR5("PR5"); //K @ PR4 partial result");

  
  const int RNZ = p_rnz; //R query nonzeroes
  const int DBSIZE = p_dbsize; //Number of docs in Database
  const int VOCABSIZE = p_vocabsize; //Number of words in dictionary

  //set sizes (reversed ordering for whoop c*r)
  input_X.Resize({DBSIZE, RNZ});
  input_K.Resize({VOCABSIZE, RNZ});
  input_KT.Resize({RNZ, VOCABSIZE});
  input_R.Resize({RNZ});
  output_X.Resize({DBSIZE, RNZ});
  PR1.Resize({DBSIZE, RNZ});
  PR2.Resize({DBSIZE, VOCABSIZE});
  PR3.Resize({DBSIZE, VOCABSIZE});
  PR4.Resize({DBSIZE, VOCABSIZE});
  input_C.Resize({DBSIZE, VOCABSIZE});
  PR5.Resize({DBSIZE, RNZ});
  
/*  PR1.SetUpdatedDynamically();
  PR2.SetUpdatedDynamically();
  PR3.SetUpdatedDynamically();
  PR4.SetUpdatedDynamically();
  PR5.SetUpdatedDynamically();
  output_X.SetUpdatedDynamically();
*/

  whoop::T(0) << "R Query Nonzeroes: "  << RNZ << whoop::EndT;
  whoop::T(0) << "Num Docs in Database: " << DBSIZE << whoop::EndT;
  whoop::T(0) << "Num Words in word2vec Dictionary: "  << VOCABSIZE << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  whoop::ASSERT(RNZ == RNZ) << "The width of Matrix A must equal the height of Matrix B. A width: " << RNZ << ", B height: " << RNZ << whoop::EndT;

  //element wise operations
  Var r("r");
  Var c("c");
  //matrix multiply
  Var m("m");
  Var k("k");
  Var n("n");


  //PR1: 1/X
  t_for(r, 0, RNZ);
  {
      t_for(c, 0, DBSIZE);
      {
          PR1[r][c] = input_X[r][c];
      }
      end();
  }
  end();

  //PR2: MatMul (mnk)
  t_for(m, 0, VOCABSIZE);
  {
      t_for(n, 0, DBSIZE);
      {
          t_for(k, 0, RNZ);
          {
              PR2[m][n] += input_KT[m][k] * PR1[k][n];
          }
          end();
      }
      end();
  }
  end();


  //PR3: 1/PR2
  t_for(r, 0, VOCABSIZE);
  {
      t_for(c, 0, DBSIZE);
      {
          PR3[r][c] = PR2[r][c];
      }
      end();
  }
  end();

  //PR4: PR3 * C
  t_for(r, 0, VOCABSIZE);
  {
      t_for(c, 0, DBSIZE);
      {
          PR4[r][c] = PR4[r][c]*input_C[r][c];
      }
      end();
  }
  end();

  //PR5: K @ PR4
  t_for(m, 0, RNZ);
  {
      t_for(n, 0, DBSIZE);
      {
          t_for(k, 0, VOCABSIZE);
          {
              PR5[m][n] += input_K[m][k] * PR4[k][n];
          }
          end();
      }
      end();
  }
  end();

  //Output X: PR5 * 1/R
  t_for(r, 0, RNZ);
  {
      t_for(c, 0, DBSIZE);
      {
          output_X[r][c] = PR5[r][c]*input_R[r];
      }
      end();
  }
  end();

           
  whoop::T(0) << "RUNNING..." << whoop::EndT;
  whoop::Run();
  whoop::T(0) << output_X.Size(0) << whoop::EndT;  // K


  whoop::Done();
}
