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
#include "flexmath.hpp"

//An output-stationary inner product 
int main(int argc, char** argv)
{

  using namespace whoop;
  
  int kTileHeight = 4;
  int kTileWidth = 4;
  int kTileDepth = 4;
  
  whoop::AddOption(&kTileHeight, "tile_height,h", "height of the L0 tile.");
  whoop::AddOption(&kTileWidth, "tile_width,w", "Width of the L0 tile.");
  whoop::AddOption(&kTileDepth, "tile_depth,d", "Depth of the L0 tile.");

  TensorIn  input_a("input_a");
  TensorIn  input_b("input_b");
  TensorOut outputs("outputs");

  whoop::Init(argc, argv);
  flexmath::Init();
  
  int kInputAWidth = input_a.Size(0);  // K
  int kInputBWidth = input_b.Size(0);  // N
  int kInputAHeight = input_a.Size(1); // M
  int kInputBHeight = input_b.Size(1); // K
  
  const int kOutputWidth = kInputBWidth;
  const int kOutputHeight = kInputAHeight;
  outputs.Resize({kOutputHeight, kOutputWidth});

  whoop::T(0) << "Input A Width: "  << kInputAWidth << whoop::EndT;
  whoop::T(0) << "Input A Height: " << kInputAHeight << whoop::EndT;
  whoop::T(0) << "Input B Width: "  << kInputBWidth << whoop::EndT;
  whoop::T(0) << "Input B Height: " << kInputBHeight << whoop::EndT;
  whoop::T(0) << "Output Width: "   << kOutputWidth << whoop::EndT;
  whoop::T(0) << "Output Height: "  << kOutputHeight << whoop::EndT;
  whoop::T(0) << "Tile Depth: "   << kTileDepth << whoop::EndT;
  whoop::T(0) << "Tile Height: "  << kTileWidth << whoop::EndT;
  whoop::T(0) << "Num Tiles Back: "   << kInputAWidth / kTileDepth << whoop::EndT;
  whoop::T(0) << "Num Tiles Across: "  << kInputBWidth / kTileWidth << whoop::EndT;
  whoop::T(0) << whoop::EndT;

  whoop::ASSERT(kInputAWidth == kInputBHeight) << "The width of Matrix A must equal the height of Matrix B. A width: " << kInputAWidth << ", B height: " << kInputBHeight << whoop::EndT;
  whoop::ASSERT(kInputAWidth % kTileDepth == 0) << "The depth of the tile must equally divide the width of matrix A. A width: " << kInputAWidth << ", tile depth: " << kTileDepth << whoop::EndT;
  whoop::ASSERT(kInputBWidth % kTileWidth == 0) << "The width of the tile must equally divide the width of matrix B. B width: " << kInputBWidth << ", tile width: " << kTileWidth << whoop::EndT;

  // Short-form variable names
  const int M = kInputAHeight;
  const int K = kInputAWidth;
  const int N = kInputBWidth;

  //FlexMath tile size
  const int MF = flexmath::TileHeight;
  const int NF = flexmath::TileWidth;
  const int KF = flexmath::TileDepth;

  //RF Buffet num tiles
  const int M0 = kTileHeight/MF;
  const int N0 = kTileWidth/NF;
  const int K0 = kTileDepth/KF;

  //Global num tiles
  const int M1 = M / kTileHeight;
  const int N1 = N / kTileWidth;
  const int K1 = K / kTileDepth;



  Var m("m");
  Var k("k");
  Var n("n");

  Var m1("m1");
  Var k1("k1");
  Var n1("n1");


  Var m0("m0");
  Var k0("k0");
  Var n0("n0");

  Var flexmath_mf("fm_mf");
  Var flexmath_kf("fm_kf");
  Var flexmath_nf("fm_nf");

  Var flexmath_m("fm_m");
  Var flexmath_k("fm_k");
  Var flexmath_n("fm_n");

  t_for(m1, 0, M1);
  {
      t_for(n1, 0, N1);
      {
        // Output-stationary, col-major L1.
        //input_a.AddTileLevel(K1 * M * K0);
        // purposely skip input b
        //outputs.AddTileLevel(M * N0);
        // purposely skip outputs.
          t_for(k1, 0, K1);
          {
              t_for(m0, 0, M0);
              {
                  //RF buffets
                  input_a.AddTileLevel(K0); // Reuse row tile until done, Reloaded N1 times
                  input_b.AddTileLevel(K0*N0); // Never reloaded (B-stationary L1) Reuse for each row of A
                  outputs.AddTileLevel(1); // Doesn't need to be Reloaded 
                  t_for(n0, 0, N0);
                  {
                      t_for(k0, 0, K0);
                      {
                          //which RF tile to send to flexmath
                          m = m1 * M0 + m0;
                          k = k1 * K0 + k0;
                          n = n1 * N0 + n0;


                          //XXX flexmath::MatrixMul(input_a, input_b, outputs, m, n, k);
                          input_b.AddTileLevel(KF);  //B Tile is shared across all spatial
                          t_for(flexmath_nf, 0, NF); //iterate across Bs
                          {

                              s_for(flexmath_mf, 0, MF);
                              {
                                  input_a.AddTileLevel(MF*KF);  //each iteration (A row worker) gets KF-sized buffer
                                  outputs.AddTileLevel(MF);  //each iteration (A row worker) gets single entry of buffer
                                  t_for(flexmath_kf, 0, KF);
                                  {
                                      flexmath_m = m * MF + flexmath_mf; 
                                      flexmath_n = n * NF + flexmath_nf;
                                      flexmath_k = k * KF + flexmath_kf;
                                      outputs[flexmath_m][flexmath_n] += input_a[flexmath_m][flexmath_k] * input_b[flexmath_k][flexmath_n];
                                  }
                                  end();
                              }
                              end();
                          }
                          end();          
                          //XXX end flexmath
                      }
                      end();
                  }
                  end();
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
  whoop::Done();
}
