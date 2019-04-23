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

#include<cstdlib>

#include "whoop.hpp"


/*
  Output-stationary convolution
  
    1. Overview
      Dataflow: K -> X -> *Y -> (C -> R -> S)
      (MAESTRO Notation: T_Map(1, 1) K -> T_Map(1,1) X -> S_Map(PartitionSize, PartitionSize,) Y -> Unroll C -> Unroll R -> Unroll S)
      * This is output stationary because output-related variables (K, X, Y) are fixed in each PE at spatially mapped loop.

    2. Command line arguments: numPEs K C R S Y X, where
      numPEs: Number of PEs
      K: Number of output channels
      C: Number of input channels
      R: Weight Height
      S: Weight Width
      Y: Input Height
      X: Input Width


  ******** Debug comment ********
  w_else seems to have a potential issue. Please compile it with block 1 (separate if-based implementation)
  and with block 2 (w_else-based implementation) and check the log message.
  (You can find explicit comment "block 1 start" and "block 1 end"; please activate one of them)
*/

int main(int argc, char** argv)
{

  using namespace whoop;

  if(argc != 8) {
    std::cout << "Need seven variables; NumPEs K C R S Y X" << std::endl;
    return -1;
  }

  int kNumPEs = std::atoi(argv[1]);

  int kNumOutputChannels = std::atoi(argv[2]);
  int kNumInputChannels = std::atoi(argv[3]);
  int kWeightHeight = std::atoi(argv[4]);
  int kWeightWidth = std::atoi(argv[5]);
  int kInputHeight = std::atoi(argv[6]);
  int kInputWidth = std::atoi(argv[7]);

  const int kNumWeightsPerWeightRow = kWeightWidth;
  const int kNumWeightsPerInputChannel = kWeightHeight * kNumWeightsPerWeightRow;
  const int kNumWeightsPerOutputChannel = kNumInputChannels * kNumWeightsPerInputChannel;
  const int kNumWeights = kNumOutputChannels * kNumWeightsPerOutputChannel;


  const int kInputWidthUpperBound = kInputWidth - kWeightWidth + 1;
  const int kInputHeightUpperBound = kInputHeight - kWeightHeight + 1;
  const int kNumInputsPerInputRow = kInputWidth;
  const int kNumInputsPerInputChannel = kInputHeight * kInputWidth;
  const int kNumInputs = kNumInputChannels * kNumInputsPerInputChannel;

  const int kNumOutputsPerColumn = kInputHeightUpperBound;
  const int kNumOutputsPerRow = kInputWidthUpperBound;
  const int kNumOutputsPerOutputChannel = kInputHeightUpperBound * kNumOutputsPerRow;
  const int kNumOutputs = kNumOutputChannels * kNumOutputsPerOutputChannel;

//  assert(kNumOutputsPerColumn % kNumPEs == 0);

  const int kPartitionSize = kNumOutputsPerColumn / kNumPEs;
  const int kLastPartitionSize = (kNumOutputsPerColumn % kNumPEs == 0)? kPartitionSize : kNumOutputsPerColumn % kNumPEs;


  Vec inputs(kNumInputs, RandomValue<int>, "inputs");
  Vec weights(kNumWeights, [](int idx) { return (idx % 2 == 0) ? 2 : 4; }, "weights");
  Vec outputs(kNumOutputs, 0, "outputs");

  Var k("k");
  Var c("c");
  Var r("r");
  Var s("s");
  Var y("y");
  Var x("x");
  Var p("p");

  // (K -> Y -> X) -> (C -> R -> S) : Output-stationary
  t_for(k, 0, kNumOutputChannels);
  {
    t_for(x, 0, kInputWidthUpperBound);
//    t_for(y, 0, kInputHeightUpperBound);
    {
      s_for(p, 0, kNumPEs);
      {
        t_for(y, 0, kPartitionSize);
        {
          inputs.AddBufferLevel(kNumInputChannels * kInputHeight * kInputWidth); //For verification
//        inputs.AddBufferLevel(kNumInputChannels * kWeightHeight * kWeightWidth);
          weights.AddBufferLevel(kNumInputChannels * kWeightHeight * kWeightWidth);
          outputs.AddBufferLevel(1);
  
          t_for(c, 0, kNumInputChannels);
          {  
            t_for(r, 0, kWeightHeight);
            {
              t_for(s, 0, kWeightWidth);
              {
                //1. Separate If-based implementation
                // Block 1 start
                w_if(kNumPEs >= p);
                {
                  outputs[k * kNumOutputsPerOutputChannel + p * kNumOutputsPerRow * kPartitionSize + y * kNumOutputsPerRow + x] 
                    += weights[k * kNumWeightsPerOutputChannel + c * kNumWeightsPerInputChannel + r * kNumWeightsPerWeightRow + s]
                     * inputs[c * kNumInputsPerInputChannel + (r + p * kPartitionSize + y ) * kNumInputsPerInputRow + x + s];
                }
                end();
                // Block 1 end

                //Edge condition
                w_if(p >= kNumPEs-1); //w_if(isEdge)
                {
                  w_if(kLastPartitionSize -1 >= y);
                  {
                    outputs[k * kNumOutputsPerOutputChannel + p * kNumOutputsPerRow * kPartitionSize + y * kNumOutputsPerRow + x] 
                      += weights[k * kNumWeightsPerOutputChannel + c * kNumWeightsPerInputChannel + r * kNumWeightsPerWeightRow + s]
                       * inputs[c * kNumInputsPerInputChannel + (r + p * kPartitionSize + y ) * kNumInputsPerInputRow + x + s];
                  }
                  end();
                } 
//                end(); //End w_if(isEdge) ;; Tried with/without this line but w_else didn't work
                // 2. w_else-based implementation
                /***********************************************
                // Block 2 start
                w_else();
                {
                  outputs[k * kNumOutputsPerOutputChannel + p * kNumOutputsPerRow * kPartitionSize + y * kNumOutputsPerRow + x] 
                    += weights[k * kNumWeightsPerOutputChannel + c * kNumWeightsPerInputChannel + r * kNumWeightsPerWeightRow + s]
                     * inputs[c * kNumInputsPerInputChannel + (r + p * kPartitionSize + y ) * kNumInputsPerInputRow + x + s];
                } 
                end();  //End w_else for steady cases
                // Block 2 end
                *************************************************/
              }
              end(); //End t_for(s)
            }
            end(); //End t_for(r)
          }
          end(); //End t_for(c)
        }
        end(); //End t_for(x)
      }
      end(); //End s_for(p)
    }
    end(); //End t_for(y)
  }
  end(); //End t_for(k)
 
 
  std::cout << "RUNNING..." << std::endl;
  whoop::the_program.Run();
  whoop::DumpStats();
  std::cout << "DONE." << std::endl;
  for (int x = 0; x < kNumWeights; x++)
  {
    std::cout <<"W " << x << " = " << weights.At(x) << std::endl;
  }
  for (int x = 0; x < kNumInputs; x++)
  {
    std::cout << "I " << x << " = " << inputs.At(x) << std::endl;
  }
  for (int x = 0; x < kNumOutputs; x++)
  {
    std::cout << "O " << x << " = " << outputs.At(x) << std::endl;
  }
  std::cout << "Correct stat numbers with large enough L1" << std::endl;
  std::cout << "Input Vec Access: " << kNumInputChannels * kWeightHeight * kWeightWidth * kNumOutputs  << std::endl;
  std::cout << "Input Offchip Access: " << kNumInputChannels * kInputHeight * kInputWidth << std::endl;
  std::cout << "Weight Vec Access: " << kNumInputChannels * kWeightHeight * kWeightWidth * kNumOutputs  << std::endl;
  std::cout << "Weight Offchip Access: " << kNumOutputChannels * kNumInputChannels * kWeightHeight * kWeightWidth  << std::endl;
  
}
