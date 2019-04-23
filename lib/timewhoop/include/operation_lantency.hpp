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

#ifndef OPERATION_LATENCY_HPP_
#define OPERATION_LATENCY_HPP_

#define TIMEWHOOP_SET_UNARY_OPERATION_LATENCY(OP, LATENCY) \
  case UnaryOperator::OP: return LATENCY;

#define TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(OP, LATENCY) \
  case BinaryOperator::OP: return LATENCY;

#include "pure-abstract-syntax-types.hpp"


namespace timewhoop
{
/*** Operation Latency
- Description: This file is to list up operations and its delay for
runtime estimation. 

- This file provides some macros for latency list.
*/
  
  int GetBinaryOpLatency(BinaryOperator op)
  {
    switch(op)
    {
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(PLUS, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(MINUS, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(MULT, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(DIV, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(BITWISE_AND, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(BITWISE_OR, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(BITWISE_XOR, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(EQ, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(NEQ, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(LEQ, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(GEQ, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(GT, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(LT, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(LOGICAL_AND, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(LOGICAL_OR, 1)
      TIMEWHOOP_SET_BINARY_OPERATION_LATENCY(LOGICAL_XOR, 1)

      default:
        return 1;
    }
  }

  int GetBinaryOpLatency(BinaryOperator op)
  {
    switch(op)
    {
      TIMEWHOOP_SET_UNARY_OPERATION_LATENCY(LOGICAL_NEGATION, 0)
      TIMEWHOOP_SET_UNARY_OPERATION_LATENCY(PRE_INCREMENT, 1)
      TIMEWHOOP_SET_UNARY_OPERATION_LATENCY(POST_INCREMENT, 1)
      TIMEWHOOP_SET_UNARY_OPERATION_LATENCY(PRE_DECREMENT, 1)
      TIMEWHOOP_SET_UNARY_OPERATION_LATENCY(POST_DECREMENT, 1)

      default:
        return 1;
    }
  }

  int GetPipelinedOpLatnecy(std::shared_ptr<std::list<int>> latency_list, int num_ops) 
  {
    int total_latency = 0;

    int pipeline_latnecy = 0;
    for(auto& latency : *latency_list)
    {
      pipeline_latency += latency;
    }

    if(num_ops <= pipeline_latency)
    {
      total_latency = pipeline_latency;
    }
    else
    {
      total_latency = num_ops + (pipeline_latency-1); // Tail latency
    }

    return total_latency;
  }

};

#endif
