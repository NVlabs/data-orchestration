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

#ifndef TIMEWHOOP_PURE_ABSTRACT_SYNTAX_TYPES_HPP_
#define TIMEWHOOP_PURE_ABSTRACT_SYNTAX_TYPES_HPP_

#include <string>

namespace timewhoop
{
  enum class StatementClass
  {
    DECLARATION,
    VARIABLE_DECLARATION,
    TENSOR_DECLARATION,
    PRIM_FOR,
    TEMPORAL_FOR,
    SPATIAL_FOR,
    PRIM_ASSIGNMENT,
    VAR_ASSIGNMENT,
    TENSOR_ASSIGNMENT,
    COND_STATEMENT,
    IF,
    ELSE,
    STATEMENT
  };

  enum class ExpressionClass
  {
    INTEGER,
    FLOAT,
    CONTAINER,
    VARIABLE,
    TENSOR,
    TENSOR_ACCESS,
    UNARYOP,
    BINARYOP,
    EXPRESSION
  };
  
  enum class UnaryOperator
  {
    LOGICAL_NEGATION,
    PRE_INCREMENT,
    POST_INCREMENT,
    PRE_DECREMENT,
    POST_DECREMENT,
    INVALID_OP
  };

  enum class BinaryOperator
  {
    PLUS,
    MINUS,
    MULT,
    DIV,  
    BITWISE_AND,
    BITWISE_OR,
    BITWISE_XOR,
    EQ,
    NEQ,
    LEQ,
    GEQ,
    GT,
    LT,
    LOGICAL_AND,
    LOGICAL_OR,
    LOGICAL_XOR,
    INVALID_OP
  };

  enum class Type
  {
    VOID,
    BOOLEAN,
    CHARACTER,
    INTEGER,
    STRING,
    VAR,
    TENSOR
  };

  std::string ToString(ExpressionClass excls);
  std::string ToString(StatementClass stcls);
  bool isPreApplied(UnaryOperator op);
  std::string ToString(UnaryOperator op);
  std::string ToString(BinaryOperator op);

};  //namespace timewhoop
#endif /* TIMEWHOOP_AST_TYPES_HPP_ */
