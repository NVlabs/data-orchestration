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


#include <string>

#include "pure-abstract-syntax-types.hpp"

namespace timewhoop
{

  std::string ToString(ExpressionClass excls)
  {
    std::string ret;
    switch(excls)
    {
      case ExpressionClass::INTEGER:
        ret = "Integer";
        break;
      case ExpressionClass::FLOAT:
        ret = "Float";
        break;
      case ExpressionClass::CONTAINER:
        ret = "Container";
        break;
      case ExpressionClass::VARIABLE:
        ret = "Variable";
        break;
      case ExpressionClass::TENSOR:
        ret = "Tensor";
        break;
      case ExpressionClass::TENSOR_ACCESS:
        ret = "TensorAccess";
        break;
      case ExpressionClass::UNARYOP:
        ret = "UnaryOp";
        break;
      case ExpressionClass::BINARYOP:
        ret = "BinaryOp";
        break;
      case ExpressionClass::EXPRESSION:
        ret = "Expression";
        break;
      default:
        break;
    }

    return ret;
  }

  std::string ToString(StatementClass stcls)
  {
    std::string ret;
    switch(stcls)
    {
      case StatementClass::DECLARATION:
        ret = "Declaration";
        break;
      case  StatementClass::PRIM_FOR:
        ret = "Prim_for (should not be exposed)";
        break;
      case  StatementClass::TEMPORAL_FOR:
        ret = "Temporal for";
        break;
      case  StatementClass::SPATIAL_FOR:
        ret = "Spatial for";
        break;
      case  StatementClass::PRIM_ASSIGNMENT:
        ret = "Prim_Assignment (should not be exposed)";
        break;
      case  StatementClass::VAR_ASSIGNMENT:
        ret = "Variable assignment";
        break;
      case  StatementClass::TENSOR_ASSIGNMENT:
        ret = "Tensor assignment";
        break;
      case  StatementClass::COND_STATEMENT:
        ret = "Cond_Statement (should not be exposed)";
        break;
      case  StatementClass::IF:
        ret = "If";
        break;
      case  StatementClass::ELSE:
        ret = "Else";
        break;
      default:
        ret = "";
        break;
    }

    return ret;
  }

  bool isPreApplied(UnaryOperator op)
  {
    bool ret;

    switch(op)
    {
      case UnaryOperator::LOGICAL_NEGATION:
      case UnaryOperator::PRE_INCREMENT:
      case UnaryOperator::PRE_DECREMENT:
        ret = true;
        break;
      case UnaryOperator::POST_INCREMENT:
      case UnaryOperator::POST_DECREMENT:
      default:
        ret = false;
        break;
    }

    return ret;
  }

  std::string ToString(UnaryOperator op)
  {
    std::string ret;
    switch(op)
    {
      case UnaryOperator::LOGICAL_NEGATION:
        ret = "Logical_negation";
        break;
      case UnaryOperator::PRE_INCREMENT:
        ret = "Pre_increment";
        break;
      case UnaryOperator::POST_INCREMENT:
        ret = "Post_increment";
        break;
      case UnaryOperator::PRE_DECREMENT:
        ret = "Pre_decrement";
        break;
      case UnaryOperator::POST_DECREMENT:
        ret = "Post_decrement";
        break;
      default:
        ret = "";
        break;
    }

    return ret;
  }

  std::string ToString(BinaryOperator op)
  {
    std::string ret;
    switch(op)
    {
      case BinaryOperator::PLUS:
        ret = "Plus";
        break;
      case BinaryOperator::MINUS:
        ret = "Minus";
        break;
      case BinaryOperator::MULT:
        ret = "Mult";
        break;
      case BinaryOperator::DIV: 
        ret = "Div";
        break;
      case BinaryOperator::BITWISE_AND:
        ret = "BitwiseAnd";
        break;
      case BinaryOperator::BITWISE_OR:
        ret = "BitwiseOr";
        break;
      case BinaryOperator::BITWISE_XOR:
        ret = "BitwiseXor";
        break;
      case BinaryOperator::EQ:
        ret = "Equal";
        break;
      case BinaryOperator::NEQ:
        ret = "NotEqual";
        break;
      case BinaryOperator::LEQ:
        ret = "LessOrEqual";
        break;
      case BinaryOperator::GEQ:
        ret = "GreaterOrEqual";
        break;
      case BinaryOperator::GT:
        ret = "GreaterThan";
        break;
      case BinaryOperator::LT:
        ret = "LessThan";
        break;
      case BinaryOperator::LOGICAL_AND:
        ret = "LogicalAnd";
        break;
      case BinaryOperator::LOGICAL_OR:
        ret = "LogicalOr";
        break;
      case BinaryOperator::LOGICAL_XOR:
        ret = "LogicalXor";
        break;
      default:
        ret = "Undefined binary operator";
        break;
    }

    return ret;
  }


};
