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

#ifndef TIMEWHOOP_PURE_ABSTRACT_SYNTAX_HPP_
#define TIMEWHOOP_PURE_ABSTRACT_SYNTAX_HPP_

#include <iostream>
#include <list>
#include <vector>
#include <map>
#include <stack>
#include <algorithm>
#include <memory>

#include <boost/format.hpp>

#include "maybe-type.hpp"
#include "pure-abstract-syntax-types.hpp"
#include "pure-abstract-primitive-syntax.hpp"

#include "analysis-structure.hpp"


/**** Primitive pure abstract syntax ****
- Description
  - This file includes syntax classes for whoop EDSL for pure AST.
  - Unlike whoop's AST, timewhoop's AST does not include actual data and execution
    context, and include actual labels of operations (whoop's AST has them as function 
    pointers so it is hard to be processed)
  - The AST constructed using timewhoop's syntax classes allow various walk-through
    for further information extraction and analysis
  - Each class contains virtual APIs designed for some specific walk-throughs for various analysis

- Defined classes (:: represents inheritance)
  Expression::Integer
  Expression::Float (currently not supported; whoop does not have float data type yet)
  Expression::Container::Variable
  Expression::Container::Tensor
  Expression::Container::Tensor::TensorAccess
  Expression::UnaryOp
  Expression::BinaryOp

  Statement::Declaration::VariableDeclaration
  Statement::Declaration::TensorDeclaration
  Statement::PrimAssignment::VariableAssignement
  Statement::PrimAssignment::TensorAssignement
  Statement::PrimFor::TemporalFor
  Statement::PrimFor::SpatialFor
  Statement::ConditionalStatement::If
  Statement::ConditionalStatement::Else

*/



namespace timewhoop
{

  class Integer : public Expression
  {
    protected:
      const int val_;
    public:
      Integer(const int& v) : val_(v) {}
      //No setter; because an integer is constant
      int getValue()
      {
        return val_;
      }
  
      virtual ExpressionClass GetClass()
      {
        return ExpressionClass::INTEGER;
      }

      virtual void GetLeaves(std::shared_ptr<std::list<std::shared_ptr<Expression>>> leaf_list)
      {
      }

      virtual std::string GetName()
      {
        std::string ret = "";
        return ret;
      }

      virtual void GetTensorIndexExpressions (std::shared_ptr<Expression> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
      }
  
      virtual std::string ToString()
      {
        std::string ret = boost::str(boost::format("(Integer, val: %d)") % val_);
        return ret; 
      }

      virtual std::string ToCSV()
      {
        std::string ret = boost::str(boost::format("Integer, %d") % val_);
        return ret;
      }

      virtual Maybe<int> FirstOrderEval()
      {
        Maybe<int> ret(val_);
        return ret;
      }

      virtual void SetLoopBlockID(int blk_id)
      {
        loop_block_id_ = blk_id;
      }

      virtual void SetNumAccess(int num_access)
      {
        num_access_ = num_access;
      }

      virtual void SetAccessProb(double access_prob)
      {
        access_probability_ *= access_prob;
      }

      virtual Maybe<std::tuple<int, int>> MinMaxEval(std::shared_ptr<LoopInformationTable> loop_info_table, int loop_block_id)
      {
        Maybe<std::tuple<int, int>> ret(std::make_tuple(val_, val_+1));
        return ret;
      }

      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
      }
  };
  
  class Float : public Expression
  {
    protected:
      const float val_;
    public:
      Float(const float& v) : val_(v) {}
      //No setter; because an integer is constant
      int getValue()
      {
        return val_;
      }
  
      virtual ExpressionClass GetClass()
      {
        return ExpressionClass::FLOAT;
      }


      virtual void GetLeaves(std::shared_ptr<std::list<std::shared_ptr<Expression>>> leaf_list)
      {
      }

      virtual std::string GetName()
      {
        std::string ret = "";
        return ret;
      }

      virtual void GetTensorIndexExpressions (std::shared_ptr<Expression> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
      }
  
      virtual std::string ToString()
      {
        std::string ret = boost::str(boost::format("Float, val: %f") % val_);
        return ret; 
      }

      virtual std::string ToCSV()
      {
        std::string ret = boost::str(boost::format("Integer, %d") % val_);
        return ret;
      }

      virtual Maybe<int> FirstOrderEval()
      {
        Maybe<int> ret(static_cast<int>(val_));
        return ret;
      }

      virtual void SetLoopBlockID(int blk_id)
      {
        loop_block_id_ = blk_id;
      }

      virtual void SetNumAccess(int num_access)
      {
        num_access_ = num_access;
      }

      virtual void SetAccessProb(double access_prob)
      {
        access_probability_ *= access_prob;
      }

      virtual Maybe<std::tuple<int, int>> MinMaxEval(std::shared_ptr<LoopInformationTable> loop_info_table, int loop_block_id)
      {
        Maybe<std::tuple<int, int>> ret(std::make_tuple(val_, val_+1));
        return ret;
      }

      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
      }
  };
  
  class Variable : public Container //class Container : public Expression
  {
    public:
      Variable(Type tp, const std::string& nm) : Container(tp, nm) {}
  
      virtual ExpressionClass GetClass()
      {
        return ExpressionClass::VARIABLE;
      }

      virtual void GetLeaves(std::shared_ptr<std::list<std::shared_ptr<Expression>>> leaf_list)
      {
      }


      virtual std::string GetName()
      {
        return name_;
      }

      virtual void GetTensorIndexExpressions (std::shared_ptr<Expression> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
      }
  
      virtual std::string ToString()
      {
        std::string ret = boost::str(boost::format("(Variable, name: %s)") % name_);
        return ret; 
      }

      virtual std::string ToCSV()
      {
        std::string ret = boost::str(boost::format("Variable, %s") % name_);
        return ret;
      }

      virtual int GetNumDims()
      {
        return 1;
      }

      virtual int GetDimSize(int dim)
      {
        if(dim == 0)
        {
          return 1;
        }
        else
        {
          return -1;
        }
      }

      virtual Maybe<int> FirstOrderEval()
      {
        Maybe<int> ret(false);
        return ret;
      }

      virtual void SetLoopBlockID(int blk_id)
      {
        loop_block_id_ = blk_id;
      }

      virtual void SetNumAccess(int num_access)
      {
        num_access_ = num_access;
      }

      virtual void SetAccessProb(double access_prob)
      {
        access_probability_ *= access_prob;
      }

      //FIXME
      virtual Maybe<std::tuple<int, int>> MinMaxEval(std::shared_ptr<LoopInformationTable> loop_info_table, int loop_block_id)
      {
        Maybe<std::tuple<int, int>> ret;
        auto loop_info = loop_info_table->SearchTable(this, loop_block_id);
        if(loop_info.IsValid())
        {
          auto loop_info_contents = loop_info.GetValue(); 
          ret.SetValue(std::make_tuple(loop_info_contents->GetIterBase(), loop_info_contents->GetIterBound()));
        }
        return ret;
      }


      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
      }

  };
  
  class Tensor : public Container // class Container : public Expression
  {
    protected:
      std::vector<int> dim_sizes_;
  
    public:
  
      //Partial initialization; missing dimension sizes
      Tensor(Type tp, const std::string& nm) :
        Container(tp, nm) 
      {
      }
      //Full initialization
      Tensor(Type tp, const std::string& nm, const std::vector<int>& dim_sizes) :
        Container(tp, nm), dim_sizes_(dim_sizes) 
      {
      }
  
      virtual ExpressionClass GetClass()
      {
        return ExpressionClass::TENSOR;
      }

      virtual void GetLeaves(std::shared_ptr<std::list<std::shared_ptr<Expression>>> leaf_list)
      {
      }

      virtual std::string GetName()
      {
        return name_;
      }

      virtual void GetTensorIndexExpressions (std::shared_ptr<Expression> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
      }

      virtual std::string ToString()
      {
        std::string ret = boost::str(boost::format("(Tensor, name: %s)") % name_);
        return ret; 
      }

      virtual std::string ToCSV()
      {
        std::string ret = boost::str(boost::format("Tensor, %s") % name_);
        return ret;
      }

      virtual int GetNumDims()
      {
        return dim_sizes_.size();
      }
  
      virtual int GetDimSize(int dim)
      {
        return dim_sizes_.at(dim);
      }

      virtual Maybe<int> FirstOrderEval()
      {
        Maybe<int> ret(false);
        return ret;
      }

      virtual void SetLoopBlockID(int blk_id)
      {
        loop_block_id_ = blk_id;
      }

      virtual void SetNumAccess(int num_access)
      {
        num_access_ = num_access;
      }

      virtual void SetAccessProb(double access_prob)
      {
        access_probability_ *= access_prob;
      }

      virtual Maybe<std::tuple<int, int>> MinMaxEval(std::shared_ptr<LoopInformationTable> loop_info_table, int loop_block_id)
      {
        Maybe<std::tuple<int, int>> ret;
        return ret;
      }


      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
      } 
  };
   
  class TensorAccess : public Tensor
  {
    protected:
      std::shared_ptr<std::list<std::shared_ptr<Expression>>> idx_exprs_;
  
    public:
      TensorAccess(Type tp, const std::string& nm) :
        Tensor(tp, nm)
      {
      }
  
      TensorAccess(Type tp, const std::string& nm, const std::shared_ptr<std::list<std::shared_ptr<Expression>>>& idx_exprs) :
        Tensor(tp, nm),
        idx_exprs_(idx_exprs)
      {
        
      }
  
      virtual std::string ToString()
      {
        std::string ret = boost::str(boost::format("(Tensor access, loop_block_id = %d\n target tensor: %s") % loop_block_id_ % name_);
 
        std::string idx_expr_string = "\n---index expressions---\n";
        for(auto& idx_e : *idx_exprs_)
        {
          idx_expr_string += idx_e->ToString();
          idx_expr_string += "\n";
        }
        idx_expr_string += "------------------";
  
        ret += idx_expr_string;
  
        return ret; 
      }

      virtual std::string ToCSV()
      {
        std::string idx_expr_string;
        std::string ret = boost::str(boost::format("Tensor access, %s") % name_);
 
        for(auto& idx_e : *idx_exprs_)
        {
          idx_expr_string += ", (";
          idx_expr_string += idx_e->ToCSV();
          idx_expr_string += ")";
        }
  
        ret += idx_expr_string;
      }

      virtual void GetLeaves(std::shared_ptr<std::list<std::shared_ptr<Expression>>> leaf_list)
      {
      }

      virtual void GetTensorIndexExpressions (std::shared_ptr<Expression> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
        if(targ_tensor->GetName() == this->GetName())
        {
          access_expr_list->push_back(idx_exprs_); 
        }
      }

  
      virtual ExpressionClass GetClass()
      {
        return ExpressionClass::TENSOR_ACCESS;
      }

      virtual std::string GetName()
      {
        return name_;
      }

      virtual Maybe<int> FirstOrderEval()
      {
        Maybe<int> ret(false);
        return ret;
      }
 
      virtual void SetLoopBlockID(int blk_id)
      {
        loop_block_id_ = blk_id;
        for(auto& idx_expr : *idx_exprs_)
        {
          idx_expr->SetLoopBlockID(blk_id);
        }
      }

      virtual void SetNumAccess(int num_access)
      {
        num_access_ = num_access;
        for(auto& idx_expr : *idx_exprs_)
        {
          idx_expr->SetNumAccess(num_access);
        }
      }

      virtual void SetAccessProb(double access_prob)
      {
        access_probability_ *= access_prob;
        for(auto& idx_expr : *idx_exprs_)
        {
          idx_expr->SetAccessProb(access_prob);
        }
      }


      //TODO: Support multi-level indirections
      virtual Maybe<std::tuple<int, int>> MinMaxEval(std::shared_ptr<LoopInformationTable> loop_info_table, int loop_block_id)
      {
        Maybe<std::tuple<int, int>> ret;
        return ret;
      }
      
      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
        if(deep_search)
        {
          for(auto& idx_e : *idx_exprs_)
          {
            if(idx_e->GetClass() == search_class)
            {
              search_res->push_back(idx_e);
            }

            idx_e->SearchExpressions(search_res, search_class, deep_search);
          }
        }
      }

      void SetIdxExpr(std::shared_ptr<std::list<std::shared_ptr<Expression>>>& idx_expr)
      {
        idx_exprs_ = idx_expr;
      }
  
      std::shared_ptr<std::list<std::shared_ptr<Expression>>>& GetIdxExpr()
      {
        return idx_exprs_;
      }
  }; // End of class TensorAccess
  
  class UnaryOp : public Expression
  {
    protected:
      std::shared_ptr<Expression> src1_;
      UnaryOperator op_;
  
    public:
      UnaryOp(UnaryOperator op, std::shared_ptr<Expression> src1) :
        src1_(src1), op_(op)
      {
      }
  
      virtual ExpressionClass GetClass()
      {
        return ExpressionClass::UNARYOP;
      }

      virtual void GetLeaves(std::shared_ptr<std::list<std::shared_ptr<Expression>>> leaf_list)
      {
        if(src1_->GetClass() == ExpressionClass::VARIABLE)
        {
          leaf_list->push_back(src1_);
        }
        else
        {
          src1_->GetLeaves(leaf_list);
        }

      }

      virtual std::string GetName()
      {
        std::string ret = "";
        return ret;
      }

      virtual void GetTensorIndexExpressions (std::shared_ptr<Expression> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
        src1_->GetTensorIndexExpressions(targ_tensor, access_expr_list);
      }

      virtual std::string ToString()
      {
        std::string ret;
        ret = boost::str(boost::format("(UnaryOp, op: %s, \n operand: %s) ") % timewhoop::ToString(op_) % src1_->ToString());
        return ret;
      }

      virtual std::string ToCSV()
      {
        std::string ret = boost::str(boost::format("UnaryOp, %s, (%s)") % timewhoop::ToString(op_) % src1_->ToCSV() );
        return ret;
      }

      virtual Maybe<int> FirstOrderEval()
      {
        auto ret = src1_->FirstOrderEval();
        return ret;
      }

      virtual void SetLoopBlockID(int blk_id)
      {
        loop_block_id_ = blk_id;
        src1_->SetLoopBlockID(blk_id);
      }

      virtual void SetNumAccess(int num_access)
      {
        num_access_ = num_access;
        src1_->SetNumAccess(num_access);
      }

      virtual void SetAccessProb(double access_prob)
      {
        access_probability_ *= access_prob;
        src1_->SetAccessProb(access_prob);
      }

      virtual Maybe<std::tuple<int, int>> MinMaxEval(std::shared_ptr<LoopInformationTable> loop_info_table, int loop_block_id)
      {
        Maybe<std::tuple<int, int>> ret;
        auto src1_range = src1_->MinMaxEval(loop_info_table, loop_block_id);

        if(src1_range.IsValid())
        {
          auto src1_range_contents = src1_range.GetValue();
          int src1_min = std::get<0>(src1_range_contents);
          int src1_max = std::get<1>(src1_range_contents);

          switch(op_)
          {
            case(UnaryOperator::PRE_INCREMENT):
            {
              ret.SetValue(std::make_tuple(src1_min+1, src1_max+1));
              break;
            }
            case(UnaryOperator::PRE_DECREMENT):
            {
              ret.SetValue(std::make_tuple(src1_min-1, src1_max-1));
              break;
            }
            case(UnaryOperator::POST_INCREMENT):
            case(UnaryOperator::POST_DECREMENT):
            {
              ret.SetValue(std::make_tuple(src1_min, src1_max));
              break;
            }
            default:
              break;
        } 
      }
        return ret;
    }

      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
        if(src1_->GetClass() == search_class)
        {
          search_res->push_back(src1_);
        }
        src1_->SearchExpressions(search_res, search_class, deep_search);
      }
 
      std::shared_ptr<Expression> getSrc1()
      {
        return src1_;
      }
  
      UnaryOperator getOp()
      {
        return op_;
      }
  }; // End of class UnaryOp
  
  class BinaryOp : public Expression
  {
    protected:
      std::shared_ptr<Expression> src1_;
      std::shared_ptr<Expression> src2_;
      BinaryOperator op_;
  
    public: 
   
      BinaryOp(BinaryOperator op, std::shared_ptr<Expression> s1, std::shared_ptr<Expression> s2) :
        src1_(s1), src2_(s2), op_(op)
      {
      }
  
      virtual ExpressionClass GetClass()
      {
        return ExpressionClass::BINARYOP;
      }

      virtual void GetLeaves(std::shared_ptr<std::list<std::shared_ptr<Expression>>> leaf_list)
      {
        if(src1_->GetClass() == ExpressionClass::VARIABLE)
        {
          leaf_list->push_back(src1_);
        }
        else
        {
          src1_->GetLeaves(leaf_list);
        }

        if(src2_->GetClass() == ExpressionClass::VARIABLE)
        {
          leaf_list->push_back(src2_);
        }
        else
        {
          src2_->GetLeaves(leaf_list);
        }
      }

      virtual std::string GetName()
      {
        std::string ret = "";
        return ret;
      }

      virtual void GetTensorIndexExpressions (std::shared_ptr<Expression> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
        src1_->GetTensorIndexExpressions(targ_tensor, access_expr_list);
        src2_->GetTensorIndexExpressions(targ_tensor, access_expr_list);
      }
  
      virtual std::string ToString()
      {
        std::string ret;
        ret = boost::str(boost::format("BinaryOp op:%s, \n operand1: %s, \n operand2: %s ") % timewhoop::ToString(op_) % src1_->ToString() % src2_->ToString() );
        return ret;
      }

      virtual std::string ToCSV()
      {
        std::string ret = boost::str(boost::format("BinaryOp, %s, (%s), (%s)") % timewhoop::ToString(op_) % src1_->ToCSV() % src2_->ToCSV() );
        return ret;
      }

      virtual Maybe<int> FirstOrderEval()
      {
        auto ret = src2_->FirstOrderEval();
        return ret;
      }

      virtual void SetLoopBlockID(int blk_id)
      {
        loop_block_id_ = blk_id;
        src1_->SetLoopBlockID(blk_id);
        src2_->SetLoopBlockID(blk_id);
      }

      virtual void SetNumAccess(int num_access)
      {
        num_access_ = num_access;
        src1_->SetNumAccess(num_access);
        src2_->SetNumAccess(num_access);
      }

      virtual void SetAccessProb(double access_prob)
      {
        access_probability_ *= access_prob;
        src1_->SetAccessProb(access_prob);
        src2_->SetAccessProb(access_prob);
      }

      virtual Maybe<std::tuple<int, int>> MinMaxEval(std::shared_ptr<LoopInformationTable> loop_info_table, int loop_block_id)
      {
        Maybe<std::tuple<int, int>> ret;
        auto src1_range = src1_->MinMaxEval(loop_info_table, loop_block_id);
        auto src2_range = src2_->MinMaxEval(loop_info_table, loop_block_id);

        if(src1_range.IsValid() && src2_range.IsValid()) //Evaluation is possible only if src1/src2 eval results are available
        {
          auto src1_range_contents = src1_range.GetValue();
          auto src2_range_contents = src2_range.GetValue();

          int src1_min = std::get<0>(src1_range_contents);
          int src1_max = std::get<1>(src1_range_contents);

          int src2_min = std::get<0>(src2_range_contents);
          int src2_max = std::get<1>(src2_range_contents);

          //Caution: Double-bound problem
          // if a is in [0, 3) and b is in [0, 5), the range of a+b is [0+0, 3+5 "-1") 
          switch(op_)
          {
            case(BinaryOperator::PLUS):
            {
              ret.SetValue(std::make_tuple(src1_min + src2_min, src1_max + src2_max-1));
              break;
            }
            case(BinaryOperator::MINUS):
            {
              int resMin1 = src1_min - src2_min;
              int resMin2 = src1_max - src2_max;
              int resMax1 = src1_max - src2_max;
              int resMax2 = src1_min - src2_min;

              ret.SetValue(std::make_tuple(std::min(resMin1, resMin2), std::max(resMax1, resMax2) +1 ));
              break;
            }
            case(BinaryOperator::MULT):
            {
              ret.SetValue(std::make_tuple(src1_min * src2_min, (src1_max-1) * (src2_max-1) +1  ));
              break;
            }
            case(BinaryOperator::DIV):
            {
              //TODO: Apply better error handler 
              assert(src2_min != 0 && src2_max != 0);
              int resMin1 = src1_min / src2_min;
              int resMin2 = src1_max / src2_max;
              int resMax1 = src1_max / src2_max;
              int resMax2 = src1_min / src2_min;

              ret.SetValue(std::make_tuple(std::min(resMin1, resMin2), std::max(resMax1, resMax2)));
              break;
            }
            default:
              break;
          } //End of switch (op_)
        } //End of if (ranges->isValid)

        return ret;
      }


      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
        if(src1_->GetClass() == search_class)
        {
          search_res->push_back(src1_);
        }
        src1_->SearchExpressions(search_res, search_class, deep_search);

        if(src2_->GetClass() == search_class)
        {
          search_res->push_back(src2_);
        }
        src2_->SearchExpressions(search_res, search_class, deep_search);
      }

      std::shared_ptr<Expression> getSrc1()
      {
        return src1_;
      }
  
      std::shared_ptr<Expression> getSrc2()
      {
        return src2_;
      }
  
      BinaryOperator getOp()
      {
        return op_;
      }
  }; // End of class BinaryOp
 
  class VariableDeclaration : public Declaration
  {
    public:
      VariableDeclaration(std::shared_ptr<Variable> target) :
        Declaration(target)
      {
      }

      VariableDeclaration(std::shared_ptr<Variable> target, double exec_prob) :
        Declaration(target, exec_prob)
      {
      }
  
      std::shared_ptr<Expression> GetTarget()
      {
        return target_;
      }

      virtual StatementClass GetClass()
      {
        return StatementClass::VARIABLE_DECLARATION;
      }
  
      virtual std::shared_ptr<Container> GetOwner()
      {
        return target_;
      }

      virtual Maybe<int> FirstOrderEvalBody()
      {
        auto ret = Maybe<int>(false);
        return ret;
      }

      virtual std::string ToString()
      {
         std::string ret = boost::str(boost::format("(Variable Declaration, loop_block_id: %d, num_exec: %d \n target_container: %s)") % loop_block_id_  % num_executions_ % target_->ToString() );
         return ret;
      }

      virtual void ExtractTensorAccessExpressions(std::shared_ptr<Container> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
      }

      virtual void SearchStatements(std::shared_ptr<std::list<std::shared_ptr<Statement>>> search_res, StatementClass search_class, bool deep_search)
      {
        if(next_ != nullptr && next_ != NULL)
        {
          if(next_->GetClass() == search_class)
          {
            search_res->push_back(next_);
          }
          next_->SearchStatements(search_res, search_class, deep_search);
        }
      }

      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
        if(target_->GetClass() == search_class)
        {
          search_res->push_back(target_);
        }

        target_->SearchExpressions(search_res, search_class, deep_search);
        if(next_ != nullptr && next_ != NULL)
        {
          next_->SearchExpressions(search_res, search_class, deep_search);
        }
      }

      virtual void ExtractLoopInfo(std::shared_ptr<LoopInformationTable> loop_info_table)
      {
      }

      virtual void EvaluateNumExecs(int mult)
      {
        num_executions_ *= mult;
        target_->SetNumAccess(num_executions_);
      }

      virtual int AnalyzeLoopBlockID(int current_id)
      {
        target_->SetLoopBlockID(current_id);
        loop_block_id_ = current_id;
        return current_id;
      }

      virtual void SetExecProb(double exec_prob)
      {
        execution_probability_ *= exec_prob;
      }

  }; // End of class VariableDeclaration
 
  class TensorDeclaration : public Declaration
  {
    public:
      TensorDeclaration(std::shared_ptr<Tensor> target) :
        Declaration(target)
      {
      }

      TensorDeclaration(std::shared_ptr<Tensor> target, double exec_prob) :
        Declaration(target, exec_prob)
      {
      }

      std::shared_ptr<Expression> GetTarget()
      {
        return target_;
      }

      virtual StatementClass GetClass()
      {
        return StatementClass::TENSOR_DECLARATION;
      }
 
      virtual std::shared_ptr<Container> GetOwner()
      {
        return target_;
      }
 
      virtual Maybe<int> FirstOrderEvalBody()
      {
        auto ret = Maybe<int>(false);
        return ret;
      }

      virtual std::string ToString()
      {
         std::string ret = boost::str(boost::format("(Tensor Declaration, loop_block_id: %d, num_exec %d, \n target container: %s )") % loop_block_id_ % num_executions_ % target_->ToString() );
         return ret;
      }

      virtual void ExtractTensorAccessExpressions(std::shared_ptr<Container> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
      }

      virtual void SearchStatements(std::shared_ptr<std::list<std::shared_ptr<Statement>>> search_res, StatementClass search_class, bool deep_search)
      {
        if(next_ != nullptr && next_ != NULL)
        {
          if(next_->GetClass() == search_class)
          {
            search_res->push_back(next_);
          }
          next_->SearchStatements(search_res, search_class, deep_search);
        }
      }

      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
        if(target_->GetClass() == search_class)
        {
          search_res->push_back(target_);
        }
        target_->SearchExpressions(search_res, search_class, deep_search);

        if(next_ != nullptr && next_ != NULL)
        {
          next_->SearchExpressions(search_res, search_class, deep_search);
        }
      }

      virtual void ExtractLoopInfo(std::shared_ptr<LoopInformationTable> loop_info_table)
      {
      }

      virtual void EvaluateNumExecs(int mult)
      {
        num_executions_ *= mult;
        target_->SetNumAccess(num_executions_);
      }

      virtual int AnalyzeLoopBlockID(int current_id)
      {
        target_->SetLoopBlockID(current_id);
        loop_block_id_ = current_id;
        return current_id;
      }

      virtual void SetExecProb(double exec_prob)
      {
        execution_probability_ *= exec_prob;
      }

  }; // End of class TensorDeclaration

  class TemporalFor : public PrimFor
  {
    protected:
      std::shared_ptr<std::list<std::string>> buffered_tensors_;

    public:
      TemporalFor(std::shared_ptr<Statement> init_s, std::shared_ptr<Expression> test_e, std::shared_ptr<Statement> incr_s, std::shared_ptr<Statement> body_s) :
        PrimFor(init_s, test_e, incr_s, body_s)
      {
      }

      TemporalFor(std::shared_ptr<Statement> init_s, std::shared_ptr<Expression> test_e, std::shared_ptr<Statement> incr_s, std::shared_ptr<Statement> body_s, std::shared_ptr<std::list<std::string>> buffered_tensors) :
        buffered_tensors_(buffered_tensors),
        PrimFor(init_s, test_e, incr_s, body_s)
      {
      }

      TemporalFor(std::shared_ptr<Statement> init_s, std::shared_ptr<Expression> test_e, std::shared_ptr<Statement> incr_s, std::shared_ptr<Statement> body_s, double exec_prob) :
        PrimFor(init_s, test_e, incr_s, body_s, exec_prob)
      {
      }
    
      virtual StatementClass GetClass()
      {
        return StatementClass::TEMPORAL_FOR;
      }

      virtual std::shared_ptr<Container> GetOwner()
      {
        return init_stmt_->GetOwner();
      }

      virtual Maybe<int> FirstOrderEvalBody()
      {
        auto ret = Maybe<int>(false);
        return ret;
      }

      virtual std::string ToString()
      {
        std::string ret = boost::str(boost::format("(Temporal For, loop_block_id: %d, loop_block_id_base: %d, loop_block_id_bound: %d, num_exec: %d, \n init_statement: %s\n, \n test_expression: %s\n, \n incr_statement: %s\n, \n body_statement: %s\n)") % loop_block_id_ % loop_block_id_base_ % loop_block_id_bound_ % num_executions_ % init_stmt_->ToString() % test_expr_->ToString() % incr_stmt_->ToString() % body_stmt_->ToString() );

        std::shared_ptr<Statement> curr_body = body_stmt_->GetNextStmt();
        while(curr_body != nullptr)
        {
          ret += curr_body->ToString();
          curr_body = curr_body->GetNextStmt();
        }
        return ret; 
      }

      virtual void ExtractTensorAccessExpressions(std::shared_ptr<Container> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
        test_expr_->GetTensorIndexExpressions(targ_tensor, access_expr_list);
        init_stmt_->ExtractTensorAccessExpressions(targ_tensor, access_expr_list);
        incr_stmt_->ExtractTensorAccessExpressions(targ_tensor, access_expr_list);
        body_stmt_->ExtractTensorAccessExpressions(targ_tensor, access_expr_list);

        std::shared_ptr<Statement> curr_body = body_stmt_->GetNextStmt();
        while(curr_body != nullptr)
        {
          curr_body->ExtractTensorAccessExpressions(targ_tensor, access_expr_list);
          curr_body = curr_body->GetNextStmt();
        }
      }

      virtual void SearchStatements(std::shared_ptr<std::list<std::shared_ptr<Statement>>> search_res, StatementClass search_class, bool deep_search)
      {
        if(deep_search)
        {
          init_stmt_->SearchStatements(search_res, search_class, deep_search);
          incr_stmt_->SearchStatements(search_res, search_class, deep_search);
        }

        body_stmt_->SearchStatements(search_res, search_class, deep_search);

        if(next_ != nullptr && next_ != NULL)
        {
          if(next_->GetClass() == search_class)
          {
            search_res->push_back(next_);
          }
          next_->SearchStatements(search_res, search_class, deep_search);
        }
      }

      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
        if(deep_search)
        {
          if(test_expr_->GetClass() == search_class)
          {
            search_res->push_back(test_expr_);
          }

          init_stmt_->SearchExpressions(search_res, search_class, deep_search);
          test_expr_->SearchExpressions(search_res, search_class, deep_search);
          incr_stmt_->SearchExpressions(search_res, search_class, deep_search);
        }

        body_stmt_->SearchExpressions(search_res, search_class, deep_search);

        if(next_ != nullptr && next_ != NULL)
        {
          next_->SearchExpressions(search_res, search_class, deep_search);
        }
      }

      virtual void ExtractLoopInfo(std::shared_ptr<LoopInformationTable> loop_info_table)
      {
        auto base_query = init_stmt_->FirstOrderEvalBody();
        auto bound_query = test_expr_->FirstOrderEval();

        //TODO: Add error handling in case of invalid values
        int base = (base_query.IsValid())? base_query.GetValue() : 0;
        int bound = (bound_query.IsValid())? bound_query.GetValue() : 0;

        auto loop_var = init_stmt_->GetOwner();

        auto new_loop_info = std::make_shared<LoopInformation>(this, loop_var, loop_block_id_base_, loop_block_id_bound_, base, bound);

        loop_info_table->AddLoopInfo(new_loop_info);

        auto stmt = body_stmt_;
        while(stmt != nullptr && stmt != NULL)
        {
          stmt->ExtractLoopInfo(loop_info_table);
          stmt = stmt->GetNextStmt();
        } 
      }

      virtual void EvaluateNumExecs(int mult = 1)
      {
        num_executions_ *= mult;
        init_stmt_->EvaluateNumExecs(mult);
        incr_stmt_->EvaluateNumExecs(mult);
        test_expr_->SetNumAccess(num_executions_);
        int body_mult = mult * GetNumIterations(init_stmt_, test_expr_);
        auto stmt = body_stmt_;
        while(stmt != nullptr && stmt != NULL)
        {
          stmt->EvaluateNumExecs(body_mult);
          stmt = stmt->GetNextStmt();
        } 
      }

      virtual int AnalyzeLoopBlockID(int current_id)
      {
        loop_block_id_ = current_id;
        loop_block_id_base_ = current_id;
        test_expr_->SetLoopBlockID(current_id);
        int last_id = current_id;
        init_stmt_->AnalyzeLoopBlockID(last_id);
        last_id = incr_stmt_->AnalyzeLoopBlockID(last_id);
        last_id = body_stmt_->AnalyzeLoopBlockID(last_id);
        
        auto stmt = body_stmt_->GetNextStmt();
        while(stmt != nullptr && stmt != NULL)
        {
          if(stmt->GetClass() != StatementClass::STATEMENT) //EndLoop
          {
            last_id++;
          }
          last_id = stmt->AnalyzeLoopBlockID(last_id);
          stmt = stmt->GetNextStmt();
        }
        loop_block_id_bound_ = last_id;
        return last_id;
      }

      virtual void SetExecProb(double exec_prob)
      {
        execution_probability_ *= exec_prob;
        init_stmt_->SetExecProb(exec_prob);
        test_expr_->SetAccessProb(exec_prob);
        incr_stmt_->SetExecProb(exec_prob);

        std::shared_ptr<Statement> stmt = body_stmt_;
        while(stmt != nullptr && stmt != NULL)
        {
          stmt->SetExecProb(exec_prob);
          stmt = stmt->GetNextStmt();
        }
      }


  }; // End of class TemporalFor
  
  
  class SpatialFor : public PrimFor
  {
    protected:
      int num_partitions_;
   
    public:
  
      SpatialFor(const int& num_parts, std::shared_ptr<Statement> init_s, std::shared_ptr<Expression> test_e, std::shared_ptr<Statement> incr_s, std::shared_ptr<Statement> body_s) :
        num_partitions_(num_parts),
        PrimFor(init_s, test_e, incr_s, body_s)
      {
      }

      SpatialFor(const int& num_parts, std::shared_ptr<Statement> init_s, std::shared_ptr<Expression> test_e, std::shared_ptr<Statement> incr_s, std::shared_ptr<Statement> body_s, double exec_prob) :
        num_partitions_(num_parts),
        PrimFor(init_s, test_e, incr_s, body_s, exec_prob)
      {
      }
    
      virtual StatementClass GetClass()
      {
        return StatementClass::SPATIAL_FOR;
      }

      virtual std::shared_ptr<Container> GetOwner()
      {
        return init_stmt_->GetOwner();
      }

      virtual Maybe<int> FirstOrderEvalBody()
      {
        auto ret = Maybe<int>(false);
        return ret;
      }

      virtual std::string ToString()
      {

        std::string ret = boost::str(boost::format("(Spatial For, loop_block_id: %d, loop_block_id_base: %d, loop_block_id_bound: %d, num_exec: %d, \n init_statement: %s\n, \n test_expression: %s\n, \n incr_statement: %s\n, \n body_statement: %s\n)") % loop_block_id_ % loop_block_id_base_ % loop_block_id_bound_ % num_executions_ % init_stmt_->ToString() % test_expr_->ToString() % incr_stmt_->ToString() % body_stmt_->ToString() );
        return ret; 
      }

      virtual void ExtractTensorAccessExpressions(std::shared_ptr<Container> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
        test_expr_->GetTensorIndexExpressions(targ_tensor, access_expr_list);
        init_stmt_->ExtractTensorAccessExpressions(targ_tensor, access_expr_list);
        incr_stmt_->ExtractTensorAccessExpressions(targ_tensor, access_expr_list);
        body_stmt_->ExtractTensorAccessExpressions(targ_tensor, access_expr_list);
        std::shared_ptr<Statement> curr_body = body_stmt_->GetNextStmt();
        while(curr_body != nullptr)
        {
          curr_body->ExtractTensorAccessExpressions(targ_tensor, access_expr_list);
          curr_body = curr_body->GetNextStmt();
        }
      }

      virtual void SearchStatements(std::shared_ptr<std::list<std::shared_ptr<Statement>>> search_res, StatementClass search_class, bool deep_search)
      {
        if(deep_search)
        {
          init_stmt_->SearchStatements(search_res, search_class, deep_search);
          incr_stmt_->SearchStatements(search_res, search_class, deep_search);
        }

        body_stmt_->SearchStatements(search_res, search_class, deep_search);

        if(next_ != nullptr && next_ != NULL)
        {
          if(next_->GetClass() == search_class)
          {
            search_res->push_back(next_);
          }
          next_->SearchStatements(search_res, search_class, deep_search);
        }
      }

      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
        if(deep_search)
        {
          if(test_expr_->GetClass() == search_class)
          {
            search_res->push_back(test_expr_);
          }

          init_stmt_->SearchExpressions(search_res, search_class, deep_search);
          test_expr_->SearchExpressions(search_res, search_class, deep_search);
          incr_stmt_->SearchExpressions(search_res, search_class, deep_search);
        }

        body_stmt_->SearchExpressions(search_res, search_class, deep_search);

        if(next_ != nullptr && next_ != NULL)
        {
          next_->SearchExpressions(search_res, search_class, deep_search);
        }
      }

      virtual void ExtractLoopInfo(std::shared_ptr<LoopInformationTable> loop_info_table)
      {
        auto base_query = init_stmt_->FirstOrderEvalBody();
        auto bound_query = test_expr_->FirstOrderEval();

        //TODO: Add error handling in case of invalid values
        int base = (base_query.IsValid())? base_query.GetValue() : 0;
        int bound = (bound_query.IsValid())? bound_query.GetValue() : 0;

        auto loop_var = init_stmt_->GetOwner();

        auto new_loop_info = std::make_shared<LoopInformation>(this, loop_var, loop_block_id_base_, loop_block_id_bound_, base, bound);

        loop_info_table->AddLoopInfo(new_loop_info);

        auto stmt = body_stmt_;
        while(stmt != nullptr && stmt != NULL)
        {
          stmt->ExtractLoopInfo(loop_info_table);
          stmt = stmt->GetNextStmt();
        } 
      }

      virtual void EvaluateNumExecs(int mult)
      {
        num_executions_ *= mult;
        init_stmt_->EvaluateNumExecs(mult);
        incr_stmt_->EvaluateNumExecs(mult);
        test_expr_->SetNumAccess(num_executions_);
        int body_mult = mult * GetNumIterations(init_stmt_, test_expr_);
        auto stmt = body_stmt_;
        while(stmt != nullptr && stmt != NULL)
        {
          stmt->EvaluateNumExecs(body_mult);
          stmt = stmt->GetNextStmt();
        } 
      }

      virtual int AnalyzeLoopBlockID(int current_id)
      {
        loop_block_id_ = current_id;
        loop_block_id_base_ = current_id;
        test_expr_->SetLoopBlockID(current_id);
        int last_id = current_id;
        init_stmt_->AnalyzeLoopBlockID(last_id);
        last_id = body_stmt_->AnalyzeLoopBlockID(last_id);
        
        auto stmt = body_stmt_->GetNextStmt();
        while(stmt != nullptr && stmt != NULL)
        {
          if(stmt->GetClass() != StatementClass::STATEMENT) //EndLoop
          {
            last_id++;
          }
          last_id = stmt->AnalyzeLoopBlockID(last_id);
          stmt = stmt->GetNextStmt();
        }
        loop_block_id_bound_ = last_id;
        return last_id;
      }

      virtual void SetExecProb(double exec_prob)
      {
        execution_probability_ *= exec_prob;
        init_stmt_->SetExecProb(exec_prob);
        test_expr_->SetAccessProb(exec_prob);
        incr_stmt_->SetExecProb(exec_prob);

        std::shared_ptr<Statement> stmt = body_stmt_;
        while(stmt != nullptr && stmt != NULL)
        {
          stmt->SetExecProb(exec_prob);
          stmt = stmt->GetNextStmt();
        }
      }

  }; //End of class SpatialFor
    
  class VariableAssignment : public PrimAssignment
  {
    public:
      VariableAssignment(std::shared_ptr<Container> target, std::shared_ptr<Expression> body) :
        PrimAssignment(target, body)
      {
      }

      VariableAssignment(std::shared_ptr<Container> target, std::shared_ptr<Expression> body, double exec_prob) :
        PrimAssignment(target, body, exec_prob)
      {
      }
  
      virtual StatementClass GetClass()
      {
        return StatementClass::VAR_ASSIGNMENT;
      }
 
      virtual std::shared_ptr<Container> GetOwner()
      {
        return target_;
      }
 
      virtual Maybe<int> FirstOrderEvalBody()
      {
        auto ret = body_expr_->FirstOrderEval();
        return ret;
      }

      virtual std::string ToString()
      {
        std::string ret = boost::str(boost::format("(Variable Assignment, loop_block_id: %d, num_exec: %d, \n target_container: %s, \n assignment_expression: %s") % loop_block_id_ % num_executions_ % target_->ToString() % body_expr_->ToString() );
        return ret; 
      }

      virtual void ExtractTensorAccessExpressions(std::shared_ptr<Container> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
        body_expr_->GetTensorIndexExpressions(targ_tensor, access_expr_list);
      }
 
      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
        if(target_->GetClass() == search_class)
        {
          search_res->push_back(target_);
        }
        if(body_expr_->GetClass() == search_class)
        {
          search_res->push_back(body_expr_);
        }

        target_->SearchExpressions(search_res, search_class, deep_search);
        body_expr_->SearchExpressions(search_res, search_class, deep_search);

        if(next_ != nullptr && next_ != NULL)
        {
          next_->SearchExpressions(search_res, search_class, deep_search);
        }
      }

      virtual void ExtractLoopInfo(std::shared_ptr<LoopInformationTable> loop_info_table)
      {
      }

      virtual void EvaluateNumExecs(int mult)
      {
        num_executions_ *= mult;
        target_->SetNumAccess(num_executions_);
        body_expr_->SetNumAccess(num_executions_);
      }

      virtual int AnalyzeLoopBlockID(int current_id)
      {
        target_->SetLoopBlockID(current_id);
        body_expr_->SetLoopBlockID(current_id);
        loop_block_id_ = current_id;
        return current_id;
      }

      virtual void SetExecProb(double exec_prob)
      {
        execution_probability_ *= exec_prob;
        target_->SetAccessProb(exec_prob);
        body_expr_->SetAccessProb(exec_prob);
      }

  }; // End of class VariableAssignment
  
  class TensorAssignment : public PrimAssignment
  {
    protected:
      std::shared_ptr<std::list<std::shared_ptr<Expression>>> idx_exprs_;
    public:
   
      TensorAssignment(std::shared_ptr<Container> target, std::shared_ptr<Expression> body, std::shared_ptr<std::list<std::shared_ptr<Expression>>> idx_exprs) :
        PrimAssignment(target, body),
        idx_exprs_(idx_exprs)
      {
      }

      TensorAssignment(std::shared_ptr<Container> target, std::shared_ptr<Expression> body, std::shared_ptr<std::list<std::shared_ptr<Expression>>> idx_exprs, double exec_prob) :
        PrimAssignment(target, body, exec_prob),
        idx_exprs_(idx_exprs)
      {
      }

  
      virtual StatementClass GetClass()
      {
        return StatementClass::TENSOR_ASSIGNMENT;
      }
 
      virtual std::shared_ptr<Container> GetOwner()
      {
        return target_;
      }
 
      virtual Maybe<int> FirstOrderEvalBody()
      {
        auto ret = body_expr_->FirstOrderEval();
        return ret;
      }

      virtual std::string ToString()
      {
        std::string idx_expr_string = "\n---";
        for(auto& idx_e : *idx_exprs_)
        {
          idx_expr_string += idx_e->ToString();
        }
        idx_expr_string += "---";

        std::string ret = boost::str(boost::format("(Tensor Assignment, loop_block_id: %d, num_exec:%d, \n target_container: %s, \n index_expressions: %s, \n assignment_expression: %s)") % loop_block_id_ % num_executions_ % target_->ToString() % idx_expr_string % body_expr_->ToString() );
        return ret; 
      }

      virtual void ExtractTensorAccessExpressions(std::shared_ptr<Container> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
        if(targ_tensor->GetName() == target_->GetName())
        {
          access_expr_list->push_back(idx_exprs_);
        }

        target_->GetTensorIndexExpressions(targ_tensor, access_expr_list);
        body_expr_->GetTensorIndexExpressions(targ_tensor, access_expr_list);
      }

      virtual void SearchStatements(std::shared_ptr<std::list<std::shared_ptr<Statement>>> search_res, StatementClass search_class, bool deep_search)
      {
        if(next_ != nullptr && next_ != NULL)
        {
          if(next_->GetClass() == search_class)
          {
            search_res->push_back(next_);
          }
          next_->SearchStatements(search_res, search_class, deep_search);
        }
      }

      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
        if(target_->GetClass() == search_class)
        {
          search_res->push_back(target_);
        }
        if(body_expr_->GetClass() == search_class)
        {
          search_res->push_back(body_expr_);
        }

        target_->SearchExpressions(search_res, search_class, deep_search);
        body_expr_->SearchExpressions(search_res, search_class, deep_search);

        if(next_ != nullptr && next_ != NULL)
        {
          next_->SearchExpressions(search_res, search_class, deep_search);
        }
      }

      virtual void ExtractLoopInfo(std::shared_ptr<LoopInformationTable> loop_info_table)
      {
      }

      virtual void EvaluateNumExecs(int mult)
      {
        num_executions_ *= mult;
        target_->SetNumAccess(num_executions_);
        body_expr_->SetNumAccess(num_executions_);
      }

      virtual int AnalyzeLoopBlockID(int current_id)
      {
        target_->SetLoopBlockID(current_id);
        body_expr_->SetLoopBlockID(current_id);
        loop_block_id_ = current_id;
        return current_id;
      }

      virtual void SetExecProb(double exec_prob)
      {
        execution_probability_ *= exec_prob;
        target_->SetAccessProb(exec_prob);
        body_expr_->SetAccessProb(exec_prob);
      }

      std::shared_ptr<std::list<std::shared_ptr<Expression>>>& getIdxExpr()
      {
        return idx_exprs_;
      }
  }; // End of class TensorAssignment
  
  
  class If : public CondStatement // class CondStatement : public Statement
  {
    public:
      If(std::shared_ptr<Expression> test_e, std::shared_ptr<Statement> body_s) :
        CondStatement(test_e, body_s)
      {
      }

      If(std::shared_ptr<Expression> test_e, std::shared_ptr<Statement> body_s, double exec_prob) :
        CondStatement(test_e, body_s, exec_prob)
      {
      }
    
      virtual StatementClass GetClass()
      {
        return StatementClass::IF;
      }
 
      virtual std::shared_ptr<Container> GetOwner()
      {
        return nullptr;
      }
 
      virtual Maybe<int> FirstOrderEvalBody()
      {
        auto ret = Maybe<int>(false);
        return ret;
      }

      virtual std::string ToString()
      {
        std::string ret = boost::str(boost::format("(If, loop_block_id: %d, num_exec: %d, \n test_expression: %s, \n body_statment: %s)") % loop_block_id_ % num_executions_ % test_expr_->ToString() % body_stmt_->ToString());
        return ret;
      }

      virtual void ExtractTensorAccessExpressions(std::shared_ptr<Container> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
     {
       test_expr_->GetTensorIndexExpressions(targ_tensor, access_expr_list);
       body_stmt_->ExtractTensorAccessExpressions(targ_tensor, access_expr_list);
       std::shared_ptr<Statement> curr_body = body_stmt_->GetNextStmt();
       while(curr_body != nullptr)
       {
         curr_body->ExtractTensorAccessExpressions(targ_tensor, access_expr_list);
         curr_body = curr_body->GetNextStmt();
       }
     }

      virtual void SearchStatements(std::shared_ptr<std::list<std::shared_ptr<Statement>>> search_res, StatementClass search_class, bool deep_search)
      {
        body_stmt_->SearchStatements(search_res, search_class, deep_search);
        if(next_ != nullptr && next_ != NULL)
        {
          if(next_->GetClass() == search_class)
          {
            search_res->push_back(next_);
          }
          next_->SearchStatements(search_res, search_class, deep_search);
        }
      }

      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
        if(deep_search)
        {
          if(test_expr_->GetClass() == search_class)
          {
            search_res->push_back(test_expr_);
          }

          test_expr_->SearchExpressions(search_res, search_class, deep_search);
        }

        body_stmt_->SearchExpressions(search_res, search_class, deep_search);

        if(next_ != nullptr && next_ != NULL)
        {
          next_->SearchExpressions(search_res, search_class, deep_search);
        }
      }

      virtual void ExtractLoopInfo(std::shared_ptr<LoopInformationTable> loop_info_table)
      {
        auto stmt = body_stmt_;
        while(stmt != nullptr && stmt != NULL)
        {
          stmt->ExtractLoopInfo(loop_info_table);
          stmt = stmt->GetNextStmt();
        } 
      }

      virtual void EvaluateNumExecs(int mult)
      {
        num_executions_ *= mult;
        test_expr_->SetNumAccess(num_executions_);

        std::shared_ptr<Statement> stmt = body_stmt_;
        while(stmt != nullptr && stmt != NULL)
        {
          stmt->EvaluateNumExecs(mult);
          stmt = stmt->GetNextStmt();
        }
      }
 
      virtual int AnalyzeLoopBlockID(int current_id)
      {
        loop_block_id_ = current_id;
        int last_id = body_stmt_->AnalyzeLoopBlockID(current_id);
        return last_id;
      }

      virtual void SetExecProb(double exec_prob)
      {
        execution_probability_ *= exec_prob;
        test_expr_->SetAccessProb(exec_prob);

        std::shared_ptr<Statement> stmt = body_stmt_;
        while(stmt != nullptr && stmt != NULL)
        {
          stmt->SetExecProb(execution_probability_);
          stmt = stmt->GetNextStmt();
        }
      }
  }; // End of class If
  
  class Else : public CondStatement // class CondStatement : public Statement
  {
    public:
      Else(std::shared_ptr<Statement> body_s) :
        CondStatement (nullptr, body_s)
      {
      }

      Else(std::shared_ptr<Statement> body_s, double exec_prob) :
        CondStatement (nullptr, body_s, exec_prob)
      {
      }

  
      virtual StatementClass GetClass()
      {
        return StatementClass::ELSE;
      }

      virtual std::shared_ptr<Container> GetOwner()
      {
        return nullptr;
      }
 
      virtual Maybe<int> FirstOrderEvalBody()
      {
        auto ret = Maybe<int>(false);
        return ret;
      }

      virtual std::string ToString()
      {
        std::string ret = boost::str(boost::format("(Else, loop_block_id: %d, num_exec: %d, \n body_statement: %s)") % loop_block_id_ % num_executions_ % body_stmt_->ToString());
        return ret;
      }

      virtual void ExtractTensorAccessExpressions(std::shared_ptr<Container> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
     {
       body_stmt_->ExtractTensorAccessExpressions(targ_tensor, access_expr_list);
       std::shared_ptr<Statement> curr_body = body_stmt_->GetNextStmt();
       while(curr_body != nullptr)
       {
         curr_body->ExtractTensorAccessExpressions(targ_tensor, access_expr_list);
         curr_body = curr_body->GetNextStmt();
       }
     }

      virtual void SearchStatements(std::shared_ptr<std::list<std::shared_ptr<Statement>>> search_res, StatementClass search_class, bool deep_search)
      {
        body_stmt_->SearchStatements(search_res, search_class, deep_search);
        if(next_ != nullptr && next_ != NULL)
        {
          if(next_->GetClass() == search_class)
          {
            search_res->push_back(next_);
          }
          next_->SearchStatements(search_res, search_class, deep_search);
        }
      }

      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
        if(deep_search)
        {
          if(test_expr_ != nullptr)
          {
            if(test_expr_->GetClass() == search_class)
            {
              search_res->push_back(test_expr_);
            }

            test_expr_->SearchExpressions(search_res, search_class, deep_search);
          }
        }

        body_stmt_->SearchExpressions(search_res, search_class, deep_search);

        if(next_ != nullptr && next_ != NULL)
        {
          next_->SearchExpressions(search_res, search_class, deep_search);
        }
      }

      virtual void ExtractLoopInfo(std::shared_ptr<LoopInformationTable> loop_info_table)
      {
        auto stmt = body_stmt_;
        while(stmt != nullptr && stmt != NULL)
        {
          stmt->ExtractLoopInfo(loop_info_table);
          stmt = stmt->GetNextStmt();
        } 
      }

      virtual void EvaluateNumExecs(int mult)
      {
        num_executions_ *= mult;
        if(test_expr_ != nullptr && test_expr_ != NULL)
        {
          test_expr_->SetNumAccess(num_executions_);
        }

        std::shared_ptr<Statement> stmt = body_stmt_;
        while(stmt != nullptr && stmt != NULL)
        {
          stmt->EvaluateNumExecs(mult);
          stmt = stmt->GetNextStmt();
        }
      }

      virtual int AnalyzeLoopBlockID(int current_id)
      {
        loop_block_id_ = current_id;
        int last_id = current_id;

        std::shared_ptr<Statement> stmt = body_stmt_;
        while(stmt != nullptr && stmt != NULL)
        {
          last_id = body_stmt_->AnalyzeLoopBlockID(current_id);
        }
        return last_id;
      }

      virtual void SetExecProb(double exec_prob)
      {
        execution_probability_ *= exec_prob;
        if(test_expr_ != nullptr && test_expr_ != NULL)
        {
          test_expr_->SetAccessProb(exec_prob);
        }
        std::shared_ptr<Statement> stmt = body_stmt_;
        while(stmt != nullptr && stmt != NULL)
        {
          stmt->SetExecProb(execution_probability_);
          stmt = stmt->GetNextStmt();
        }
      }

  }; //End of class Else

};  // namespace timewhoop
#endif /* TIMEWHOOP_PURE_ABSTRACT_SYNTAX_HPP_ */
