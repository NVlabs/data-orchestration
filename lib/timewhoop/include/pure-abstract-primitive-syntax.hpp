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

#ifndef TIMEWHOOP_PURE_ABSTRACT_PRIMITIVE_SYNTAX_HPP_
#define TIMEWHOOP_PURE_ABSTRACT_PRIMITIVE_SYNTAX_HPP_

#include <iostream>
#include <list>
#include <vector>
#include <map>
#include <stack>
#include <tuple>
#include <memory>

#include <boost/format.hpp>

#include "maybe-type.hpp"
#include "pure-abstract-syntax-types.hpp"

//#include "analysis-structure.hpp"


/**** Primitive pure abstract syntax ****

- Description: This file includes base classes for timewhoop's pure AST
- Defined classes (Indentation represents inheritance)

  Expression
    Container
  
  Statement
    Declaration
    PrimAssignment
    PrimFor
    ConditionalStatement

- Printout APIs for both of Expression and Statement
  - Printout formats are different in each derived classes.
  - If the class has substructure, it prints out all of them.
  - Printout APIs:
    1) virtual std::string ToString()
      -> Prints out the information(Name, internal data value, and so on) of each class in human-readable format.

    2) virtual std::string ToCSV()
      -> Prints out the information of each class in CSV format for futher processing
*/

namespace timewhoop
{

  /* Forward declarations from analysis-structure.hpp */
  class ExpressionInformation;
  class StatementInformation;
  class LoopInformation;
  class LoopInformationTable;

  /* The first base class for timewhoop's pure-AST  */
  class Expression
  {
    protected:
      int num_access_ = 0;
      double access_probability_ = 1.0;
      int loop_block_id_ = -1;
    public:
      Expression () = default;

      Expression (int loop_blk_id) :
        loop_block_id_(loop_blk_id)
      {
      }
  
      virtual ExpressionClass GetClass() 
      {
        return ExpressionClass::EXPRESSION;
      }

      virtual void GetTensorIndexExpressions (std::shared_ptr<Expression> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
      }

      virtual void GetLeaves(std::shared_ptr<std::list<std::shared_ptr<Expression>>> leaf_list)
      {
      }
 
      virtual std::string GetName()
      {
        std::string ret = "";
        return ret;
      }
 
      virtual std::string ToString()
      {
         std::string ret = "";
         return ret;
      }

      virtual std::string ToCSV()
      {
        std::string ret = "";
        return ret;
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

      int GetLoopBlockID()
      {
        return loop_block_id_;
      }

      virtual void SetNumAccess(int num_access)
      {
        num_access_ = num_access;
      }

      int GetNumAccess()
      {
        return num_access_;
      }

      virtual void SetAccessProb(double access_prob)
      {
        access_probability_ *= access_prob;
      }

      double GetAccessProb()
      {
        return access_probability_;
      }

      virtual Maybe<std::tuple<int, int>> MinMaxEval(std::shared_ptr<LoopInformationTable> loop_info_table, int loop_block_id)
      {
        Maybe<std::tuple<int, int>> ret; //Invalid in default
        return ret;
      }

      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
      }
  }; //End of class Expression
  
  class Container : public Expression
  {
    protected:
      Type type_;
      std::string name_;
  
    public:
      Container(Type tp, const std::string& nm) : 
        type_(tp),
        name_(nm)
      {
      }

      Container(Type tp, const std::string& nm, int blk_id) :
        Expression(blk_id),
        type_(tp),
        name_(nm) 
      {
      }
  
      virtual ExpressionClass GetClass()
      {
        return ExpressionClass::CONTAINER;
      }
  
      virtual void GetLeaves(std::shared_ptr<std::list<std::shared_ptr<Expression>>> leaf_list)
      {
      }

      virtual std::string GetName()
      {
        return name_;
      }

      virtual std::string ToString()
      {
         std::string ret = "";
         return ret;
      }

      virtual std::string ToCSV()
      {
        std::string ret = boost::str(boost::format("Container") );
        return ret;
      }

      virtual void GetTensorIndexExpressions (std::shared_ptr<Expression> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
      }

      virtual int GetNumDims()
      {
        return -1;
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
        Maybe<std::tuple<int, int>> ret; //Invalid in default
        return ret;
      }

      virtual void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search)
      {
      }
  }; // End of class Container


  /* The second base class for timewhoop's pure-AST  */
  class Statement
  {
    protected:
      std::shared_ptr<Statement> next_ = nullptr;
      double execution_probability_ = 1.0;
      int num_executions_ = 1;
      int loop_block_id_ = 0;
      int loop_block_id_base_ = 0;
      int loop_block_id_bound_ = 0;

    public:
      Statement() = default;

      Statement(double exe_prob)
      {
        if(exe_prob > 1.0 || exe_prob < 0.0)
        {
          execution_probability_ = 1.0;
        }

        else 
        {
          execution_probability_ = exe_prob;
        }
      }

      Statement(int num_exec) :
        num_executions_(num_exec)
      {
      }

      Statement(double exe_prob, int num_exec) :
        num_executions_(num_exec)
      {
        if(exe_prob > 1.0 || exe_prob < 0.0)
        {
          execution_probability_ = 1.0;
        }

        else 
        {
          execution_probability_ = exe_prob;
        }

      }
  
      virtual StatementClass GetClass()
      {
        return StatementClass::STATEMENT;
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
         // Whoop utilizes pure Statement for EndLoop
         std::string ret = "EndLoop\n";
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
      }

      virtual int AnalyzeLoopBlockID(int current_id)
      {
        loop_block_id_ = current_id;
        return current_id;
      }

      void SetNextStmt(std::shared_ptr<Statement> next)
      {
        next_ = next;
      }
  
      std::shared_ptr<Statement> GetNextStmt()
      {
        return next_;
      }

      virtual void SetExecProb(double exec_prob)
      {
        execution_probability_ *= exec_prob;
      }

      double  GetExecProb()
      {
        return execution_probability_;
      }

      void SetBlockID(int blk_id)
      {
        loop_block_id_ = blk_id;
      }

      int GetBlockID()
      {
        return loop_block_id_;
      }

      int GetBlockIDBase()
      {
        return loop_block_id_base_;
      }

      int GetBlockIDBound()
      {
        return loop_block_id_bound_;
      }

      void SetNumExec(int num_exec)
      {
        num_executions_ = num_exec;
      }

      int GetNumExec()
      {
        return num_executions_;
      }
  }; // End of class Statement

  int GetNumIterations(std::shared_ptr<Statement> init_s, std::shared_ptr<Expression> test_e);

  class Declaration : public Statement
  {
    protected:
      std::shared_ptr<Container> target_;
  
    public:
      Declaration(std::shared_ptr<Container> target) :
        Statement(1),
        target_(target)
      {
      }

      Declaration(std::shared_ptr<Container> target, double exec_prob) :
        Statement(exec_prob, 1),
        target_(target)
      {
      }
  
      std::shared_ptr<Expression> GetTarget()
      {
        return target_;
      }

      virtual StatementClass GetClass()
      {
        return StatementClass::DECLARATION;
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
         std::string ret = boost::str(boost::format("(Declaration %s, loop_block_id = %d, num_exec = %s)") % target_->ToString() % loop_block_id_  % num_executions_ );
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
      }

      virtual int AnalyzeLoopBlockID(int current_id)
      {
        loop_block_id_ = current_id;
        return current_id;
      }

  }; //End of class Declaration

  class PrimAssignment : public Statement
  {
    protected:
      std::shared_ptr<Container> target_;
      std::shared_ptr<Expression> body_expr_;
  
    public:
  
      PrimAssignment(std::shared_ptr<Container> target, std::shared_ptr<Expression> body) :
        target_(target),
        body_expr_(body)
      {
      }

      PrimAssignment(std::shared_ptr<Container> target, std::shared_ptr<Expression> body, double exec_prob) :
        Statement(exec_prob),
        target_(target), 
        body_expr_(body)
      {
      }
  
      virtual StatementClass GetClass()
      {
        return StatementClass::PRIM_ASSIGNMENT;
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
      }

      virtual void ExtractTensorAccessExpressions(std::shared_ptr<Container> targ_tensor, std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>  access_expr_list)
      {
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
      }

      virtual int AnalyzeLoopBlockID(int current_id)
      {
        loop_block_id_ = current_id;
        return current_id;
      }

      std::shared_ptr<Container> getTarget()
      {
        return target_;
      }
  
      std::shared_ptr<Expression> getBodyExpr()
      {
        return body_expr_;
      }
  }; // End of class PrimAssignment


  class PrimFor : public Statement
  {
    protected:
      std::shared_ptr<Statement> init_stmt_;
      std::shared_ptr<Expression> test_expr_;
      std::shared_ptr<Statement> incr_stmt_;
      std::shared_ptr<Statement> body_stmt_;
  
    public:
      PrimFor(std::shared_ptr<Statement> init_s, std::shared_ptr<Expression> test_e, std::shared_ptr<Statement> incr_s) :
        Statement(GetNumIterations(init_s, test_e)),
        init_stmt_(init_s),
        test_expr_(test_e),
        incr_stmt_(incr_s)
      {
      }

      PrimFor(std::shared_ptr<Statement> init_s, std::shared_ptr<Expression> test_e, std::shared_ptr<Statement> incr_s, std::shared_ptr<Statement> body_s) :
        Statement(GetNumIterations(init_s, test_e)),
        init_stmt_(init_s),
        test_expr_(test_e),
        incr_stmt_(incr_s),
        body_stmt_(body_s)
      {
      }

      PrimFor(std::shared_ptr<Statement> init_s, std::shared_ptr<Expression> test_e, std::shared_ptr<Statement> incr_s, std::shared_ptr<Statement> body_s, double exec_prob) :
        Statement(exec_prob, GetNumIterations(init_s, test_e)),
        init_stmt_(init_s),
        test_expr_(test_e),
        incr_stmt_(incr_s),
        body_stmt_(body_s)
      {
      }
 
      virtual StatementClass GetClass()
      {
        return StatementClass::PRIM_FOR;
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
         //TODO
         std::string ret = "";
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
      }

      virtual void EvaluateNumExecs(int mult)
      {
        num_executions_ *= mult;
        init_stmt_->EvaluateNumExecs(mult);
        incr_stmt_->EvaluateNumExecs(mult);
        int body_mult = mult * GetNumIterations(init_stmt_, test_expr_);
        auto stmt = body_stmt_;
        while(stmt != nullptr && stmt != NULL)
        {
          stmt->EvaluateNumExecs(body_mult);
        } 
      }

      virtual int AnalyzeLoopBlockID(int current_id)
      {
        loop_block_id_ = current_id;
        loop_block_id_base_ = current_id;
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

      void setBodyStmt(std::shared_ptr<Statement> stmt)
      {
        body_stmt_ = stmt;;
      }
  
      std::shared_ptr<Statement> getInitStmt()
      {
        return init_stmt_;
      }
  
      std::shared_ptr<Expression> getTestExpr()
      {
        return test_expr_;
      }
  
      std::shared_ptr<Statement> getIncrStmt()
      {
        return incr_stmt_;
      }
  
      std::shared_ptr<Statement> getBodyStmt()
      {
        return body_stmt_;
      }
  }; // End of class PrimFor

  class CondStatement : public Statement
  {
    protected:
      std::shared_ptr<Expression> test_expr_;
      std::shared_ptr<Statement> body_stmt_;
  
    public:
  
      CondStatement(std::shared_ptr<Statement> body_s) :
        body_stmt_(body_s)
      {
      }

      CondStatement(std::shared_ptr<Statement> body_s, double exec_prob) :
        Statement(exec_prob),
        body_stmt_(body_s)
      {
      }
 
      CondStatement(std::shared_ptr<Expression> test_e, std::shared_ptr<Statement> body_s) :
        test_expr_(test_e),
        body_stmt_(body_s)
      {
      }

      CondStatement(std::shared_ptr<Expression> test_e, std::shared_ptr<Statement> body_s, double exec_prob) :
        Statement(exec_prob),
        test_expr_(test_e),
        body_stmt_(body_s)
      {
      }

  
      virtual StatementClass GetClass()
      {
        return StatementClass::COND_STATEMENT;
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
        std::string ret = boost::str(boost::format("(Conditional statement, loop_block_id: %d, num_exec: %d, \n test_expression: %s, \n body_statement: %s)") % loop_block_id_ % num_executions_ % test_expr_->ToString() % body_stmt_->ToString());
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
      }

      virtual void EvaluateNumExecs(int mult)
      {
        num_executions_ *= mult;
        body_stmt_->EvaluateNumExecs(mult);
      }

      virtual int AnalyzeLoopBlockID(int current_id)
      {
        loop_block_id_ = current_id;
        int last_id = body_stmt_->AnalyzeLoopBlockID(current_id);
        return last_id;
      }
  }; // End of class CondStatement


};// End of namespace whoop
#endif
