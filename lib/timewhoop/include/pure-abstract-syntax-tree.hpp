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

#ifndef TIMEWHOOP_PURE_ABSTRACT_SYNTAX_TREE_HPP_
#define TIMEWHOOP_PURE_ABSTRACT_SYNTAX_TREE_HPP_

#include <memory>

#include "pure-abstract-syntax-types.hpp"
#include "pure-abstract-syntax.hpp"
#include "analysis-structure.hpp"

namespace timewhoop
{
/*** Pure abstract syntax tree
- Description: This file encapsulates the entire AST structure.

- The APIs for various analysis passes are implemented in this class

*/

  class PureAbstractSyntaxTree
  {
    protected:
	  std::shared_ptr<Statement> root_ = nullptr;
      std::shared_ptr<Statement> curr_ = nullptr;

    public:
      PureAbstractSyntaxTree() {}

      void ConstructAST(std::shared_ptr<Statement> expr_chain_root)
      {
        if(!root_)
        {
          root_ = expr_chain_root;
        }
        else
        {
          curr_->SetNextStmt(expr_chain_root);
          curr_ = expr_chain_root;
        }
      }

      void AddStatement(std::shared_ptr<Statement> incoming_stmt)
      {
        if(!root_)
        {
          root_ = incoming_stmt;
          curr_ = incoming_stmt;
        }
        else
        {
          curr_->SetNextStmt(incoming_stmt);
          curr_ = incoming_stmt;
        }
      }

      //Need to be called to analyze buffer access counts
      void PostProcessAST()
      {
        auto stmt = root_;
        int last_blk_id = -1;
        while(stmt != nullptr && stmt != NULL)
        {
          stmt->EvaluateNumExecs(1);
          last_blk_id++;
          last_blk_id = stmt->AnalyzeLoopBlockID(last_blk_id);

          stmt->SetExecProb(1.0);
          stmt = stmt->GetNextStmt();
        }
      }

      void PrintAST()
      {
        std::shared_ptr<Statement> stmt = root_;
        std::string printouts;

        while(stmt != nullptr && stmt != NULL)
        {
          printouts += stmt->ToString();
          printouts += "\n";
          stmt = stmt->GetNextStmt();
        }

        std::cout << printouts << std::endl;
      }

      std::string ToString()
      {
        std::shared_ptr<Statement> stmt = root_;
        std::string printouts;

        while(stmt != nullptr && stmt != NULL)
        {
          printouts += stmt->ToString();
          printouts += "\n";
          stmt = stmt->GetNextStmt();
        }

        return printouts;
      }


      std::shared_ptr<LoopInformationTable> GetLoopInformation()
      {
        std::shared_ptr<Statement> stmt = root_;
        auto ret = std::make_shared<LoopInformationTable>();

        while(stmt != nullptr && stmt != NULL)
        {
          stmt->ExtractLoopInfo(ret);
          stmt = stmt->GetNextStmt();
        }

        return ret;
      }

      void SearchStatements(std::shared_ptr<std::list<std::shared_ptr<Statement>>> search_res, StatementClass search_class, bool deep_search = false)
      {
        std::shared_ptr<Statement> stmt = root_;

        if(root_->GetClass() == search_class)
        {
          search_res->push_back(root_);
        }

        if(stmt->GetNextStmt() != nullptr && stmt->GetNextStmt() != NULL)
        {
          stmt->SearchStatements(search_res, search_class, deep_search);
          stmt = stmt->GetNextStmt();
        }
      }

      void SearchExpressions(std::shared_ptr<std::list<std::shared_ptr<Expression>>> search_res, ExpressionClass search_class, bool deep_search = true)
      {
        std::shared_ptr<Statement> stmt = root_;

        if(stmt != nullptr && stmt != NULL)
        {
          stmt->SearchExpressions(search_res, search_class, deep_search);
          stmt = stmt->GetNextStmt();
        }
      }


      /* Passes for the AST */
      /* 1. Problem shape analysis pass */

      // 1-1) Construct tensor list from AST
      //  - This function assumes that no tensor definition is 
      //    made within loop or conditional statements.
      //  - This assumption is valid because the pure-ast 
      //    construction algorithm places all the declartions 
      //    at the beginning of the pure-ast
    public:
      std::shared_ptr<std::list<std::shared_ptr<Container>>> GetTensorList()
      {
        std::shared_ptr<Statement> stmt = root_;
        std::shared_ptr<std::list<std::shared_ptr<Container>>> tensor_list = std::make_shared<std::list<std::shared_ptr<Container>>>(); 

        while(stmt != nullptr && stmt != NULL)
        {
          if(stmt->GetClass() == StatementClass::TENSOR_DECLARATION)
          {
            std::shared_ptr<Container> tensor = stmt->GetOwner(); 
            tensor_list->push_back(tensor);
          }
          stmt = stmt->GetNextStmt();
        }

        return tensor_list; 
      }

      // 1-2) Extract index expressions
      //   - This function extracts all the raw index expressions of the target 
      //     tensor given as an argument
      //   - The structure of return type:
      //         A pointer to a list of 
      //           pointers to an index expression list of each tensor accesses
      //         An index expression list 
      //     e.g.,) tensorA[1][2][3] = tensorA[a+b][c+d][a-b] + tensorB[b][d][b]
      //            For tensorA query,
      //              [ &[1, 2, 3], &[a+b, c+d, a-b] ]
      //            For tensorB query,
      //              [ &[b,d,b] ]
      //              * '&' means a smart pointer(shared_ptr)
    private:
      std::shared_ptr<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>> GetRawIndexExpressions(std::shared_ptr<Container> targ_tensor)
      {
        std::shared_ptr<Statement> stmt = root_;
        auto expr_list = std::make_shared<std::list<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>();

        while(stmt != nullptr && stmt != NULL)
        {
          stmt->ExtractTensorAccessExpressions(targ_tensor, expr_list); 
          stmt = stmt->GetNextStmt();
        }

        return expr_list;
      }

      // 1-3) Get index expression list for each dimension
    public:
      std::shared_ptr<std::vector<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>> GetIndexExpressions(std::shared_ptr<Container> targ_tensor)
      {
        auto raw_expr_list = GetRawIndexExpressions(targ_tensor);
        int num_dims = targ_tensor->GetNumDims();
        auto ret = std::make_shared<std::vector<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>(num_dims);

        for(auto& expr_dim : *ret)
        {
          expr_dim = std::make_shared<std::list<std::shared_ptr<Expression>>>();
        }

        for(auto& it : *raw_expr_list)
        {
          int dim = 0;
          for(auto& sub_exp : *it)
          {
            ret->at(dim)->push_back(sub_exp);
            dim++;
          }
        }

        for(auto& expr_dim : *ret)
        {
          expr_dim->sort();
          expr_dim->unique();
        }

        return ret;
      }

      // 1-4) Extracts only variables
    public:
      std::shared_ptr<std::vector<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>> GetTensorIndexVariables(std::shared_ptr<Container> targ_tensor)
      {
        std::less<std::shared_ptr<Expression>> expr_cmp;
        auto idx_exprs = GetIndexExpressions(targ_tensor);
        int num_dims = targ_tensor->GetNumDims();
        auto ret = std::make_shared<std::vector<std::shared_ptr<std::list<std::shared_ptr<Expression>>>>>(num_dims);

        int dim = 0;
        for(auto& it : *idx_exprs)
		{
          auto leaf_list = std::make_shared<std::list<std::shared_ptr<Expression>>>();
          for(auto& sub_exp : *it) 
          {
            if(sub_exp->GetClass() == ExpressionClass::VARIABLE)
            {
              leaf_list->push_back(sub_exp);
            }
            else 
            {
              sub_exp->GetLeaves(leaf_list);
            }
          }

          leaf_list->sort(expr_cmp);
          leaf_list->unique(expr_cmp);

          ret->at(dim) = leaf_list;
          dim++;
        }

        return ret;
      }

      /* 2. Loop analysis pass */
    public:


      /* 3. Combinational pass  */
      void AnalyzeTensorAccess()
      {
        std::shared_ptr<std::list<std::shared_ptr<Statement>>> forloop_list = std::make_shared<std::list<std::shared_ptr<Statement>>>();
        root_->SearchStatements(forloop_list, StatementClass::TEMPORAL_FOR, false);
        root_->SearchStatements(forloop_list, StatementClass::SPATIAL_FOR, false);

        std::shared_ptr<std::list<std::shared_ptr<Expression>>> tensor_access_list = std::make_shared<std::list<std::shared_ptr<Expression>>>();
        root_->SearchExpressions(tensor_access_list, ExpressionClass::TENSOR_ACCESS, true);
    
        for(auto& tensor_access : *tensor_access_list) 
        {
          std::cout << tensor_access->ToString() << std::endl;
        }
      }

  }; //End of class PureAbstractSyntaxTree
}; //End of namespace timewhoop

#endif
