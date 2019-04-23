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

#ifndef TIMEWHOOP_ANALYSIS_STRUCTURE_HPP_
#define TIMEWHOOP_ANALYSIS_STRUCTURE_HPP_

#include "maybe-type.hpp"
#include "pure-abstract-syntax-types.hpp"
#include "pure-abstract-primitive-syntax.hpp"

#include "analysis-structure-types.hpp"

namespace timewhoop
{

/*** Analytsis structure
- Description: This flie includes definitions of some data structure
  that contains some extracted information from timewhoop's AST

- The data structures (or, classes) in this file provides better
abstraction of whoop program information than raw c++ STL data
structures

*/

  class ExpressionInformation
  {
    protected:
      Expression* target_expr_;

    public:
      ExpressionInformation() = default;

      ExpressionInformation(Expression* targ_expr) :
        target_expr_(targ_expr)
      {
      }

      Expression* GetTargetExpr()
      {
        return target_expr_;
      }
  }; // End of class Expression Information

  class ContainerInformation : public ExpressionInformation
  {
    protected:
      Container* target_ctnr_;

    public:
      ContainerInformation() = default;

      ContainerInformation(Container* targ_ctnr) :
        ExpressionInformation(targ_ctnr),
        target_ctnr_(targ_ctnr)
      {
      }

      Container* GetTargetCtnr()
      {
        return target_ctnr_;
      }
  }; // End of class Container Information


  class StatementInformation
  {
    protected:
      Statement* target_stmt_ = nullptr;

    public:
      StatementInformation() = default;

      StatementInformation(Statement* targ_stmt) :
        target_stmt_(targ_stmt)
      {
      }

      Statement* GetTargetStmt()
      {
        return target_stmt_;
      }
  }; // End of class StatementInformation


  class LoopInformation : public StatementInformation
  {
    protected:
      std::shared_ptr<Container> loop_variable_;
      std::shared_ptr<std::list<std::pair<std::string, int>>> tensor_buffer_levels_;

      int loop_block_id_base_ = 0;
      int loop_block_id_bound_ = 0;
      int iteration_base_ = 0;
      int iteration_bound_ = 0;

      bool IsUnderThisLoop(int block_id)
      {
        return ((loop_block_id_base_ <= block_id) && (block_id <= loop_block_id_bound_));
      }

    public:
      LoopInformation(Statement* targ_stmt,
                      std::shared_ptr<Container> loop_var,
                      int loop_blk_id_base,
                      int loop_blk_id_bound,
                      int iter_base,
                      int iter_bound
                     ) :
        StatementInformation(targ_stmt),
        loop_variable_(loop_var),
        loop_block_id_base_(loop_blk_id_base),
        loop_block_id_bound_(loop_blk_id_bound),
        iteration_base_(iter_base),
        iteration_bound_(iter_bound)
      {
      }

      LoopInformation(std::shared_ptr<Statement> for_loop_stmt, int iter_base, int iter_bound) :
        loop_variable_(for_loop_stmt->GetOwner()),
        loop_block_id_base_(for_loop_stmt->GetBlockIDBase()),
        loop_block_id_bound_(for_loop_stmt->GetBlockIDBound()),
        iteration_base_(iter_base),
        iteration_bound_(iter_bound)
      {
      }

      std::string ToString()
      {
        std::string ret = boost::str(boost::format("Loop Varialbe: (%s), loop_block_id_base: %d, loop_block_id_bound_: %d, iter_base: %d, iter_bound: %d") % loop_variable_->ToString() % loop_block_id_base_ % loop_block_id_bound_ % iteration_base_ % iteration_bound_);

        return ret;
      }

      std::shared_ptr<Container> GetLoopVar()
      {
        return loop_variable_;
      }

      bool IsOwner(std::shared_ptr<Container> test_var)
      {
        return test_var == loop_variable_; // This works because timewhoop construct each statement once and always use pointer to refer it
      }

      bool IsTargetLoop(std::shared_ptr<Container> test_var, int block_id)
      {
        return IsUnderThisLoop(block_id) && IsOwner(test_var);
      }

      int GetIterBase()
      {
        return iteration_base_;
      }

      int GetIterBound()
      {
        return iteration_bound_;
      }

  };  // End of class LoopInformation


  class LoopInformationTable
  {
    private:
      void HandleError(LoopInfoError error_type)
      {
        //TODO: Use better command line message printouts
        //TODO: Add error count to report "compilation error"
        switch(error_type)
        {
          case(LoopInfoError::Duplicate):
          {
            std::cout << "[Timewhoop] Warning: Detected an invalid loop nest - Within a nest, the same loop variable appears multiple times" << std::endl;
            break;
          }
          default:
          {
            std::cout << "[Timewhoop] Warning: Detected an unknown errror" << std::endl;
          }
        }
      }

    protected:
      std::list<std::shared_ptr<LoopInformation>> loop_info_table_;

    public:
      LoopInformationTable()
      {
      }

      std::string ToString()
      {
        std::string ret;
        for(auto& loop_info : loop_info_table_)
        {
          if(loop_info != nullptr)
          {
            ret += loop_info->ToString();
            ret += "\n";
          }
        }
        return ret;
      }

      bool HasDuplicate(std::shared_ptr<Container> test_var)
      {
        bool found_it = false;

        for(auto& it : loop_info_table_)
        {
          if(it->IsOwner(test_var))
            found_it = true;
        }

        return found_it;
      }

      void AddLoopInfo(std::shared_ptr<LoopInformation> new_info)
      {
          loop_info_table_.push_back(new_info);
      }

      void AddLoop(  Statement* targ_stmt,
                     std::shared_ptr<Container> loop_var,
                     int loop_blk_id,
                     int loop_blk_id_base,
                     int loop_blk_id_bound,
                     int iter_base,
                     int iter_bound)
      {
          auto new_entry = std::make_shared<LoopInformation>(targ_stmt, loop_var, loop_blk_id_base, loop_blk_id_bound, iter_base, iter_bound);
          loop_info_table_.push_back(new_entry);
      }

      Maybe<std::shared_ptr<LoopInformation>> SearchTable(Container* var, int loop_block_id)
      {
        Maybe<std::shared_ptr<LoopInformation>> ret;

        for(auto& loop_info : loop_info_table_)
        {
          if(loop_info->GetLoopVar()->GetName() == var->GetName()) //TODO: verify this comparison
          {
            ret.SetValue(loop_info);
          }
        }

        return ret;
      }

      Maybe<std::shared_ptr<LoopInformation>> SearchTable(Container* var, int loop_block_id, int buffer_level)
      {
        Maybe<std::shared_ptr<LoopInformation>> ret;

        for(auto& loop_info : loop_info_table_)
        {
          if(loop_info->GetLoopVar()->GetName() == var->GetName())
          {
            ret.SetValue(loop_info);
          }
        }

        return ret;
      }



  }; //End of class LoopInformationTable

}; //End of namespace timewhoop

#endif
