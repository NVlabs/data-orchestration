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

#include <iostream>
#include <memory>

#include "whoop.hpp"
#include "options.hpp"

#include "pure-abstract-syntax-types.hpp"
#include "pure-abstract-syntax.hpp"
#include "pure-abstract-syntax-tree.hpp"

#include "yaml-writer.hpp"

namespace timewhoop
{
  //Execution syntax tree
  using whoop::the_program;
  using whoop::all_vars;
  using whoop::all_tensors;

  PureAbstractSyntaxTree pure_ast;

  YamlWriter yaml_writer("problem_shape.yaml");


  void ConstructAST()
  {
    /* 1. Process declarations */
    // 1-(a) Process variable declarations
    for(auto& var : all_vars)
    {
      std::shared_ptr<Variable> var_exp = std::shared_ptr<Variable>(new Variable(Type::INTEGER, var->name_));
      std::shared_ptr<Statement> var_decl = std::shared_ptr<VariableDeclaration>(new VariableDeclaration(var_exp));

      pure_ast.AddStatement(var_decl);
    } 

    // 1-(b) Process tensor declarations

    // For reference check, whoop framework adds another tensor (TensorIn)
    // to read reference results. This is not algorithmic; need to be removed
    // in AST.
    if(whoop::options::kShouldCheckReferenceOutput)
    {
      all_tensors.pop_back();
    }
    for(auto& tensor : all_tensors)
    {
      std::shared_ptr<Tensor> tensor_exp = std::shared_ptr<Tensor>(new Tensor(Type::INTEGER, tensor->name_, tensor->dim_sizes_));
      std::shared_ptr<Statement> tensor_decl = std::shared_ptr<TensorDeclaration>(new TensorDeclaration(tensor_exp));

      pure_ast.AddStatement(tensor_decl);
    } 

    /* 2. Walk through execution syntax tree and construct abstract syntax tree */
    std::shared_ptr<whoop::ast::Statement> stmt = std::shared_ptr<whoop::ast::Statement>(the_program.beginning_stmt_);

    if(stmt != NULL && stmt != nullptr)
    {
      // Virtual function, ConvertStatement, recursively constructs AST
      std::shared_ptr<Statement> pure_ast_root = stmt->ConvertStatement(); 
      pure_ast.ConstructAST(pure_ast_root);
    }
  
    /* 3. Post-process AST for further analysis */
    pure_ast.PostProcessAST();

  } // End of void constructAST()

  void AnalyzeTensors()
  {
    auto lst = pure_ast.GetTensorList();
    yaml_writer.write_mapping_key(0, "problemShape");
    yaml_writer.write_mapping_key(1, "computeSpaces");
    yaml_writer.write_list_item(2, "name: " + the_program.name_, false);


    auto tensor_asgn_list = std::make_shared<std::list<std::shared_ptr<Statement>>>();
    pure_ast.SearchStatements(tensor_asgn_list, StatementClass::TENSOR_ASSIGNMENT);


    //TODO: Deternmine what to put in operations field
    std::string tensor_asgn_str;

    for(auto& tensor_asgn: *tensor_asgn_list)
    {
      std::cout << tensor_asgn->ToString() << std::endl;
      tensor_asgn_str += tensor_asgn->ToString();
    }

    yaml_writer.write_mapping_key(2, "  operations", false);

    yaml_writer.write_mapping_contents(0, tensor_asgn_str + "\n");


    auto var_decl_list = std::make_shared<std::list<std::shared_ptr<Statement>>>();
    pure_ast.SearchStatements(var_decl_list, StatementClass::VARIABLE_DECLARATION);

    std::string variable_list;

    for(auto& var_decl : *var_decl_list)
    {
      variable_list += var_decl->GetOwner()->GetName();
      variable_list += ",";
    }

    variable_list.pop_back();

    yaml_writer.write_mapping_contents(2, "  dimensions: [" + variable_list  + "]\n");
   

    yaml_writer.write_mapping_key(1, "dataSpaces");
    for(auto& tensor : *lst)
    {
      std::cout << tensor->ToString() << std::endl;
      auto leaf_list = pure_ast.GetTensorIndexVariables(tensor);
      int dim = 0;
      yaml_writer.write_list_item(2, "name: " + tensor->GetName(), false);
      yaml_writer.write_mapping_key(2, "  pointMappings");
      yaml_writer.write_list_item(3, "computeSpace: " + the_program.name_, false);

      for(auto& it : *leaf_list)
      {
        std::cout << "dimension" << dim << std::endl;
        for(auto& expr: *it)
        {
          std::cout << expr->ToString() << std::endl;
          yaml_writer.write_list_item(4, expr->GetName(), false);
        }
        dim++;
      }
    }
  }


  void PrintAST()
  {
    pure_ast.PrintAST();
  }

/* TODO: Extend functionality for separte read/write counts
  std::shared_ptr<std::list<std::shared_ptr<Expression>>>> GetTensorAccesses()
  {
    //Tensor reads
    auto tensor_access_list = std::make_shared<std::list<std::shared_ptr<Expression>>>();
    pure_ast.SearchExpressions(tensor_access_list, ExpressionClass::TENSOR_ACCESS);

    //Tensor writes
    auto tensor_assignment_list = std::make_shared<std::list<std::shared_ptr<Statement>>>();
    pure_ast.SearchStatements(tensor_assignment_list, StatementClass::TENSOR_ASSIGNMENT);
  }
*/

  void AnalyzeBufferAccess()
  {
    std::map<std::string, uint64_t> local_buffer_requirement_stats;
    std::map<std::string, uint64_t> local_buffer_access_stats;
    std::map<std::string, uint64_t> backing_buffer_access_stats;

    auto loop_info_table = pure_ast.GetLoopInformation();
    std::cout << loop_info_table->ToString() << std::endl;

    auto tensor_access_list = std::make_shared<std::list<std::shared_ptr<Expression>>>();
    pure_ast.SearchExpressions(tensor_access_list, ExpressionClass::TENSOR_ACCESS);
   
    for(auto& tensor_access: *tensor_access_list)
    {
      auto tensor_name = tensor_access->GetName();
      int block_id = tensor_access->GetLoopBlockID();
      auto idx_expr_list = dynamic_cast<TensorAccess*>(tensor_access.get())->GetIdxExpr();
      int dim = 0;

      int accessed_volume = 1;

      if(local_buffer_requirement_stats.find(tensor_name) == local_buffer_requirement_stats.end())
      {
        local_buffer_requirement_stats[tensor_name] = 1;
      }

      for(auto& idx_expr : *idx_expr_list)
      {
        auto range = idx_expr->MinMaxEval(loop_info_table, block_id);
        if(range.IsValid())
        {
          auto range_value = range.GetValue();
          accessed_volume *= std::get<1>(range_value) - std::get<0>(range_value);
        }
        else
        {
          // TODO: Add better error handler
          std::cout << "[timewhoop-AnalyzeLoops] Range evaluation failed for expression " << idx_expr->ToString() << " ; Non-static index variables" << std::endl;
        } 
      }

      // Update buffer requirements
      if(local_buffer_requirement_stats[tensor_name] < accessed_volume)
      {
        local_buffer_requirement_stats[tensor_name] = accessed_volume;
      }

//      int backing_buffer_access = (tensor_access->GetAccessProb() == 0.0)? 0 : accessed_volume;
      int backing_buffer_access = static_cast<int>(tensor_access->GetAccessProb() * static_cast<double>(accessed_volume));
      int local_buffer_access = static_cast<int>(static_cast<double>(tensor_access->GetNumAccess()) * tensor_access->GetAccessProb());

      std::cout << std::endl;
      std::cout << "-------------" << tensor_access->GetName() << " -----------" << std::endl;
      std::cout << "For Tensor Access " << tensor_access->ToString() << std::endl;
      std::cout << "=>Backing buffer access: "  << backing_buffer_access << std::endl; 
      std::cout << "=>Local buffer access " << ": " << local_buffer_access << std::endl;
      std::cout << std::endl;

      local_buffer_access_stats[tensor_name] = local_buffer_access_stats[tensor_name] + local_buffer_access;
      backing_buffer_access_stats[tensor_name] = backing_buffer_access_stats[tensor_name] + backing_buffer_access;
    }

    std::cout << "\n\n==========Aggregated results==========" << std::endl;
    std::cout << "Local Buffer accesses" << std::endl;
    for(auto& tensor : local_buffer_access_stats)
    {
      std::cout << tensor.first << ": " << tensor.second << std::endl;
    }

    std::cout << "\nBacking Buffer accesses" << std::endl;
    for(auto& tensor : backing_buffer_access_stats)
    {
      std::cout << tensor.first << ": " << tensor.second << std::endl;
    }

    std::cout << "\nLocal Buffer size requirements" << std::endl;
    for(auto& tensor : local_buffer_requirement_stats)
    {
      std::cout << tensor.first << ": " << tensor.second << std::endl;
    }

  }

};
