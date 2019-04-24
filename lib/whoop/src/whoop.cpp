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
#include <assert.h>

#include "whoop.hpp"
#include "abstract-syntax.hpp"
#include "operator-semantics.hpp"

namespace whoop
{

Program the_program{};
StatsCollection top_stats{"top"};
std::deque<int> spatial_partition_levels{};
std::list<InFromFile*> need_input{};
std::list<OutToFile*> need_output{};
std::list<Tensor*> all_tensors{};
std::list<Var*> all_vars{};
int num_warnings_ = 0;
UserTracer user_tracer_;
Flusher EndT{};


WHOOP_DEFINE_STMT_BINARY_OPERATOR(=, ToAssignment(body_e.expr_));
WHOOP_DEFINE_STMT_BINARY_OPERATOR(+=, ToUpdateOp(PlusOp, body_e.expr_));
WHOOP_DEFINE_STMT_BINARY_OPERATOR(-=, ToUpdateOp(MinusOp, body_e.expr_));
WHOOP_DEFINE_STMT_BINARY_OPERATOR(*=, ToUpdateOp(MulOp, body_e.expr_));
WHOOP_DEFINE_STMT_BINARY_OPERATOR(/=, ToUpdateOp(DivOp, body_e.expr_));
WHOOP_DEFINE_STMT_BINARY_OPERATOR(&=, ToUpdateOp(ANDOp, body_e.expr_));
WHOOP_DEFINE_STMT_BINARY_OPERATOR(|=, ToUpdateOp(OROp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(+, ToBinaryOp(PlusOp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(-, ToBinaryOp(MinusOp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(*, ToBinaryOp(MulOp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(/, ToBinaryOp(DivOp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(%, ToBinaryOp(ModOp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(==, ToBinaryOp(EQOp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(!=, ToBinaryOp(NEQOp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(>=, ToBinaryOp(GTEOp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(<=, ToBinaryOp(LTEOp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(>, ToBinaryOp(GTOp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(<, ToBinaryOp(LTOp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(&&, ToBinaryOp(ANDOp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(||, ToBinaryOp(OROp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(&, ToBinaryOp(BWANDOp, body_e.expr_));
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(|, ToBinaryOp(BWOROp, body_e.expr_));


WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(+, ToBinaryOp(c, PlusOp, body_e.expr_));
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(-, ToBinaryOp(c, MinusOp, body_e.expr_));
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(*, ToBinaryOp(c, MulOp, body_e.expr_));
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(/, ToBinaryOp(c, DivOp, body_e.expr_));
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(%, ToBinaryOp(c, ModOp, body_e.expr_));
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(==, ToBinaryOp(c, EQOp, body_e.expr_));
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(!=, ToBinaryOp(c, NEQOp, body_e.expr_));
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(>=, ToBinaryOp(c, GTEOp, body_e.expr_));
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(<=, ToBinaryOp(c, LTEOp, body_e.expr_));
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(>, ToBinaryOp(c, GTOp, body_e.expr_));
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(<, ToBinaryOp(c, LTOp, body_e.expr_));
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(&&, ToBinaryOp(c, ANDOp, body_e.expr_));
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(||, ToBinaryOp(c, OROp, body_e.expr_));
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(&, ToBinaryOp(c, BWANDOp, body_e.expr_));
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(|, ToBinaryOp(c, BWOROp, body_e.expr_));

WHOOP_DEFINE_DERIVED_OPS(TreeBuilder)
WHOOP_DEFINE_DERIVED_OPS(Var)
WHOOP_DEFINE_DERIVED_OPS(TensorDisambiguator)



void s_for(ast::PrimVar& v, const int& init_const, const int& end_const)
{
  // The initialization statement is always a var assignment to 0.
  assert(init_const == 0);
  ast::Constant* init_expr = new ast::Constant(init_const);
  ast::VarAssignment* init_s = new ast::VarAssignment(v, init_expr);
  // The test is always less-than.
  ast::VarAccess* test_v = new ast::VarAccess(v);
  ast::Constant* end_expr = new ast::Constant(end_const);
  ast::BinaryOp* test_e = new ast::BinaryOp(LTOp, test_v, end_expr);
  // The increment is always by 1.
  ast::VarAccess* incr_v = new ast::VarAccess(v);
  ast::Constant* incr_c = new ast::Constant(1);
  ast::BinaryOp* incr_e = new ast::BinaryOp(PlusOp, incr_v, incr_c);
  ast::VarAssignment* incr_s = new ast::VarAssignment(v, incr_e);
  // Now put it all together.
  ast::SpatialFor* stmt = new ast::SpatialFor(end_const, init_s, test_e, incr_s);
  spatial_partition_levels.push_back(end_const);
  the_program.AddIncomplete(stmt);
}

void s_for(ast::PrimVar& v,  Var& init_const,  Var& end_const)
{
    s_for(v, init_const.Access(), end_const.Access());
}

void t_for(ast::PrimVar& v, TreeBuilder init_expr, TreeBuilder end_expr)
{
  // The initialization statement is always a var assignment for now.
  ast::VarAssignment* init_s = new ast::VarAssignment(v, init_expr.expr_);
  // The test is always less-than, non-inclusive
  ast::VarAccess* test_v = new ast::VarAccess(v);
  ast::BinaryOp* test_e = new ast::BinaryOp(LTOp, test_v, end_expr.expr_);
  // The increment is always by 1.
  ast::VarAccess* incr_v = new ast::VarAccess(v);
  ast::Constant* incr_c = new ast::Constant(1);
  ast::BinaryOp* incr_e = new ast::BinaryOp(PlusOp, incr_v, incr_c);
  ast::VarAssignment* incr_s = new ast::VarAssignment(v, incr_e);
  // Now put it all together.
  ast::TemporalFor* stmt = new ast::TemporalFor(init_s, test_e, incr_s);
  spatial_partition_levels.push_back(1);
  the_program.AddIncomplete(stmt);
}

void t_for(ast::PrimVar& v, const int& init_const, TreeBuilder end_expr)
{
  ast::Constant* init_e = new ast::Constant(init_const);
  t_for(v, TreeBuilder(init_e), end_expr);
}


void t_for(ast::PrimVar& v, TreeBuilder init_expr, const int& end_const)
{
  ast::Constant* end_e = new ast::Constant(end_const);
  t_for(v, init_expr, TreeBuilder(end_e));
}

void t_for(ast::PrimVar& v, const int& init_const, const int& end_const)
{
  ast::Constant* init_e = new ast::Constant(init_const);
  ast::Constant* end_e = new ast::Constant(end_const);
  t_for(v, TreeBuilder(init_e), TreeBuilder(end_e));
}

void t_for(ast::PrimVar& v,  Var& init_const,  Var& end_const)
{
  ast::VarAccess* init_e = new ast::VarAccess(init_const);
  ast::VarAccess* end_e = new ast::VarAccess(end_const);
  t_for(v, TreeBuilder(init_e), TreeBuilder(end_e));
}

void t_for(ast::PrimVar& v,  const int& init_const,  Var& end_const)
{
  ast::Constant* init_e = new ast::Constant(init_const);
  ast::VarAccess* end_e = new ast::VarAccess(end_const);
  t_for(v, TreeBuilder(init_e), TreeBuilder(end_e));
}

void t_for(ast::PrimVar& v,  TreeBuilder init_const,  Var& end_const)
{
  ast::VarAccess* end_e = new ast::VarAccess(end_const);
  t_for(v, init_const, TreeBuilder(end_e));
}


void t_for(ast::PrimVar& v, const TensorDisambiguator& init_const, const TensorDisambiguator& end_const)
{
  ast::TensorAccess* init_e = new ast::TensorAccess(init_const.target_, init_const.idx_exprs_);
  ast::TensorAccess* end_e = new ast::TensorAccess(end_const.target_, end_const.idx_exprs_);
  t_for(v, TreeBuilder(init_e), TreeBuilder(end_e));
}

void w_while(TreeBuilder test_expr)
{
  ast::While* stmt = new ast::While(test_expr.expr_);
  spatial_partition_levels.push_back(1);
  the_program.AddIncomplete(stmt);
}

void w_while(const int& test_const)
{
  ast::Constant* const_expr = new ast::Constant(test_const);
  w_if(TreeBuilder(const_expr));
}


void w_if(TreeBuilder test_expr)
{
  ast::If* stmt = new ast::If(test_expr.expr_);
  spatial_partition_levels.push_back(1);
  the_program.AddIncomplete(stmt);
}

void w_if(const int& test_const)
{
  ast::Constant* const_expr = new ast::Constant(test_const);
  w_if(TreeBuilder(const_expr));
}

void w_if(TreeBuilder test_expr, double exec_prob)
{
  ast::If* stmt = new ast::If(test_expr.expr_, exec_prob);
  spatial_partition_levels.push_back(1);
  the_program.AddIncomplete(stmt);
}

void w_if(const int& test_const, double exec_prob)
{
  ast::Constant* const_expr = new ast::Constant(test_const);
  w_if(TreeBuilder(const_expr), exec_prob);
}

void w_else()
{
  ast::Else* stmt = new ast::Else();
  the_program.AddElse(stmt);
}

void w_else_if(TreeBuilder test_expr)
{
  ast::If* stmt = new ast::If(test_expr.expr_);
  the_program.AddElse(stmt);
}

void w_else_if(const int& test_const)
{
  ast::Constant* const_expr = new ast::Constant(test_const);
  w_else_if(TreeBuilder(const_expr));
}


void end()
{
  the_program.CompleteCurrent();
  spatial_partition_levels.pop_back();
  ast::EndLoop* stmt = new ast::EndLoop();
  the_program.Add(stmt);
}

// Sub-classes must define these.
ast::Expression* ToAccess(const int& c)
{
   ast::Constant* const_expr = new ast::Constant(c);
   return const_expr;
}

ast::Statement* ToAssignment(const int& c, ast::Expression* body_e)
{
  assert(false);
  return NULL;
}


// For things like --
ast::Expression* ToUnaryOp(const int& c, int (*op)(const int& a))
{
  ast::Expression* access_e = ToAccess(c);
  ast::UnaryOp* op_e = new ast::UnaryOp(op, access_e);
  return op_e;
}

// For things like *, +
ast::Expression* ToBinaryOp(const int& c, int (*op)(const int& a, const int& b), ast::Expression* body_e)
{
  ast::Expression* access_e = ToAccess(c);
  ast::BinaryOp* op_e = new ast::BinaryOp(op, access_e, body_e);
  return op_e;
}

// For things like +=, *=
ast::Statement* ToUpdateOp(const int& c, int (*op)(const int& a, const int& b), ast::Expression* body_e)
{
  ast::Expression* plus_e = ToBinaryOp(c, op, body_e);
  ast::Statement* assign_stmt = ToAssignment(c, plus_e);
  return assign_stmt;
}

void DumpStats(std::string fname)
{ 
  if (top_stats.ChildrenHaveStats())
  {
    std::ofstream ofile(fname.c_str()); 
    top_stats.DumpStats(ofile, 0, NULL);
  }
}

void LogActivity(std::string fname)
{ 
  std::ofstream ofile(fname.c_str()); 
  ofile << "{" << std::endl;
  ofile << "  \"Units\": [" << std::endl;
  for (auto it = all_tensors.begin(); it != all_tensors.end(); it++)
  {
    (*it)->LogActivity(ofile);
    if (it != std::prev(all_tensors.end()))
    {
      ofile << ",";
    }
    ofile << std::endl;
  }
  ofile << "  ]" << std::endl;
  ofile << "}" << std::endl;
}

int NumSpatialPartitionsFlattened()
{
  int res = 1;
  for (auto it = spatial_partition_levels.begin(); it != spatial_partition_levels.end(); it++)
  {
    res *= (*it);
  }
  return res;
}

void Init(int argc, char** argv)
{
  options::SetOverrides();
  ParseOptions(argc, argv);
  for (auto it = need_input.begin(); it != need_input.end(); it++)
  {
    (*it)->ReadInput();
  }
  the_program.SetName(argv[0]);
  user_tracer_ = {};
}

void Done()
{
    Done("stats.txt");
}

void Done(std::string ofile)
{ 
  for (auto it = all_tensors.begin(); it != all_tensors.end(); it++)
  {
    (*it)->DrainAll();
  }
  DumpStats(ofile);
  if (options::kShouldLogActivity)
  {
    LogActivity("activity.trcl");
  }
  for (auto it = need_output.begin(); it != need_output.end(); it++)
  {
    (*it)->DumpOutput();
  }
  if (options::kShouldCheckReferenceOutput)
  {
    for (auto it = need_output.begin(); it != need_output.end(); it++)
    {
      (*it)->CheckOutput();
    }
  }
  if (num_warnings_ != 0)
  {
    user_tracer_.T(0) << num_warnings_ << " ERRORS OCCURRED." << EndT;
    std::exit(num_warnings_);
  }
  else if (options::kShouldCheckReferenceOutput)
  {
    user_tracer_.T(0) << "PASSED." << EndT;
  }
}

void Run()
{
  user_tracer_.ASSERT(spatial_partition_levels.size() == 0) << "You have a missing end() call in the loop nest -- Please check!" << EndT;
    
  for (auto it = all_tensors.begin(); it != all_tensors.end(); it++)
  {
    (*it)->FixupBufferLevelNumbering();
    (*it)->SetTraceLevel(options::kCurrentTraceLevel);
  }
  the_program.Run();
}

Tracer& T(int l)
{
  //user_tracer_.ASSERT_WARNING(spatial_partition_levels.size() == 0) << "Trace statements are yet not supported within t_for/s_for" << EndT;
  return user_tracer_.T(l);
}


Tracer& ASSERT(bool term)
{
  return user_tracer_.ASSERT(term);
}


}