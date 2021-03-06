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
std::deque<std::pair<ast::SpatialFor*, int>> spatial_partition_levels{};
std::deque<int> spatial_partition_levels_sanity_check{};
std::vector<std::vector<std::deque<int>>> tile_level_deliminators{};
std::vector<std::vector<int>> current_tile_level{};
int max_tile_level = 0;
std::deque<int> global_tile_level_deliminators{};
std::vector<int> compute_tile_levels{1};
int current_global_tile_level = 0;
bool need_global_tile_level = false;
std::vector<int> tile_level_spatial_expansions{1};
int max_flat_expansion = 1;
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
WHOOP_DEFINE_EXPR_BINARY_OPERATOR(<<=, ToBinaryOp(IntEQOp, body_e.expr_));
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
WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(<<=, ToBinaryOp(c, IntEQOp, body_e.expr_));
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
  the_program.AddIncomplete(stmt);
  // Mark the spatial expansions as proper.
  int height = ActualSpatialPartitionHeight(true);
  if (height >= spatial_partition_levels_sanity_check.size())
  {
    // This is a new "highest" spatial for level, so
    // expand all previous spatial fors in the stack by our number.
    for (auto sp : spatial_partition_levels)
    {
      if (sp.first != NULL)
      {
        sp.first->flat_expansion_factor_ *= end_const;
      }
    }
    spatial_partition_levels.push_back(std::make_pair(stmt, end_const));
    spatial_partition_levels_sanity_check.push_back(end_const);
    tile_level_spatial_expansions[max_tile_level] *= end_const;
    need_global_tile_level = true;
    compute_tile_levels.back() *= end_const;
  }
  else
  {
    // The stack has been popped previously, so
    // do a sanity check that the new sequential loops are consistent. 
    bool is_consistent = spatial_partition_levels_sanity_check[height] == end_const;
    user_tracer_.ASSERT(is_consistent) << "Inconsistent spatial partitioning across sequential loops. Previously level " << height  << " was: " << spatial_partition_levels_sanity_check[height] << ", now it is: " << end_const << EndT;
    
    // Just re-push, but don't re-mutate anything.
    spatial_partition_levels.push_back(std::make_pair(stmt, end_const));
    // If I'm not the last "old" s_for...
    if (height != spatial_partition_levels_sanity_check.size() - 1)
    {
      // Fixup our expansion factor by any "future" s_fors we know are coming.
       for (auto it = spatial_partition_levels_sanity_check.begin() + height + 1; it != spatial_partition_levels_sanity_check.end(); it++)
      {
        stmt->flat_expansion_factor_ *= (*it);
      }
    }
  }
}

/*
void s_for(ast::PrimVar& v,  Var& init_const,  Var& end_const)
{
    s_for(v, init_const.Access(), end_const.Access());
}
*/
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
  spatial_partition_levels.push_back(std::pair<ast::SpatialFor*, int>(NULL, 1));
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
  ast::TensorAccess* init_e = init_const.ToTensorAccess();
  ast::TensorAccess* end_e = end_const.ToTensorAccess();
  t_for(v, TreeBuilder(init_e), TreeBuilder(end_e));
}

void w_while(TreeBuilder test_expr)
{
  ast::While* stmt = new ast::While(test_expr.expr_);
  spatial_partition_levels.push_back(std::pair<ast::SpatialFor*, int>(NULL, 1));
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
  spatial_partition_levels.push_back(std::pair<ast::SpatialFor*, int>(NULL, 1));
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
  spatial_partition_levels.push_back(std::pair<ast::SpatialFor*, int>(NULL, 1));
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
  // Check if this end() matches up with an AddTileLevel for any tensor.
  for (int x = 0; x < all_tensors.size(); x++)
  {
    for (int p = 0; p < tile_level_deliminators[x].size(); p++)
    {
      if (tile_level_deliminators[x][p].size() > 0)
      {
        if (spatial_partition_levels.size() == tile_level_deliminators[x][p].back())
        {
          tile_level_deliminators[x][p].pop_back();
          current_tile_level[x][p]--;
        }
      }
    }
  }
  if (global_tile_level_deliminators.size() > 0)
  {
    if (spatial_partition_levels.size() == global_tile_level_deliminators.back())
    {
      global_tile_level_deliminators.pop_back();
      current_global_tile_level--;
    }
  }
  spatial_partition_levels.pop_back();
  ast::EndLoop* stmt = new ast::EndLoop();
  the_program.Add(stmt);
}

// Sub-classes must define these.
ast::Expression* ToAccess(const DataType_t& c)
{
   ast::Constant* const_expr = new ast::Constant(c);
   return const_expr;
}

ast::Statement* ToAssignment(const DataType_t& c, ast::Expression* body_e)
{
  assert(false);
  return NULL;
}


// For things like --
ast::Expression* ToUnaryOp(const DataType_t& c, DataType_t (*op)(const DataType_t& a))
{
  ast::Expression* access_e = ToAccess(c);
  ast::UnaryOp* op_e = new ast::UnaryOp(op, access_e);
  return op_e;
}

// For things like *, +
ast::Expression* ToBinaryOp(const DataType_t& c, DataType_t (*op)(const DataType_t& a, const DataType_t& b), ast::Expression* body_e)
{
  ast::Expression* access_e = ToAccess(c);
  ast::BinaryOp* op_e = new ast::BinaryOp(op, access_e, body_e);
  return op_e;
}

// For things like +=, *=
ast::Statement* ToUpdateOp(const DataType_t& c, DataType_t (*op)(const DataType_t& a, const DataType_t& b), ast::Expression* body_e)
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
    ofile << "," << std::endl;
  }
  LogComputeActivity(ofile);
  ofile << "  ]" << std::endl;
  ofile << "}" << std::endl;
}

void LogPhysicalActivity(std::string fname)
{ 
  std::ofstream ofile(fname.c_str()); 
  ofile << "{" << std::endl;
  ofile << "  \"Units\": [" << std::endl;
  LogPhysicalMap(ofile, false, physical_buffer_map);
  ofile << "," << std::endl;
  LogPhysicalMap(ofile, true, physical_compute_map);
  ofile << "  ]" << std::endl;
  ofile << "}" << std::endl;
}

void LogTensorTopology(std::string fname)
{
  std::ofstream ofile(fname.c_str());
  ofile << "--- #!" << whoop::options::kProgramName << std::endl;
  ofile << "config_name: " << whoop::options::kProgramName << std::endl;
  ofile << "logical:" << std::endl;
  for (auto it = all_tensors.begin(); it != all_tensors.end(); it++)
  {
    (*it)->LogTopologyModules(ofile);
  }
  LogComputeTopology(ofile, all_tensors.size());
  ofile << std::endl;
  ofile << "  - type: logical_connection" << std::endl;
  ofile << "    connections:" << std::endl;

  for (auto it = all_tensors.begin(); it != all_tensors.end(); it++)
  {
    (*it)->LogTopologyConnections(ofile);
  }
  ofile << std::endl;
}


void LogTensorBindings(std::string fname)
{ 
  std::ofstream ofile(fname.c_str());
  ofile << "--- #!" << whoop::options::kProgramName << std::endl;
  ofile << "config_name: " << whoop::options::kProgramName << std::endl;
  ofile << "binding:" << std::endl;

  for (auto it = all_tensors.begin(); it != all_tensors.end(); it++)
  {
    (*it)->LogTopologyRoutes(ofile);
  }
  ofile << std::endl;
}

void LogDefaultKnobs(std::string fname)
{ 
  std::ofstream ofile(fname.c_str());
  ofile << "//     Symsim Knob file     //" << std::endl;
  ofile << "//////////////////////////////" << std::endl;
  ofile << "// Generated by whoop for workload: " << options::kProgramName << std::endl;
  ofile <<  std::endl;
  ofile << "// Simulation knobs" << std::endl;
  ofile << "random_seed = 12345" << std::endl;
  ofile << "sys:freq_mhz = 1950" << std::endl;
  ofile << "global:debug = 2" << std::endl;
  ofile << "default:trace_level = 6" << std::endl;
  ofile << "trace_server:error_on_subscription_to_unknown_instance = false" << std::endl;
  ofile << std::endl;
  ofile << "physical_trace = " << options::kStatsFileName << ".physical.trc" << std::endl;
  ofile << "workload#0 = " << options::kStatsFileName << ".logical.trc" << std::endl;
  ofile << "sys:config_file = " << options::kStatsFileName << ".logical.yaml" << std::endl;
  ofile << "sys:physical_config_file = " << options::kPhysicalPath << "/" << options::kPhysicalFile << std::endl;
  ofile << "sys:binding_config_file = " << options::kStatsFileName << ".bindings.yaml" << std::endl;
  ofile << std::endl;
  ofile << "default_trace_reader_params:lat_cycles = 1" << std::endl;
  ofile << "default_trace_reader_params:bw_bytes_cycle = 128" << std::endl;
  ofile << "default_trace_reader_params:nactions_to_read = 2000" << std::endl;
  ofile << "" << std::endl;
  ofile << "default_buffet_params:actions_per_cycle = 100" << std::endl;
  ofile << "default_buffet_params:bw_bytes_cycle = 128" << std::endl;
  ofile << "default_buffet_params:max_actions_to_process = 10" << std::endl;
  ofile << "default_buffet_params:internal_read_output_channel:lat_cycles = 1" << std::endl;
  ofile << "default_buffet_params:internal_read_output_channel:bw_bytes_cycle = -1" << std::endl;
  ofile << "default_buffet_params:internal_read_output_channel:max_capacity_nbytes = -1" << std::endl;
  ofile << "default_buffet_params:internal_drain_output_channel:lat_cycles = 1" << std::endl;
  ofile << "default_buffet_params:internal_drain_output_channel:bw_bytes_cycle = -1" << std::endl;
  ofile << "default_buffet_params:internal_drain_output_channel:max_capacity_nbytes = -1" << std::endl;
  ofile << std::endl;
  ofile << "default_pattern_generator_params:actions_per_cycle = 100" << std::endl;
  ofile << "default_pattern_generator_params:bw_bytes_cycle = 128" << std::endl;
  ofile << "default_pattern_generator_params:max_actions_to_process = 10" << std::endl;
  ofile << "" << std::endl;
  ofile << "" << std::endl;
  ofile << "" << std::endl;
  ofile << "//Now the connections" << std::endl;
  ofile << "" << std::endl;
  ofile << "//default_connection_params:freq_mhz = 1950" << std::endl;
  ofile << "default_connection_params:element_size = 8" << std::endl;
  ofile << "default_connection_params:feedback_element_size = 8" << std::endl;
  ofile << "default_connection_params:ch_data:ch_freq_mhz = 1950" << std::endl;
  ofile << "default_connection_params:ch_data:lat_cycles = 1" << std::endl;
  ofile << "default_connection_params:ch_data:bw_bytes_cycle = 8" << std::endl;
  ofile << "default_connection_params:ch_data:max_capacity_nbytes =8" << std::endl;
  ofile << "default_connection_params:ch_feedback:ch_freq_mhz = 1950" << std::endl;
  ofile << "default_connection_params:ch_feedback:lat_cycles = 1" << std::endl;
  ofile << "default_connection_params:ch_feedback:bw_bytes_cycle = 8" << std::endl;
  ofile << "default_connection_params:ch_feedback:max_capacity_nbytes =8" << std::endl;
  ofile << "default_connection_params:output_feedback_internal_channel:lat_cycles = 1" << std::endl;
  ofile << "default_connection_params:output_feedback_internal_channel:bw_bytes_cycle = 128" << std::endl;
  ofile << "" << std::endl;
  ofile << "////  ---- DIRECT connections ---- ////" << std::endl;
  ofile << "" << std::endl;
  ofile << "//roundtrip latency assumptions:" << std::endl;
  ofile << "//16 clks CT-DOTC" << std::endl;
  ofile << "//40 clks CT-DOTML2" << std::endl;
  ofile << "//80 clks CT-Mem" << std::endl;
  ofile << "" << std::endl;
  ofile << "FMCT_CT_CONNECTION:lat_cycles = 1" << std::endl;
  ofile << "FMCT_CT_CONNECTION:bw_bytes_cycle = 128" << std::endl;
  ofile << "FMCT_CT_CONNECTION:support_multi_packet_issue = 1" << std::endl;
  ofile << "" << std::endl;
  ofile << "//16 clk routrip latency to DOTC" << std::endl;
  ofile << "CT_DOTC_CONNECTION:lat_cycles = 8" << std::endl;
  ofile << "CT_DOTC_CONNECTION:bw_bytes_cycle = 128" << std::endl;
  ofile << "CT_DOTC_CONNECTION:support_multi_packet_issue = 1" << std::endl;
  ofile << "" << std::endl;
  ofile << "DOTC_SW_CONNECTION:lat_cycles = 1" << std::endl;
  ofile << "DOTC_SW_CONNECTION:bw_bytes_cycle = 256" << std::endl;
  ofile << "DOTC_SW_CONNECTION:support_multi_packet_issue = 1" << std::endl;
  ofile << "" << std::endl;
  ofile << "//40 clk overal routrip latency to DOTC" << std::endl;
  ofile << "SW_DOTML2_CONNECTION:lat_cycles = 11" << std::endl;
  ofile << "SW_DOTML2_CONNECTION:bw_bytes_cycle = 512" << std::endl;
  ofile << "SW_DOTML2_CONNECTION:support_multi_packet_issue = 1" << std::endl;
  ofile << "" << std::endl;
  ofile << "DOTML2_DOTMDRAM_CONNECTION:lat_cycles = 1" << std::endl;
  ofile << "DOTML2_DOTMDRAM_CONNECTION:bw_bytes_cycle = 256" << std::endl;
  ofile << "DOTML2_DOTMDRAM_CONNECTION:support_multi_packet_issue = 1" << std::endl;
  ofile << "" << std::endl;
  ofile << "////  ---- BYPASS connections ---- ////" << std::endl;
  ofile << "" << std::endl;
  ofile << "SW_DOTMDRAM_CONNECTION:lat_cycles = 12" << std::endl;
  ofile << "SW_DOTMDRAM_CONNECTION:bw_bytes_cycle = 256" << std::endl;
  ofile << "SW_DOTMDRAM_CONNECTION:support_multi_packet_issue = 1" << std::endl;
  ofile << "" << std::endl;
  ofile << "CT_SW_CONNECTION:lat_cycles = 9" << std::endl;
  ofile << "CT_SW_CONNECTION:bw_bytes_cycle = 128" << std::endl;
  ofile << "CT_SW_CONNECTION:support_multi_packet_issue = 1" << std::endl;
  ofile << "" << std::endl;
  ofile << "DOTC_DOTML2_CONNECTION:lat_cycles = 12" << std::endl;
  ofile << "DOTC_DOTML2_CONNECTION:bw_bytes_cycle = 256" << std::endl;
  ofile << "DOTC_DOTML2_CONNECTION:support_multi_packet_issue = 1" << std::endl;

}

int NumSpatialPartitionsFlattened()
{
  int res = 1;
  for (auto it = spatial_partition_levels.begin(); it != spatial_partition_levels.end(); it++)
  {
    res *= (*it).second;
  }
  return res;
}

int ActualSpatialPartitionHeight(int count_ones)
{
  int res = 0;
  for (auto sp : spatial_partition_levels)
  {
    if (sp.first != NULL)
    {
      if (sp.second != 1 || count_ones)
      {
        res++;
      }
    }
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
  the_program.AddInitialStatements();
  user_tracer_ = {};
}

void Done()
{
    Done(options::kStatsFileName + ".stats");
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
    DisableIdleCompute();
    LogActivity(options::kStatsFileName + ".logical.trc");
    LogPhysicalActivity(options::kStatsFileName + ".physical.trc");
    LogTensorTopology(options::kStatsFileName + ".logical.yaml");
    LogTensorBindings(options::kStatsFileName + ".bindings.yaml");
    LogDefaultKnobs(options::kStatsFileName + ".knobs");
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
  the_program.AddEndStatements();

  user_tracer_.ASSERT(spatial_partition_levels.size() == 0) << "You have a missing end() call in the loop nest -- Please check!" << EndT;
  
  InitializeComputeLogs(compute_tile_levels);
  
  int max_flat_expansion = 1;
  for (auto level : spatial_partition_levels_sanity_check)
  {
    max_flat_expansion *= level;
  }
  
  for (auto var : all_vars)
  {
    var->InitializePartitioning(max_flat_expansion);
  }
  
  for (auto it = all_tensors.begin(); it != all_tensors.end(); it++)
  {
    (*it)->BindToDefaults();
    (*it)->FixupBufferLevelNumbering();
    (*it)->SetTraceLevel(options::kCurrentTraceLevel);
  }
  the_program.Run(max_flat_expansion);
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

void Program::AddInitialStatements()
{
  // Add an s_for = 0..1 around the whole program...
  // This keeps things much cleaner with spatial expansion.
  Var* v = new Var("__root");
  s_for(*v, 0, 1);
}

/* TODO: FIX
void BindCurrentComputeLevel(const BindingTarget& target, int expansion_factor)
{
  BindComputeLevel(current_global_tile_level, target, expansion_factor);
}
*/
}
