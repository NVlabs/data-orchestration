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

#ifndef WHOOP_HPP_
#define WHOOP_HPP_


#include <iostream>
#include <fstream>
#include <deque>
#include <stack>
#include <memory>
#include <assert.h>

#include "stats.hpp"
#include "abstract-syntax.hpp"

extern int ARG_FLUSH_BUFFETS;

namespace whoop
{

template <typename T>
T RandomValue(int idx)
{
  T v = static_cast<T>(rand() % 255);
  return v;
}

class TreeBuilder;


// Macros for helping us quickly and concisely support many operators.


#define WHOOP_DECLARE_UNARY_OPERATOR(OP, RET) \
  RET operator OP(); \

#define WHOOP_DECLARE_BINARY_OPERATOR(OP, RET) \
  RET operator OP(TreeBuilder body_e); \
  RET operator OP(ast::PrimVar& v2); \
  RET operator OP(const int& c); \
  RET operator OP(const TensorDisambiguator& v); \

#define WHOOP_DECLARE_STMT_UNARY_OPERATOR(OP) \
  WHOOP_DECLARE_UNARY_OPERATOR(OP, void)  

#define WHOOP_DECLARE_STMT_BINARY_OPERATOR(OP) \
  WHOOP_DECLARE_BINARY_OPERATOR(OP, void)
  
#define WHOOP_DECLARE_EXPR_UNARY_OPERATOR(OP) \
  WHOOP_DECLARE_UNARY_OPERATOR(OP, TreeBuilder)

#define WHOOP_DECLARE_EXPR_BINARY_OPERATOR(OP) \
  WHOOP_DECLARE_BINARY_OPERATOR(OP, TreeBuilder)

#define WHOOP_DECLARE_CONST_BINARY_OPERATOR(OP, RET) \
  RET operator OP(const int& c, TreeBuilder body_e); \
  RET operator OP(const int& c, ast::PrimVar& body_e); \
  RET operator OP(const int& c, TensorDisambiguator& v); \

#define WHOOP_DECLARE_CONST_STMT_BINARY_OPERATOR(OP) \
  WHOOP_DECLARE_CONST_BINARY_OPERATOR(OP, void)

#define WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(OP) \
  WHOOP_DECLARE_CONST_BINARY_OPERATOR(OP, TreeBuilder)

#define WHOOP_DEFINE_STMT_BINARY_OPERATOR(OP, CONVERSION_OP) \
  void Container::operator OP(TreeBuilder body_e) \
  { \
    ast::Statement* assign_stmt = CONVERSION_OP; \
    the_program.Add(assign_stmt); \
  } \
  void Container::operator OP(ast::PrimVar& v2) \
  { \
    ast::VarAccess* body_e = new ast::VarAccess(v2); \
    this->operator OP(TreeBuilder(body_e)); \
  } \
  void Container::operator OP(const int& c) \
  { \
    ast::Constant* body_e = new ast::Constant(c); \
    this->operator OP(TreeBuilder(body_e)); \
  } \
  void Container::operator OP(const TensorDisambiguator& v) \
  { \
    ast::TensorAccess* body_e = v.ToTensorAccess(); \
    this->operator OP(TreeBuilder(body_e)); \
  } \


#define WHOOP_DEFINE_DERIVED_OPS(CLASSNAME) \
  void CLASSNAME::operator =(TreeBuilder body_e) \
  { \
    ast::Statement* assign_stmt = ToAssignment(body_e.expr_); \
    the_program.Add(assign_stmt); \
  } \
  void CLASSNAME::operator =(ast::PrimVar& v2) \
  { \
    ast::VarAccess* body_e = new ast::VarAccess(v2); \
    this->operator =(TreeBuilder(body_e)); \
  } \
  void CLASSNAME::operator =(const int& c) \
  { \
    ast::Constant* body_e = new ast::Constant(c); \
    this->operator =(TreeBuilder(body_e)); \
  } \
  void CLASSNAME::operator =(const TensorDisambiguator& v) \
  { \
    ast::TensorAccess* body_e = v.ToTensorAccess(); \
    this->operator =(TreeBuilder(body_e)); \
  } \

#define WHOOP_DEFINE_EXPR_BINARY_OPERATOR(OP, CONVERSION_OP) \
  TreeBuilder Container::operator OP(TreeBuilder body_e) \
  { \
    TreeBuilder expr(CONVERSION_OP); \
    return expr; \
  } \
  TreeBuilder Container::operator OP(ast::PrimVar& v2) \
  { \
    ast::VarAccess* body_e = new ast::VarAccess(v2); \
    return this->operator OP(TreeBuilder(body_e)); \
  } \
  TreeBuilder Container::operator OP(const int& c) \
  { \
    ast::Constant* body_e = new ast::Constant(c); \
    return this->operator OP(TreeBuilder(body_e)); \
  } \
  TreeBuilder Container::operator OP(const TensorDisambiguator& v) \
  { \
    ast::TensorAccess* body_e = v.ToTensorAccess(); \
    return this->operator OP(TreeBuilder(body_e)); \
  } \


#define WHOOP_DEFINE_CONST_STMT_BINARY_OPERATOR(OP, CONVERSION_OP) \
  void operator OP(const int& c, TreeBuilder body_e) \
  { \
    ast::Statement* assign_stmt = CONVERSION_OP; \
    the_program.Add(assign_stmt); \
  } \
  void operator OP(const int& c, ast::PrimVar& v2) \
  { \
    ast::VarAccess* body_e = new ast::VarAccess(v2); \
    operator OP(c, TreeBuilder(body_e)); \
  } \
  void operator OP(const int& c, const TensorDisambiguator& v) \
  { \
    ast::TensorAccess* body_e = v.ToTensorAccess(); \
    operator OP(c, TreeBuilder(body_e)); \
  } \


#define WHOOP_DEFINE_CONST_EXPR_BINARY_OPERATOR(OP, CONVERSION_OP) \
  TreeBuilder operator OP(const int& c, TreeBuilder body_e) \
  { \
    TreeBuilder expr(CONVERSION_OP); \
    return expr; \
  } \
  TreeBuilder operator OP(const int& c, ast::PrimVar& v2) \
  { \
    ast::VarAccess* body_e = new ast::VarAccess(v2); \
    return operator OP(c, TreeBuilder(body_e)); \
  } \
  TreeBuilder operator OP(const int& c, const TensorDisambiguator& v) \
  { \
    ast::TensorAccess* body_e = v.ToTensorAccess(); \
    return operator OP(c, TreeBuilder(body_e)); \
  } \



// spatial_partition_levels
// Global variable to hold the current partitioning.
// We use a stack so we can add to it as we traverse loops.
// (Implemented as a deque so we can get an iterator to it.)
extern std::deque<std::pair<ast::SpatialFor*, int>> spatial_partition_levels;

// tile level tracking
// Global variable to hold the current tile level for each tensor,
// as well as the current spatial_partition_level where that level was added.
// This info is used to pop this stack in end();
// Note: Tracks separate levels per tensor, per port.
extern std::vector<std::vector<std::deque<int>>> tile_level_deliminators;
extern std::vector<std::vector<int>> current_tile_level;
// Record the maximum value that has ever been present in current_tile_level.
extern int max_tile_level;
// Not all tensors add all tile levels. This increments/decrements globally.
extern std::deque<int> global_tile_level_deliminators;
extern std::vector<int> compute_tile_levels;
extern int current_global_tile_level;
extern bool need_global_tile_level;
// Track how many spatial tiles there are at each tile level.
extern std::vector<int> tile_level_spatial_expansions;

int NumSpatialPartitionsFlattened();
int ActualSpatialPartitionHeight(int count_ones = false);
void BindCurrentComputeLevel(const BindingTarget& target, int expansion_factor = 1);


class Tensor;
class Vec;
class Var;

class InFromFile
{
 public:
  virtual void ReadInput(bool is_ref = false) = 0;
};


class OutToFile
{
 public:
  virtual void DumpOutput() = 0;
  virtual void CheckOutput() = 0;
};

// need_input
// Global variable to list which tensors should read their input from a file.
extern std::list<InFromFile*> need_input;

// need_output
// Global variable to list which tensors should be output to disk at the end.
extern std::list<OutToFile*> need_output;

// all_tensors
// Global variable to list all tensors for miscellaneous ops.
extern std::list<Tensor*> all_tensors;
extern std::list<Var*> all_vars;

// user_tracer
// Global variable for user trace statements
extern UserTracer user_tracer_;
extern Flusher EndT;


class TensorDisambiguator;
class TensorPort;


void s_for(ast::PrimVar& v, const int& init_const, const int& end_const);
//void s_for(ast::PrimVar& v,  Var& init_const,  Var& end_const);

void t_for(ast::PrimVar& v, TreeBuilder init_expr, TreeBuilder end_expr);

void t_for(ast::PrimVar& v, const int& init_const, TreeBuilder end_expr);

void t_for(ast::PrimVar& v, TreeBuilder init_expr,   const int& end_const);
void t_for(ast::PrimVar& v, TreeBuilder init_const,  Var& end_const);

void t_for(ast::PrimVar& v, const int& init_const, const int& end_const);

void t_for(ast::PrimVar& v, const TensorDisambiguator& init_const, const TensorDisambiguator& end_const);

void t_for(ast::PrimVar& v,  Var& init_const,  Var& end_const);
void t_for(ast::PrimVar& v,  const int& init_const,  Var& end_const);

void w_while(TreeBuilder test_expr);

void w_while(const int& test_const);
    

void w_if(TreeBuilder test_expr);

void w_if(const int& test_const);

void w_if(TreeBuilder test_expr, double exec_prob);

void w_if(const int& test_const, double exec_prob);

void w_else();

void w_else_if(TreeBuilder test_expr);

void w_else_if(TreeBuilder test_const);

void end();

void DumpStats();

ast::Expression* ToAccess(const int& c);
ast::Statement* ToAssignment(const int& c, ast::Expression* body_e);
ast::Expression* ToBinaryOp(const int& c, int (*op)(const int& a, const int& b), ast::Expression* body_e);
ast::Statement* ToUpdateOp(const int& c, int (*op)(const int& a, const int& b), ast::Expression* body_e);

void Init(int argc, char** argv);

void Run();

void Done();
void Done(std::string);

Tracer& T(int l = 0);
Tracer& ASSERT(bool term);


// A "program" is the environment in which we build up the syntax tree
// for processing. 

class Program
{
 public:
  ast::Statement* beginning_stmt_ = NULL;
  ast::Statement* cur_stmt_ = NULL;
  bool cur_is_complete_ = false;
  std::stack<ast::Statement*> loop_stack_;
  std::string name_ = "";

  void AddInitialStatements();
  
  void AddEndStatements()
  {
    end();
  }
  
  // Add
  // Add a statement to the program.

  void Add(ast::Statement* stmt)
  {
    if (!beginning_stmt_)
    {
      beginning_stmt_ = stmt;
      cur_stmt_ = stmt;
    }
    else if (cur_is_complete_)
    {
      cur_stmt_->next_ = stmt;
    }
    else
    {
      cur_stmt_->inner_ = stmt;
    }
    for (int x = 0; x < loop_stack_.size(); x++)
    {
      stmt->PrependToName("  ");
    }
    cur_stmt_ = stmt;
    cur_is_complete_ = true;
  }
  

  // AddIncomplete
  // Add a statement that needs an "inner body", like a loop.
  
  void AddIncomplete(ast::Statement* stmt)
  {
    Add(stmt);
    loop_stack_.push(stmt);
    cur_is_complete_ = false;
  }

  // AddElse
  // Add a statement that is an alternative for an if.
  
  void AddElse(ast::Statement* stmt)
  {
    CompleteCurrent();
    for (int x = 0; x < loop_stack_.size(); x++)
    {
      stmt->PrependToName("  ");
    }
    cur_stmt_->else_ = stmt;
    cur_stmt_ = stmt;
    loop_stack_.push(stmt);
    cur_is_complete_ = false;
  }

  // CompleteCurrent
  // Say that the current "inner body" is done: e.g., an end-loop.
  
  void CompleteCurrent()
  {
    cur_stmt_ = loop_stack_.top();
    loop_stack_.pop();
  }


  // Run
  // Execute the program after the syntax tree has been constructed.
  
  void Run(int num_flat_partitions)
  {
    ast::ExecutionContext ctx(num_flat_partitions);
    beginning_stmt_->Init(ctx);
    ast::Statement* continuation = beginning_stmt_->Execute(ctx);
    // Keep executing until nothing is paused.
    while (continuation != NULL)
    {
      continuation = beginning_stmt_->Execute(ctx);
    }
  }
  
  void SetName(const std::string& nm)
  {
    name_ = nm;
  }
  
  std::string& GetName()
  {
    return name_;
  }
};


// the_program
// Global variable for the syntax tree we are constructing.
extern Program the_program;

// Container
// Base class for vars and tensors. Used to implement all C++ operators in
// one place. Sub-classes must define how to construct an
// accesss and assignment statement from their member vars.
// Once we have those two methods we can define all other operators here.
// Note that we use macros because 99% of the code for different operators
// is the same.

class Container
{
 public:
 
  // Sub-classes must define these.
  virtual ast::Expression* ToAccess() = 0;
  virtual ast::Statement* ToAssignment(ast::Expression* body_e) = 0;


  // For things like -
  ast::Expression* ToUnaryOp(int (*op)(const int& a))
  {
    ast::Expression* access_e = ToAccess();
    ast::UnaryOp* op_e = new ast::UnaryOp(op, access_e);
    return op_e;
  }

  // For things like *, +
  ast::Expression* ToBinaryOp(int (*op)(const int& a, const int& b), ast::Expression* body_e)
  {
    ast::Expression* access_e = ToAccess();
    ast::BinaryOp* op_e = new ast::BinaryOp(op, access_e, body_e);
    return op_e;
  }

  // For things like +=, *=
  ast::Statement* ToUpdateOp(int (*op)(const int& a, const int& b), ast::Expression* body_e)
  {
    ast::Expression* plus_e = ToBinaryOp(op, body_e);
    ast::Statement* assign_stmt = ToAssignment(plus_e);
    return assign_stmt;
  }

  // Actual supported operators.
  WHOOP_DECLARE_STMT_BINARY_OPERATOR(=)
  WHOOP_DECLARE_STMT_BINARY_OPERATOR(+=)
  WHOOP_DECLARE_STMT_BINARY_OPERATOR(-=)
  WHOOP_DECLARE_STMT_BINARY_OPERATOR(*=)
  WHOOP_DECLARE_STMT_BINARY_OPERATOR(/=)
  WHOOP_DECLARE_STMT_BINARY_OPERATOR(&=)
  WHOOP_DECLARE_STMT_BINARY_OPERATOR(|=)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(+)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(-)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(*)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(/)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(%)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(==)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(!=)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(>=)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(<=)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(>)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(<)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(&)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(|)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(&&)
  WHOOP_DECLARE_EXPR_BINARY_OPERATOR(||)
};


WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(+)
WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(-)
WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(*)
WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(/)
WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(%)
WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(==)
WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(!=)
WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(>=)
WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(<=)
WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(>)
WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(<)
WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(&&)
WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(||)
WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(&)
WHOOP_DECLARE_CONST_EXPR_BINARY_OPERATOR(|)


class TreeBuilder : public Container
{
 public:
  ast::Expression* expr_ = NULL;
  
  explicit TreeBuilder(ast::Expression* e) :
    expr_(e)
  {
  }

  virtual ast::Expression* ToAccess()
  {
    return expr_;
  }

  virtual ast::Statement* ToAssignment(ast::Expression* body_e)
  {
    assert(0);
    return NULL;
  }
  
  WHOOP_DECLARE_STMT_BINARY_OPERATOR(=)
};


class Var : public ast::PrimVar, public Container
{
 private:
   void Init()
   {
     all_vars.push_back(this);
   }

 public:
  Var() :
    ast::PrimVar()
  {
     Init();
  }

  ~Var()
  {
      all_vars.remove(this);
  }
  
  Var(const std::string& name) :
    ast::PrimVar(name)
  {
    Init();
  }
  
  operator ast::PrimVar()
  {
    return static_cast<ast::PrimVar>(*this);
  } 
  
  virtual ast::Expression* ToAccess()
  {
    ast::VarAccess* access_e = new ast::VarAccess(*this);
    return access_e;
  }

  virtual ast::Statement* ToAssignment(ast::Expression* body_e)
  {
    ast::VarAssignment* assign_stmt = new ast::VarAssignment(*this, body_e);
    return assign_stmt;
  }
  
  // We must declare this case explicitly.
  Var& operator =(Var& t)
  {
    ast::Expression* access_e = new ast::VarAccess(t);
    ast::VarAssignment* assign_stmt = new ast::VarAssignment(*this, access_e);
    the_program.Add(assign_stmt);
  }

  WHOOP_DECLARE_STMT_BINARY_OPERATOR(=)
};


// Class that distinguishes access/assign of tensors by [][][]
class TensorDisambiguator : public Container
{
 public:
  ast::PrimTensor& target_;
  std::list<ast::Expression*> idx_exprs_;
  int tile_level_ = 0;
  int compute_level_ = 0;
  int port_ = 0;

  TensorDisambiguator(ast::PrimTensor& v, const std::list<ast::Expression*>& e, int tile_level, int compute_level, int port = 0) : 
    target_(v), 
    idx_exprs_(e),
    tile_level_(tile_level),
    compute_level_(compute_level),
    port_(port)
  {
  }
 
  ast::TensorAccess* ToTensorAccess() const
  {
    ast::TensorAccess* access_e = new ast::TensorAccess(target_, idx_exprs_, tile_level_, compute_level_, port_);
    return access_e;
  }

  virtual ast::Expression* ToAccess()
  {
    return ToTensorAccess();
  }

  virtual ast::Statement* ToAssignment(ast::Expression* body_e)
  {
    ast::TensorAssignment* assign_stmt = new ast::TensorAssignment(target_, idx_exprs_, body_e, tile_level_, compute_level_, port_);
    return assign_stmt;
  }

  // Handle chaining [] for multi-dimensional tensors
  TensorDisambiguator operator[](TreeBuilder idx_expr)
  {
    idx_exprs_.push_back(idx_expr.expr_);
    return *this;
  }

  TensorDisambiguator operator[](Var& v2)
  {
    ast::VarAccess* body_e = new ast::VarAccess(v2);
    return this->operator[](TreeBuilder(body_e));
  }

  TensorDisambiguator operator[](const int& c)
  {
    ast::Constant* body_e = new ast::Constant(c);
    return this->operator[](TreeBuilder(body_e));
  }

  TensorDisambiguator operator[](const TensorDisambiguator& v2)
  {
    ast::TensorAccess* body_e = v2.ToTensorAccess();
    return this->operator[](TreeBuilder(body_e));
  }

  WHOOP_DECLARE_STMT_BINARY_OPERATOR(=)

};


class Tensor : public ast::PrimTensor
{
 private:
  void Init()
  {
    std::shared_ptr<buff::BufferModel> backing(new buff::BackingBufferModel(name_, vals_.size(), 1));
    std::shared_ptr<std::vector<std::shared_ptr<buff::BufferModel>>> 
        backing_vec(new std::vector<std::shared_ptr<buff::BufferModel>>(1, backing));
    // Set up tile-level tracking state.
    id_ = all_tensors.size();
    tile_level_deliminators.push_back(std::vector<std::deque<int>>());
    current_tile_level.push_back(std::vector<int>({0}));
    all_tensors.push_back(this);
    // Add port 0, and add the backing buffer to it.
    AddPort();
    buffer_levels_[0]->push_back(backing_vec);
  }

 public:

  int port_ = 0;
  int cur_port_ = 0;

  Tensor(const std::vector<int>& dim_sizes, int (*init_func)(const std::vector<int>& idxs), const std::string& nm = "") :
    ast::PrimTensor(dim_sizes, init_func, nm)
  {
    Init();
  }
  
  ~Tensor()
  {
        //
      all_tensors.remove(this);
  }
    

  Tensor(const std::vector<int>& dim_sizes, const int& init_val, const std::string& nm = "") :
    ast::PrimTensor(dim_sizes, init_val, nm)
  {
    Init();
  }
  
  explicit Tensor(const std::vector<int>& dim_sizes, const std::string& nm = "") :
    ast::PrimTensor(dim_sizes, nm)
  {
    Init();
  }
  
  explicit Tensor(const std::string& nm = "") :
    ast::PrimTensor(nm)
  {
    Init();
  }
  
  TensorDisambiguator operator[](TreeBuilder idx_expr)
  {
    TensorDisambiguator vd(*this, {idx_expr.expr_}, current_tile_level[id_][port_], current_global_tile_level, port_);
    return vd;
  }

  TensorDisambiguator operator[](Var& v2)
  {
    ast::VarAccess* body_e = new ast::VarAccess(v2);
    return this->operator[](TreeBuilder(body_e));
  }

  TensorDisambiguator operator[](const int& c)
  {
    ast::Constant* body_e = new ast::Constant(c);
    return this->operator[](TreeBuilder(body_e));
  }

  TensorDisambiguator operator[](const TensorDisambiguator& v2)
  {
    ast::TensorAccess* body_e = v2.ToTensorAccess();
    return this->operator[](TreeBuilder(body_e));
  }

  void SetBackingGranularity(int g)
  {
    assert(g > 0);
    (*buffer_levels_[0])[0]->at(0)->SetAccessGranularity(g);
  }

  void AddBufferLevel(int size)
  {
    user_tracer_.T(0) << "WARNING: AddBufferLevel is deprecated. Use AddTileLevel() instead." << EndT;
    AddTileLevel(size);
  }

  void AddTileLevel(int size, int shrink_granularity = 0, int access_granularity = 1, int extra_buffering = 0, int port = 0)
  {
      
    user_tracer_.ASSERT(size > 0) << "AddTileLevel(): size must be greater than 0." << EndT;
      
    int num_flat = NumSpatialPartitionsFlattened();
    
    std::shared_ptr<std::vector<std::shared_ptr<buff::BufferModel>>> 
        new_buffs(new std::vector<std::shared_ptr<buff::BufferModel>>(num_flat));

    int level = buffer_levels_[port]->size();
    int num_backing_stores = buffer_levels_[port]->back()->size();

    int backing_iteration_interval = num_flat / num_backing_stores;

    auto backing_it = buffer_levels_[port]->back()->begin();

    current_tile_level[id_][port]++;

    // The first AddTileLevel after one (or more) s_for increments the compute level.
    if (need_global_tile_level || current_tile_level[id_][port] > max_tile_level)
    {
      current_global_tile_level++;
      if (current_global_tile_level+1 > compute_tile_levels.size())
      {
        compute_tile_levels.push_back(compute_tile_levels.back());
      }
      global_tile_level_deliminators.push_back(spatial_partition_levels.size());
      need_global_tile_level = false;
    }
    
    // Record where this tile level should be removed from the stack of tiles.
    if (current_tile_level[id_][port] > max_tile_level)
    {
      max_tile_level = current_tile_level[id_][port];
      tile_level_spatial_expansions.push_back(1);
    }

    // Tell the backing stores they are done serving compute.
    for (auto& backing_store : (*buffer_levels_[port]->back()))
    {
      backing_store->ending_global_tile_level_ = current_global_tile_level;
    }
    
    tile_level_deliminators[id_][port].push_back(spatial_partition_levels.size());

    for (int x = 0; x < num_flat; x++)
    {
      std::string nm = name_;
      if (port != 0)
      {
        nm = nm + "_port_" + std::to_string(port);
      }
      if (num_flat != 1)
      {
        nm += "_" + std::to_string(x);
      }
      if (x != 0 && (x % backing_iteration_interval == 0))
      {
        backing_it++;
      }
      // My fill granularity is the previous guy's read granularity.
      int fill_granularity = (*backing_it)->access_granularity_;
      if (size % fill_granularity != 0)
      {
        size += (fill_granularity - (size % fill_granularity));
      }
      int local_idx = (*backing_it)->fronting_buffers_.size();
      std::shared_ptr<buff::BufferModel> new_buff(new buff::AssociativeBufferModel(size, level, current_global_tile_level, local_idx, nm, shrink_granularity, access_granularity, fill_granularity, extra_buffering));
      (*new_buffs)[x] = new_buff;
      (*new_buffs)[x]->backing_buffer_ = *backing_it;
      (*backing_it)->fronting_buffers_.push_back(new_buff);
    }
    buffer_levels_[port]->push_back(new_buffs);

    if( whoop::options::kShouldFlushTiles )
    {
        // Calling FlushBufferHandler --ajaleel
        std::cout<<"Setting Up Auto Flush For Buffet Levels of: "<<name_<<" Level: "<<level<<" flat: "<<num_flat<<std::endl;
        ast::Flush *stmt = new ast::Flush( this, level, new_buffs );
        the_program.Add(stmt);
    }

  }

  void BypassTileLevel(int granularity = 1, int extra_buffering = 0, int port = 0)
  {
    // TODO: This should be a No-Op
    AddTileLevel(granularity, granularity, granularity, extra_buffering, port);
  }
  
  int AddPort()
  {
    std::shared_ptr<std::deque<std::shared_ptr<std::vector<std::shared_ptr<buff::BufferModel>>>>> 
      new_deq(new std::deque<std::shared_ptr<std::vector<std::shared_ptr<buff::BufferModel>>>>());
    if (buffer_levels_.size() > 0)
    {
      // the new port should share the same off-chip buffering.
      new_deq->push_back((*buffer_levels_[0])[0]);
    }
    buffer_levels_.push_back(new_deq);
    // The new port starts over at tile level zero.
    tile_level_deliminators[id_].push_back(std::deque<int>());
    current_tile_level[id_].push_back(0);
    return buffer_levels_.size() - 1;
  }
  
  void BindTile(int tile_level, int spatial_idx, const BindingTarget& target, int port = 0)
  {
    (*(*buffer_levels_[port])[tile_level])[spatial_idx]->Bind(target);
  };
  
  void BindTileLevel(int tile_level, const BindingTarget& target, int expansion_factor = 1, int port = 0)
  {
    int logical_level_size = (*buffer_levels_[port])[tile_level]->size();
    int tiles_per_target = std::max(logical_level_size / expansion_factor, 1);
    int cur_target_idx = std::min(target.GetIndex(), logical_level_size - 1);
    for (int x = 0; x < logical_level_size; x++)
    {
      BindTile(tile_level, x, {target.GetName(), cur_target_idx}, port);
      if ((x+1) % tiles_per_target == 0)
      {
        // Handle the remainder if it doesn't divide evenly. Just put them all on the final.
        if (cur_target_idx < expansion_factor - 1)
        {
          cur_target_idx++;
        }
      }
    }
  };

  void BindCurrentTileLevel(const BindingTarget& target, int expansion_factor = 1, int port = 0)
  {
    BindTileLevel(current_tile_level[id_][port], target, expansion_factor, port);
  };

  void BindBacking(const BindingTarget& target)
  {
    BindTileLevel(0, target, 1);
  }

};


class TensorIn : public Tensor, public InFromFile
{
 public:
 
  std::string filename_;
 
  TensorIn(const std::string& nm) :
    Tensor(nm)
  {
    filename_ = nm + ".in.txt";
    AddOption(&filename_, "tensor_" + nm + "_file", "Input datafile name for TensorIn " + nm);
    need_input.push_back(this);
  }
  
  void ReadInput(bool is_ref = false)
  {
    std::ifstream ifs;
    ifs.open(filename_);
    if (ifs.fail())
    {
      if (is_ref)
      {
        std::cerr << "Error: TensorOut " << name_ << " when attempting to read reference data from file: " << filename_ << std::endl;
        std::cerr << "Error: " << strerror(errno) << std::endl;
        std::cerr << "Note: file name can be set with option --ref_" << name_ << "_file=<filename>" << std::endl;
      }
      else
      {
        std::cerr << "Error: TensorIn " << name_ << " when attempting to read data from file: " << filename_ << std::endl;
        std::cerr << "Error: " << strerror(errno) << std::endl;
        std::cerr << "Note: file name can be set with option --tensor_" << name_ << "_file=<filename>" << std::endl;
      }
      std::exit(1);
    }
    boost::archive::text_iarchive ia(ifs);
    ia >> (*this);
    FixupSize();
  }
};


class TensorOut : public Tensor, public OutToFile, public Traceable
{
 private:
  void Init()
  {
    filename_ = Traceable::name_ + ".out.txt";
    ref_filename_ = Traceable::name_ + ".ref.txt";

    AddOption(&filename_, "tensor_" + Traceable::name_ + "_file", "Output datafile name for TensorOut " + Traceable::name_);
    AddOption(&ref_filename_, "ref_" + Traceable::name_ + "_file", "Reference datafile name for TensorOut " + Traceable::name_);
    
    need_output.push_back(this);
  }
 public:
  std::string filename_;
  std::string ref_filename_;
  
  explicit TensorOut(const std::string& nm) :
    Tensor(nm), Traceable(nm)
  {
    Init();
  }
  
  TensorOut(const std::vector<int>& dim_sizes, int (*init_func)(const std::vector<int>& idxs), const std::string& nm) :
    Tensor(dim_sizes, init_func, nm), Traceable(nm)
  {
    Init();
  }
  
  TensorOut(const std::vector<int>& dim_sizes, const int& init_val, const std::string& nm) :
    Tensor(dim_sizes, init_val, nm), Traceable(nm)
  {
    Init();
  }
  
  explicit TensorOut(const std::vector<int>& dim_sizes, const std::string& nm = "") :
    Tensor(dim_sizes, nm), Traceable(nm)
  {
    Init();
  }
  
  void DumpOutput()
  {
    std::ofstream ofs(filename_);
    boost::archive::text_oarchive oa(ofs);
    oa << (*this);
  }
  
  void CheckOutput()
  {
    TensorIn ref_in{Traceable::name_};
    ref_in.filename_ = ref_filename_;
    ref_in.ReadInput(true);
    
    user_tracer_.T(1) << "Beginning to check TensorOut " << Traceable::name_ << " against reference data." << EndT;
    
    user_tracer_.ASSERT_WARNING(ref_in.dim_sizes_.size() == dim_sizes_.size()) << "Number of dimensions mismatch versus reference tensor. Ref: " << ref_in.dim_sizes_.size() << ", This: " << ref_in.dim_sizes_.size() << EndT;

    auto dim_it = dim_sizes_.begin();
    int dim = 0;
    for (auto ref_dim_it = ref_in.dim_sizes_.begin(); ref_dim_it != ref_in.dim_sizes_.end(); ref_dim_it++)
    {
      user_tracer_.ASSERT_WARNING((*dim_it) == (*ref_dim_it)) << "Size mismatch versus reference tensor in dimension: " << ref_in.dim_sizes_.size() - dim << ". Ref: " << ref_in.dim_sizes_.size() << ", This: " << (*dim_it) << EndT;
      dim_it++;
      dim++;
    }

    for (int x = 0; x < ref_in.vals_.size(); x++)
    {
      int ref_v = static_cast<PrimTensor>(ref_in).PrimAt(x);
      int my_v = PrimTensor::PrimAt(x);
      // This construction looks weird but is faster in the common case of a match.
      if (my_v != ref_v)
      {
        std::vector<int> full_idx = UnflattenIndex(x);
        user_tracer_.ASSERT_WARNING(false) << "Data mismatch versus reference tensor, index: " << ast::ShowIndices(full_idx) << ". Ref: " << ref_v << ", This: " << my_v << EndT;
      }
    }
    
    user_tracer_.T(0) << "Reference check done." << EndT;
  }
};


class TensorPort
{
 public:

  int port_ = 0;
  Tensor* target_;

  explicit TensorPort(Tensor* t) :
    target_(t), port_(t->AddPort())
  {
  }
  
  TensorDisambiguator operator[](TreeBuilder idx_expr)
  {
    TensorDisambiguator vd(*target_, {idx_expr.expr_}, current_tile_level[target_->id_][port_], current_global_tile_level, port_);
    return vd;
  }

  TensorDisambiguator operator[](Var& v2)
  {
    ast::VarAccess* body_e = new ast::VarAccess(v2);
    return this->operator[](TreeBuilder(body_e));
  }

  TensorDisambiguator operator[](const int& c)
  {
    ast::Constant* body_e = new ast::Constant(c);
    return this->operator[](TreeBuilder(body_e));
  }

  TensorDisambiguator operator[](const TensorDisambiguator& v2)
  {
    ast::TensorAccess* body_e = v2.ToTensorAccess();
    return this->operator[](TreeBuilder(body_e));
  }

  void AddTileLevel(int size, int shrink_granularity = 0, int granularity = 1, int extra_buffering = 0)
  {
    target_->AddTileLevel(size, shrink_granularity, granularity, extra_buffering, port_);
  }

  void BypassTileLevel(int granularity = 1)
  {
    target_->BypassTileLevel(granularity, port_);
  }
  
  void BindTile(int tile_level, int spatial_idx, const BindingTarget& target)
  {
    target_->BindTile(tile_level, spatial_idx, target, port_);
  };
  
  void BindTileLevel(int tile_level, const BindingTarget& target, int expansion_factor = 1, int port = 0)
  {
    target_->BindTileLevel(tile_level, target, expansion_factor, port_);
  };

  void BindCurrentTileLevel(const BindingTarget& target, int expansion_factor = 1, int port = 0)
  {
    target_->BindCurrentTileLevel(target, expansion_factor, port_);
  };

  void BindBacking(const BindingTarget& target)
  {
    ASSERT(false) << "Binding the backing store should be done on the main tensor, not ports." << EndT;
  }
};

class Vec : public Tensor
{

 public: 
  
  Vec(const int& size, const std::string& nm = "") :
    Tensor({size}, nm)
  {
  }
  
  explicit Vec(const std::string& nm = "") :
    Tensor(nm)
  {
  }

  Vec(const int& size, const int& init_val, const std::string& nm = "") :
    Tensor({size}, init_val, nm)
  {
  }

  Vec(const int& size, int (*init_func)(const std::vector<int>& idxs), const std::string& nm = "") :
    Tensor({size}, init_func, nm)
  {
  }
  
  
  
  int Size()
  {
    return FlattenSizes(dim_sizes_);
  }
  
  void Resize(int new_size)
  {
    Tensor::Resize({new_size});
  }

  void PushBack(const int& val)
  {
    PrimPushBack(val, 0);
  }
  
  int& At(const int& idx)
  {
    return PrimAt(idx);
  }

  const int& At(const int& idx) const
  {
    return PrimAt(idx);
  }

};


class VecIn : public Vec, public InFromFile
{
 public:
 
  std::string filename_;
 
  VecIn(const std::string& nm) :
    Vec(nm)
  {
    filename_ = nm + ".in.txt";
    AddOption(&filename_, "vec_" + nm + "_file", "Input datafile name for VecIn " + nm);
    need_input.push_back(this);
  }

  std::string GetFileName() { return filename_; }//
  
  void ReadInput(bool is_ref = false)
  {
    std::ifstream ifs;
    ifs.open(filename_);
    if (ifs.fail())
    {
      if (is_ref)
      {
        std::cerr << "Error: VecOut " << name_ << " when attempting to read reference data from file: " << filename_ << std::endl;
        std::cerr << "Error: " << strerror(errno) << std::endl;
        std::cerr << "Note: file name can be set with option --ref_" << name_ << "_file=<filename>" << std::endl;
      }
      else
      {
        std::cerr << "Error: VecIn " << name_ << " when attempting to read data from file: " << filename_ << std::endl;
        std::cerr << "Error: " << strerror(errno) << std::endl;
        std::cerr << "Note: file name can be set with option --vec_" << name_ << "_file=<filename>" << std::endl;
      }
      std::exit(1);
    }
    boost::archive::text_iarchive ia(ifs);
    ia >> (*this);
    FixupSize();
  }
};

class VecOut : public Vec, public OutToFile, public Traceable
{
 private:
  void Init()
  {
    filename_ = Traceable::name_ + ".out.txt";
    ref_filename_ = Traceable::name_ + ".ref.txt";

    AddOption(&filename_, "vec_" + Traceable::name_ + "_file", "Output datafile name for VecOut " + Traceable::name_);
    AddOption(&ref_filename_, "ref_" + Traceable::name_ + "_file", "Reference datafile name for VecOut " + Traceable::name_);
    
    need_output.push_back(this);
  }
 public:
  std::string filename_;
  std::string ref_filename_;
  
  explicit VecOut(const std::string& nm) :
    Vec(nm), Traceable(nm)
  {
    Init();
  }
  
  std::string GetFileName() { return filename_; }//

  void DumpOutput()
  {
    std::ofstream ofs(filename_);
    boost::archive::text_oarchive oa(ofs);
    oa << (*this);
  }
  
  void CheckOutput()
  {
    VecIn ref_in{Traceable::name_+"_ref_check"};
    ref_in.filename_ = ref_filename_;
    ref_in.ReadInput(true);
    int ref_size = ref_in.Size();
    user_tracer_.T(1) << "Beginning to check VecOut " << Traceable::name_ << " against reference data." << EndT;
    
    user_tracer_.ASSERT_WARNING(ref_size == vals_.size()) << "Size mismatch versus reference vector. Ref: " << ref_size << ", This: " << vals_.size() << EndT;
    for (int x = 0; x < ref_in.Size(); x++)
    {
      int ref_v = static_cast<PrimTensor>(ref_in).PrimAt(x);
      int my_v = PrimTensor::PrimAt(x);
      // This construction looks weird but is faster in the common case of a match.
      if (my_v != ref_v)
      {
        auto full_idx = UnflattenIndex(x);
        user_tracer_.ASSERT_WARNING(false) << "Data mismatch versus reference tensor, index: " << ast::ShowIndices(full_idx) << ". Ref: " << ref_v << ", This: " << my_v << EndT;
      }
    }

    user_tracer_.T(0) << "Reference check done." << EndT;
  }
};

/*
class CompressedTensor
{
 protected:
  std::vector<PrimTensor> segments_;
  std::vector<PrimTensor> coordinates_;
  PrimTensor values_;
 public:
  Scanner GetScanner()
  {
    Scanner sc(this);
    return sc;
  }
};

using std::vector<int> = Point;
using std::pair<Point, int> = PointAndValue;
// An order is just a permutation of dimensions.
using Point = Order;

class DomainIterator
{
 protected:
  Scanner* scanner_ = NULL;
  PointAndValue current_;

 public:
  bool operator !=(const DomainIterator& other)
  {
    // Purposely ignore other, it's a dummy.
    return !scanner_->IsDone();
  }
  
  void operator ++()
  {
    current_ = scanner_->Advance();
  }
  
  PointAndValue operator *()
  {
    return current_;
  }
};


class Scanner
{
 protected:
  CompressedTensor* target_;
  DomainIterator scan_end_;
  std::vector<Point> tile_sizes_;
  std::vector<int> tile_repeats_;
  Order order_;
  Point current_;

 public:
  
  Scanner(CompressedTensor* target) :
    target_(target),
    current_(target->NumDimensions())
  {
  }
  
  void SetTiling(const std::vector<Point>& tile_sizes)
  {
    // TODO: sanity checks
    tile_sizes_ = tile_sizes;
  }

  void SetRepeat(const std::vector<int>& tile_repeats)
  {
    // TODO: sanity checks
    tile_repeats_ =  tile_repeats;
  }

  void SetOrder(const Order& order)
  {
    // TODO: sanity checks
    order_ = order;
  }

  DomainIterator begin()
  {
     DomainIterator dit(this);
     return dit;
  }

  DomainIterator end()
  {
    // unused dummy
    DomainIterator dit(NULL);
    return dit;
  }
  
  PointAndValue Advance()
  {
    for (int dim = 0; dim < target_->GetNumDimensions(); dim++)
    {
      cur_pos_[dim]++;
      if (cur_pos_[dim] != cur_bound_[dim])
      {
        break; // If the inner dim wasn't done, the outer dims won't be done either.
      }
      cur_point_[dim] = target_->GetCoordinate(dim, cur_pos_[dim])
      cur_bound_[dim] = target_->GetSegment(dim, curr
    }
    if (++cur_pos_ == curr_bound_[0])
    {
      // We reached the end of this dimension.
      // Increment all dimension coordinates until one is not done.
      
      curr_bound_[0] = end - start;
      
    }
    else
    {
      // More to do in this fiber.
      current_[0] = target_->GetCoordinate(0, current_pos_);
    }
  }
};
*/

}  // end namespace: whoop

#endif /* WHOOP_HPP_ */

