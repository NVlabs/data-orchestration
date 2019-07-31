/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef WHOOP_ACTIVITY_HPP_
#define WHOOP_ACTIVITY_HPP_

#include <vector>
#include <unordered_map>
#include <list>
#include <map>
#include <stack>
#include <memory>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "options.hpp"

namespace whoop
{

namespace activity
{

class Log
{
 protected:
  //std::string name_;
  int indent_level_ = 0;
  
  void IncreaseIndent() { indent_level_++; }
  void DecreaseIndent() { indent_level_--; }
  std::string Indent()
  {
    std::string result = "";
    for (int x = 0; x < indent_level_; x++)
    {
      result += "  ";
    }
    return result;
  }
  
  void DumpCommands(std::ostream& ostr, boost::property_tree::ptree& cmds)
  {
    for (auto cmd_it =cmds.begin(); cmd_it != cmds.end(); cmd_it++)
    {
      ostr << Indent() << "{" << std::endl;
      IncreaseIndent();
      ostr << Indent() << "\"Action\": \"" << cmd_it->second.get<std::string>("Action") << "\"," << std::endl;
      ostr << Indent() << "\"Params\": {" << std::endl;
      IncreaseIndent();
      auto params = cmd_it->second.get_child("Params");
      for (auto param_it = params.begin(); param_it != params.end(); param_it++)
      {
        ostr << Indent() <<  "\"" << param_it->first << "\": " << param_it->second.data();
        if (param_it != std::prev(params.end()))
        {
          ostr << ",";
        }
        ostr << std::endl;
      }
      DecreaseIndent();
      ostr << Indent() << "}" << std::endl;
      DecreaseIndent();
      ostr << Indent() << "}";
      if (cmd_it != std::prev(cmds.end()))
      {
        ostr << ",";
      }
      ostr << std::endl;
    }
  }
  
 public:

  boost::property_tree::ptree commands_ = {};

  void AddInitialActivity(const boost::property_tree::ptree& cmd)
  {
     // Note: empty key = array in JSON
    commands_.push_front(boost::property_tree::ptree::value_type("", cmd));
  }
  
  void AddActivity(const boost::property_tree::ptree& cmd)
  {
     // Note: empty key = array in JSON
    commands_.push_back(boost::property_tree::ptree::value_type("", cmd));
  }
  
  virtual void Dump(std::ostream& ostr, const std::string& nm, const std::string& mytype)
  {
    //boost::property_tree::ptree instance;
    //instance.put("Name", nm);
    //instance.put("Class", mytype);
    //instance.put("Commands", commands_);
    //instance.push_back(boost::property_tree::ptree::value_type("Commands", commands_));
    IncreaseIndent();
    ostr << Indent() << "{" << std::endl;
    IncreaseIndent();
    ostr << Indent() << "\"Name\": \"" << nm << "\"," << std::endl;
    ostr << Indent() << "\"Class\": \"" << mytype << "\"," << std::endl;
    ostr << Indent() << "\"Commands\": [" << std::endl;
    IncreaseIndent();
    DumpCommands(ostr, commands_);
    DecreaseIndent();
    ostr << Indent() << "]" << std::endl;
    DecreaseIndent();
    ostr << Indent() << "}"; // << std::endl; omit final endl so someone can add a comma
    DecreaseIndent();
    //boost::property_tree::write_json(ostr, instance);
  }
};

class PatternGeneratorLog : public Log
{
 public:
  void Send(long int idx)
  {
    if (!options::kShouldLogActivity) return;
    boost::property_tree::ptree send_action;
    //send_action.put("UnitInstance", name_);
    //send_action.put("UnitClass", "SymphonyPatternGenerator");
    send_action.put("Action", "PatternGeneratorSendAction");
    send_action.put("Params.val_", idx);
    AddActivity(send_action);
  }
};

class ShrinkPatternGeneratorLog : public Log
{
  int shrinks_ = 0;
  int shrink_granularity_ = 1;
 public:
  void EmitShrink(int num)
  {
    if (!options::kShouldLogActivity) return;
    boost::property_tree::ptree send_action;
    //send_action.put("UnitInstance", name_);
    //send_action.put("UnitClass", "SymphonyPatternGenerator");
    send_action.put("Action", "PatternGeneratorSendAction");
    send_action.put("Params.val_", num);
    AddActivity(send_action);
  }
  void Shrink(int num = 1)
  {
    if (!options::kShouldLogActivity) return;
    shrinks_ += num;
    if (shrinks_ >= shrink_granularity_)
    {
      EmitShrink(shrinks_);
      shrinks_ = 0;
    }
  }
  
  void SetShrinkGranularity(int g)
  {
    shrink_granularity_ = g;
  }
};

class BuffetCommandLog : public Log
{
  int shrinks_ = 0;
  int shrink_granularity_ = 1;
  bool modified_ = false;
  int head_ = 0;
  int size_ = 0;
  bool is_filled_ = false;
  
 public:
  void Dump(std::ostream& ostr, const std::string& nm, const std::string& mytype)
  {
    // We wait to add the init command here because our
    // size may have changed from creation time and we want the final one.
    boost::property_tree::ptree init_action;
    //init_action.put("UnitInstance", name_);
    //init_action.put("UnitClass", "SymphonyBuffet");
    Log::Dump(ostr, nm, mytype);
  }
 
  void Init(int size, int extra = 0, bool is_filled = true)
  {
    size_ = size + extra;
    is_filled_ = is_filled;
  }
  
  void Resize(int size)
  {
    size_ = size;
  }
  
  void Read(bool will_update = false)
  {
    if (!options::kShouldLogActivity) return;
    boost::property_tree::ptree read_action;
    //read_action.put("UnitInstance", name_);
    //read_action.put("UnitClass", "SymphonyBuffet");
    read_action.put("Action", "BuffetReadAction");
    read_action.put("Params.num_", 1);
    read_action.put("Params.buffet_state_id_", 0);
    read_action.put("Params.will_update_", will_update);
    AddActivity(read_action);
  }
  
  void EmitShrink()
  {
    if (!options::kShouldLogActivity) return;
    boost::property_tree::ptree shrink_action;
    //shrink_action.put("UnitInstance", name_);
    //shrink_action.put("UnitClass", "SymphonyBuffet");
    shrink_action.put("Action", "BuffetShrinkAction");
    shrink_action.put("Params.num_", 1);
    shrink_action.put("Params.is_drain_", modified_);
    shrink_action.put("Params.buffet_state_id_", 0);
    AddActivity(shrink_action);
    modified_ = false;
  }
  
  void Shrink(int num = 1)
  {
    if (!options::kShouldLogActivity) return;
    shrinks_ += num;
    if (shrinks_ >= shrink_granularity_)
    {
      shrinks_ = 0;
      EmitShrink();
      head_ += shrink_granularity_;
      if (head_ > size_)
      {
        head_ -= size_;
      }
    }
  }
  
  void SetShrinkGranularity(int g)
  {
    shrink_granularity_ = g;
  }
  
  void SetModified()
  {
    modified_ = true;
  }
  
  int GetRelativeIndex(int abs_idx)
  {
    int rel_idx = -1;
    if (abs_idx >= head_)
    {
      rel_idx = abs_idx - head_;
    }
    else
    {
      rel_idx = size_ - head_ +  abs_idx;
    }
    return rel_idx;
  }
};


class ComputeEngineLog : public Log
{
 public:

  // Which tensors are we reading for the current op?
  std::bitset<64> read_activity_by_tensor_;
  
  PatternGeneratorLog sgen_log_;
  PatternGeneratorLog dgen_log_;

 public:
  void EmitOp(const std::bitset<64>& outputs)
  {
    if (!options::kShouldLogActivity) return;
    // Today all ops are the same. They take some number of inputs
    // and produce some number of outputs.
    sgen_log_.Send(read_activity_by_tensor_.to_ulong());
    dgen_log_.Send(outputs.to_ulong());
    read_activity_by_tensor_.reset();
  }
  void LogInputTensor(const int& id)
  {
    if (!options::kShouldLogActivity) return;
    if (read_activity_by_tensor_[id])
    {
      EmitOp(0); // No output
    }
    read_activity_by_tensor_[id] = true;
  }
  void LogOutputTensor(const std::bitset<64>& outputs)
  {
    if (!options::kShouldLogActivity) return;
    EmitOp(outputs);
  }
  
  void Dump(std::ostream& ostr, const std::string& nm, const std::string& mytype)
  {
    sgen_log_.Dump(ostr, nm + "_input_sources", "symphony::modules::LogicalGatedPatternGenerator");
    ostr << "," << std::endl;
    dgen_log_.Dump(ostr, nm + "_output_destinations", "symphony::modules::LogicalGatedPatternGenerator");
  }
};



}  // namespace activity

}  // namespace whoop

#endif /* WHOOP_ACTIVITY_HPP_ */
