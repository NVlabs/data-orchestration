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
#include <vector>
#include <list>
#include <iostream>
#include <sstream>

#include "abstract-syntax.hpp"

namespace whoop
{

// TODO: Move these to a better spot and allow command-line over-rides.
// Note: index_ in this represents the number at that level
std::vector<BindingTarget> default_bindings = 
  {
    {"DOTMDRAM", 1},
    {"DOTML2", 1},
    {"DOTC", 2},
    {"CT", 8},
    {"FMCT", 8}
  };

std::vector<std::vector<std::unique_ptr<activity::ComputeEngineLog>>> compute_logs{};
std::vector<std::vector<BindingTarget>> compute_bindings{};
std::multimap<BindingTarget, std::string> physical_compute_map{};
std::multimap<BindingTarget, std::string> physical_buffer_map{};

int BindingTarget::GetLevel()
{
  for (int level = 0; level != default_bindings.size(); level++)
  {
    if (default_bindings[level].GetName() == name_) return level;
  }
  return -1;
}

int BindingTarget::GetExpansionFactor()
{
  int level = GetLevel();
  if (level == default_bindings.size() - 1) return 0;
  return std::max(default_bindings[level+1].GetIndex() / default_bindings[level].GetIndex(), 1);
}

void InitializeComputeLogs(const std::vector<int>& flattened_tile_level_spatial_expansions)
{
  compute_logs.resize(flattened_tile_level_spatial_expansions.size());
  for (int x = 0; x < flattened_tile_level_spatial_expansions.size(); x++)
  {
    compute_logs[x].resize(flattened_tile_level_spatial_expansions[x]);
    for (int y = 0; y < flattened_tile_level_spatial_expansions[x]; y++)
    {
      compute_logs[x][y] = std::unique_ptr<activity::ComputeEngineLog>(new activity::ComputeEngineLog());
    }
  }

  // Bind any unbound compute logs to their default location, if any.

  // Call resize to ensure everything with no default is marked Unbound.
  compute_bindings.resize(compute_logs.size());
  for (int x = 0; x < compute_logs.size(); x++)
  {
    compute_bindings[x].resize(compute_logs[x].size());
  }
  
  // Bind all the ones that are not already bound and have a default provided.
  for (int tile_level = 0; tile_level < compute_logs.size(); tile_level++)
  {
    for (int local_index = 0; local_index < compute_logs[tile_level].size(); local_index++)
    {
      if (compute_bindings[tile_level][local_index].IsUnbound())
      {
        BindCompute(tile_level, local_index, GetDefaultBinding(tile_level, local_index, compute_logs[tile_level].size()));
      }
    }
  }
}

void LogComputeActivity(std::ostream& ostr)
{
  for (int tile_level = 0; tile_level < compute_logs.size(); tile_level++)
  {
    for (int local_index = 0; local_index < compute_logs[tile_level].size(); local_index++)
    {
      compute_logs[tile_level][local_index]->Dump(ostr, GetComputeBoundName(tile_level, local_index), "symphony::modules::ComputeEngineComplex");

      if (!(tile_level == compute_logs.size() - 1 &&
            local_index == compute_logs[tile_level].size() - 1))
      {
        ostr << ",";
      }
      ostr << std::endl;
    }
  }
}

void LogComputeTopology(std::ostream& ofile, int num_tensors)
{
  for (int tile_level = 0; tile_level < compute_logs.size(); tile_level++)
  {
    for (int local_index = 0; local_index < compute_logs[tile_level].size(); local_index++)
    {
      ofile << "  - type: module" << std::endl;
      ofile << "    class: symphony::modules::ComputeEngineComplex" << std::endl;
      ofile << "    base_name: " << GetComputeBoundName(tile_level, local_index) << std::endl;
      ofile << "    configuration:" << std::endl;
      ofile << "      knobs_use_prefix: false" << std::endl;
      ofile << "      knobs:" << std::endl;
      ofile << "        - \"num_in = " << num_tensors << "\""  << std::endl;
      ofile << "        - \"num_out = " << num_tensors << "\""  << std::endl;
    }
  }
}


void BindCompute(int level, int spatial_idx, const BindingTarget& target)
{
  // This is called before whoop::Run so we can't rely on these being pre-initialized.
  if (compute_bindings.size() <= level)
  {
    compute_bindings.resize(level + 1);
  }
  if (compute_bindings.at(level).size() <= spatial_idx)
  {
    compute_bindings.at(level).resize(spatial_idx + 1);
  }
  assert(compute_bindings.at(level).at(spatial_idx).IsUnbound());
  compute_bindings.at(level).at(spatial_idx) = target;
  AddPhysicalComputeMap(target, GetComputeBoundName(level, spatial_idx));
}

void BindComputeLevel(int level, const BindingTarget& target, int expansion_factor)
{
  int logical_level_size = compute_bindings.at(level).size();
  int tiles_per_target = std::max(expansion_factor / logical_level_size, 1);
  int cur_target_idx = std::min(target.GetIndex(), logical_level_size - 1);
  for (int x = 0; x < logical_level_size; x++)
  {
    BindCompute(level, x, {target.GetName(), cur_target_idx});
    if ((x+1) % tiles_per_target == 0)
    {
      // Handle the remainder if it doesn't divide evenly. Just put them all on the final.
      if (cur_target_idx < expansion_factor - 1)
      {
        cur_target_idx++;
      }
    }
  }
}


BindingTarget GetDefaultBinding(int level, int spatial_idx, int expansion_factor)
{
  if (level >= default_bindings.size())
  {
    // Default is unbound.
    return BindingTarget();
  }
  int final_idx = buff::GetBufIndex(spatial_idx, default_bindings[level].GetIndex(), expansion_factor);
  return BindingTarget(default_bindings[level].GetName(), final_idx);
}

std::string GetComputeBoundName(int tile_level, int dp_idx)
{
  return compute_bindings[tile_level][dp_idx].ToString() + "_compute_engine_" + std::to_string(tile_level) + "_" + std::to_string(dp_idx);
}

void AddPhysicalComputeMap(const BindingTarget& target, const std::string& logical)
{
  physical_compute_map.insert(std::pair<BindingTarget, std::string>(target, logical));
}

void AddPhysicalBufferMap(const BindingTarget& target, const std::string& logical)
{
  physical_buffer_map.insert(std::pair<BindingTarget, std::string>(target, logical));
}

void LogPhysicalMap(std::ostream& ostr, bool is_compute, std::multimap<BindingTarget, std::string>& physical_map)
{
  std::string old_phys_name;
  int binding_id = 0;
  for (auto phys_it = physical_map.begin(); phys_it != physical_map.end(); phys_it++)
  {
    std::string phys_nm = (*phys_it).first.ToString();
    std::string phys_class;
    if (is_compute)
    {
      phys_nm += "-CECC";
      phys_class = "symphony::modules::ComputeEngineComplexCollection";
    }
    else
    {
      phys_nm += "-BCC";
      phys_class = "symphony::modules::BuffetComplexCollection";
    }
    if (phys_nm != old_phys_name)
    {
      ostr << " {" << std::endl;
      ostr << "  \"Name\":\"" << phys_nm << "\"," << std::endl;
      ostr << "  \"Class\":\"" << phys_class << "\"," << std::endl; 
      ostr << "  \"Commands\":[" << std::endl;
      binding_id = 0;
      old_phys_name = phys_nm;
    }
    ostr << "    {" << std::endl;
    ostr << "    \"Action\":\"ComponentCollectionCreateComponentAction\"," << std::endl;
    ostr << "    \"Params\": {" << std::endl;
    ostr << "                \"logical_id_\": " << binding_id << "," << std::endl;
    ostr << "                \"name_\": \"" << (*phys_it).second << "\"" << std::endl;
    ostr << "              }" << std::endl;
    ostr << "    }";
    if (binding_id != physical_map.count((*phys_it).first) - 1)
    {
     ostr << "," << std::endl;
    }
    else
    {
      ostr << std::endl;
      ostr << "  ]" << std::endl;
      ostr << " }";
      if (phys_it != std::prev(physical_map.end()))
      {
        ostr << ",";
      }
      ostr << std::endl;
    }
    binding_id++;
  }
}

namespace buff
{
int GetBufIndex( int curr_spatial_idx, int buffs_at_level, int num_spatial_partitions )
{
  if (num_spatial_partitions < buffs_at_level) 
  {
    return curr_spatial_idx;
  }
  return std::min( curr_spatial_idx / (num_spatial_partitions/buffs_at_level),  buffs_at_level - 1);
}
    
} // end namespace buff
    

namespace ast
{

std::string ShowIndices(const std::vector<int>& idxs)
{
  std::ostringstream ostr;
  for (auto it = idxs.rbegin(); it != idxs.rend(); it++)
  {
    ostr << "[";
    ostr << std::to_string(*it);
    ostr << "]";
  }
  return ostr.str();
}

std::list<int> EvaluateAll(const std::list<Expression*>& exprs, ExecutionContext& ctx)
{
  std::list<int> res;
  for (auto it = exprs.begin(); it != exprs.end(); it++)
  {
    res.push_back((*it)->Evaluate(ctx));
  }
  return res;
}

}  // end namespace: ast

}  // end namespace: whoop
