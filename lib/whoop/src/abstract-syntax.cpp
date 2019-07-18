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

std::vector<std::vector<std::unique_ptr<activity::ComputeEngineLog>>> compute_logs{};

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
}

void LogComputeActivity(std::ostream& ostr)
{
  for (int tile_level = 0; tile_level < compute_logs.size(); tile_level++)
  {
    for (int local_index = 0; local_index < compute_logs[tile_level].size(); local_index++)
    {
      compute_logs[tile_level][local_index]->Dump(ostr, "compute_engine_" + std::to_string(tile_level) + "_" + std::to_string(local_index), "symphony::modules::ComputeEngineComplex");

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
      ofile << "    base_name: compute_engine_" << tile_level << "_" << local_index << std::endl;
      ofile << "    configuration:" << std::endl;
      ofile << "      knobs_use_prefix: false" << std::endl;
      ofile << "      knobs:" << std::endl;
      ofile << "        - \"num_in = " << num_tensors << "\""  << std::endl;
      ofile << "        - \"num_out = " << num_tensors << "\""  << std::endl;
    }
  }
}

namespace buff
{
int GetBufIndex( int curr_spatial_idx, int buffs_at_level, int num_spatial_partitions )
{
  return ( curr_spatial_idx / (num_spatial_partitions/buffs_at_level) );
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
