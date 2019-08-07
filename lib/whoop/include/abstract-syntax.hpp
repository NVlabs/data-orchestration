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

#ifndef WHOOP_ABSTRACT_SYNTAX_HPP_
#define WHOOP_ABSTRACT_SYNTAX_HPP_

#include <vector>
#include <unordered_map>
#include <list>
#include <map>
#include <stack>
#include <memory>
#include <bitset>
#include <numeric>

#include <boost/any.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>
#include <boost/dynamic_bitset.hpp>

#include "options.hpp"
#include "trace.hpp"
#include "stats.hpp"
#include "activity.hpp"
#include "operator-semantics.hpp"

/* timewhoop */
#include "pure-abstract-syntax-types.hpp"
#include "pure-abstract-syntax.hpp"

typedef unsigned long UINT64;

namespace whoop
{

class BindingTarget
{
 public:
  inline static const std::string kUnbound = "__UNBOUND__";
  inline static const std::string kDisabled = "__DISABLED__";
  
  std::string name_ = kUnbound;
  int idx_ = 0;
  
  BindingTarget() = default;
  // Note: purposely not explict.
  BindingTarget(const std::string& nm) : name_(nm) {}
  BindingTarget(const char* nm) : name_(nm) {}
  BindingTarget(const std::string& nm, int idx) : name_(nm), idx_(idx) {}
  BindingTarget(const char* nm, int idx) : name_(nm), idx_(idx) {}
  
  std::string GetName() const { return name_; }
  int GetIndex() const { return idx_; }
  
  bool IsUnbound() const { return name_ == kUnbound; }
  bool IsDisabled() const { return name_ == kDisabled; }
  void Disable() { name_ = kDisabled; }
  std::string ToString() const { return name_ + "_" + std::to_string(idx_); }
  
  // Used by std::map
  bool operator < (const BindingTarget& other) const
  {
    if (name_ < other.name_) return true;
    if (name_ == other.name_) return idx_ < other.idx_;
    return false;
  }

  bool operator == (const BindingTarget& other) const
  {
    return name_ == other.name_ && idx_ == other.idx_;
  }
  
  int GetLevel() const;
  int GetExpansionFactor() const;

};


// top_stats
// Global variable to serve as top-of-tree for stats dump
extern StatsCollection top_stats;

extern std::vector<std::vector<std::unique_ptr<activity::ComputeEngineLog>>> compute_logs;
extern std::vector<std::vector<BindingTarget>> compute_bindings;
extern std::multimap<BindingTarget, std::string> physical_compute_map;
extern std::multimap<BindingTarget, std::string> physical_buffer_map;

void InitializeComputeLogs(const std::vector<int>& flattened_tile_level_spatial_expansions);
void LogComputeActivity(std::ostream& ostr);
void LogComputeTopology(std::ostream& ofile, int num_tensors);
void DisableIdleCompute();
void InitializeExpansions(const std::vector<int>& flattened_tile_level_spatial_expansions);
void BindCompute(int level, int spatial_idx, const BindingTarget& target);
//TODO: FIX void BindComputeLevel(int level, const BindingTarget& target, int expansion_factor = 1);
BindingTarget GetDefaultBinding(int level, int spatial_idx, int expansion_factor = 1);
std::string GetComputeBoundName(int tile_level, int dp_idx);
void AddPhysicalComputeMap(const BindingTarget& target, const std::string& logical);
void AddPhysicalBufferMap(const BindingTarget& target, const std::string& logical);
void LogPhysicalMap(std::ostream& ostr, bool is_compute, std::multimap<BindingTarget, std::string>& phsyical_map);
int GetPhysicalIndex(const BindingTarget& src, const BindingTarget& dst);
void SetComputeWidth(int level, int spatial_idx, int granularity);

namespace buff
{

int GetBufIndex( int curr_spatial_idx, int buffs_at_level, int num_spatial_partitions );

class TraceableBuffer : public Traceable
{
 public:
  TraceableBuffer(const std::string& nm = "") :
    Traceable(nm)
  {
    if (options::kShouldTraceBuffers)
    {
      SetTraceLevel(options::kCurrentTraceLevel);
    }
    else
    {
      SetTraceLevel(0);
    }
  }
};

class BufferModel : public StatsCollection, public TraceableBuffer
{
 protected:
  class BuffetLogInfo
  {
   public:
    int index_ = -1;
    std::bitset<64> receiver_mask_ = 0;
    std::bitset<64> updater_mask_ = 0;
    bool is_last_access_ = false;
    bool is_dirty_ = false;
  };
  activity::BuffetCommandLog command_log_;
  activity::PatternGeneratorLog read_pgen_log_;
  activity::PatternGeneratorLog update_pgen_log_;
  activity::PatternGeneratorLog destination_pgen_log_;
  activity::PatternGeneratorLog updaters_pgen_log_;
  activity::ShrinkPatternGeneratorLog shrink_pgen_log_;
  std::deque<std::pair<int, BuffetLogInfo>> coalescing_buffer_;
  int coalescing_window_size_ = options::kCoalescingWindowSize;
  
  int num_fills_ = 0;
  int num_shrinks_ = 0;
  int num_evicts_ = 0;
  
  bool CheckCoalescing(int address, int requestor_idx, bool is_update)
  {
    // Purposely search backwards to find the most recent entry.
    // This way even if the previous entry wasn't evicted will we find the latest.
    auto it = coalescing_buffer_.rbegin();
    // Just search the coalescing window. If there's less than that, stop early.
    auto end_it = coalescing_buffer_.size() >= coalescing_window_size_ ? 
                  coalescing_buffer_.rbegin() + coalescing_window_size_ :
                  coalescing_buffer_.rend();
    for ( ; it != end_it; it++)
    {
      if ((*it).first == address) break;
    }
    
    // Nothing to coalesce onto.
    if (it == end_it)
    {
      return false;
    }

    // Don't coalesce onto requests that are out-of-order for your request 
    // stream. NOTE: This is a very easy, greedy heuristic, but is very
    // conservative and may miss multicast opportunities compared to if
    // we were willing to re-do the interleaving of request streams.
    // However this heuristic will always allow forward progress and
    // never introduce a deadlock.
    for (auto it2 = it.base() ; it2 != coalescing_buffer_.end(); it2++)
    {
      if (!is_update && (*it2).second.receiver_mask_[requestor_idx])
      {
        return false;
      }
    }
    
    // We only support a multicast bitmask of up to 64.
    ASSERT(requestor_idx < 64) << "Multicast bitmask limit of 64 exceeded at index: " << requestor_idx << ", consider using AddTileLevel() to add intermediate buffers." << EndT;

    // Don't coalesce your own requests. Can't multicast to yourself!
    if (!is_update && (*it).second.receiver_mask_[requestor_idx])
    {
      return false;
    }

    // Mark this party as a receiver.
    (*it).second.receiver_mask_[requestor_idx] = true;

    if (is_update)
    {
      // For now, don't coalesce more than one update per updater on at a time....
      if ((*it).second.updater_mask_[requestor_idx])
      {
        return false;
      }
      // Mark this party as an updater, and the data as modified (if not already).
      (*it).second.updater_mask_[requestor_idx] = true;
      (*it).second.is_dirty_ = true;
    }
    // Succesful coalesce!
    return true;
  }
  
  BuffetLogInfo* LogAccess(int address, int requestor_idx, int index = -1)
  {
    // Keep track of last accessed address.
    BuffetLogInfo info;
    info.index_ = index;
    ASSERT(requestor_idx < 64) << "Requestor index exceeded multicast mask size: " << requestor_idx << " (connected fronting buffers: " << fronting_buffers_.size() << ")" << EndT;
    info.receiver_mask_[requestor_idx] = true;
    coalescing_buffer_.push_back(std::make_pair(address, info));
    return &coalescing_buffer_.back().second;
  }

  BuffetLogInfo* LogUpdate(int address, int requestor_idx, int index = -1)
  {
    T(4) << "Logging implicit RMW to address: " << address << " by requestor: " << requestor_idx << EndT;
    // Keep track of last accessed address.
    BuffetLogInfo info;
    info.index_ = index;
    ASSERT(requestor_idx < 64) << "Requestor index exceeded multicast mask size: " << requestor_idx << "(connected fronting buffers: " << fronting_buffers_.size() << ")" << EndT;
    info.receiver_mask_[requestor_idx] = true;
    info.updater_mask_[requestor_idx] = true;
    info.is_dirty_ = true;
    coalescing_buffer_.push_back(std::make_pair(address, info));
    return &coalescing_buffer_.back().second;
  }

  void TryToDrainOldestAccesses()
  {
    // Don't try to drain until the window is overfull.
    // At that point drain it back to being full.
    if (coalescing_buffer_.size() <= coalescing_window_size_) return;
    // We can drain the oldest acceses until we find one that is incomplete.
    // E.g. has an invalid index (-1)
    int num_to_try = coalescing_buffer_.size() - coalescing_window_size_;
    auto it = coalescing_buffer_.begin();
    for ( ; it != coalescing_buffer_.begin() + num_to_try; it++)
    {
      if ((*it).second.index_ == -1) break;
      RecordInfo((*it).first, (*it).second);
    }
    // Note: do this in two steps so we don't invalidate our iterator....
    coalescing_buffer_.erase(coalescing_buffer_.begin(), it);
  }
  
  void DrainLog()
  {
    // A bit of a hack:
    // Use a drain writeback if any of the remaining entries are modified.
    bool is_modified = false;
    // Drain all remaining coalescing buffer contents, oldest first.
    for (auto it  = coalescing_buffer_.begin(); 
              it != coalescing_buffer_.end();
              it++)
    {
      is_modified |= (*it).second.is_dirty_;
      // At this point all entries must be valid.
      assert((*it).second.index_ != -1);
      RecordInfo((*it).first, (*it).second);
    }
    coalescing_buffer_.clear();

    // Shrink off anything remaining in the buffet.
    if (num_fills_ > num_shrinks_)
    {
      if (is_modified)
      {
        command_log_.SetModified();
      }
      int remaining_lines = num_fills_ - num_shrinks_;
      command_log_.Shrink(remaining_lines);
      shrink_pgen_log_.Shrink(remaining_lines);
    }
  }
  
  void RecordInfo(int address, const BuffetLogInfo& info)
  {
    if (!options::kShouldLogActivity) return;

    // Record accces.
    bool got_coalesced = command_log_.Read(address, info.is_dirty_);
    if (!got_coalesced)
    {
      // We tell the pattern gens about this, since it wasn't coalesced.
      //T(0) << "Record: " << address << ", " << command_log_.GetRelativeIndex(info.index_) << ", " << info.index_ << EndT;
      read_pgen_log_.Send(command_log_.GetRelativeIndex(info.index_));
      destination_pgen_log_.Send(info.receiver_mask_.to_ulong());

      // Record any updates
      if (info.is_dirty_)
      {
        // NOTE: Update indexes are purposely absolute in the buffer.
        update_pgen_log_.Send(info.index_);
        // Minor hack: the datapath doesn't show up as a source.
        // So if the mask is 0, make it 1.
        long int final_mask = info.updater_mask_.to_ulong();
        updaters_pgen_log_.Send(final_mask == 0 ? 1 : final_mask);
      }
    }

    // Record any shrinks (after, so as not to mess with the indexing).
    if (info.is_last_access_)
    {
      if (info.is_dirty_)
      {
        command_log_.SetModified();
      }
      
      // Log the shrink.
      command_log_.Shrink();
      shrink_pgen_log_.Shrink();
      num_shrinks_++;
    }

  }
  
  int AlignAddress(int addr)
  {
    return (addr - (addr % fill_granularity_));
  }

 public:
  
  BindingTarget binding_;
  std::shared_ptr<BufferModel> backing_buffer_ = NULL;
  std::vector<std::shared_ptr<BufferModel>> fronting_buffers_;

  int level_;
  // Individual tensors may not participate in all global tile levels.
  // E.g. the L2 weights tile may be the last tile of weights.
  int starting_global_tile_level_;
  int ending_global_tile_level_;
  int local_spatial_idx_;
  int size_;
  int fill_granularity_;
  int access_granularity_;  // Expressed as even divisor of fill_granularity. e.g., fill_granularity = 6 means access granularity can be 1/2/3
  
  BufferModel(int level, int starting_tile_level, int local_spatial_idx, const std::string& nm, int sz, int access_granularity, int fill_granularity) :
    StatsCollection(nm, &top_stats),
    TraceableBuffer(nm),
    level_(level),
    starting_global_tile_level_(starting_tile_level),
    ending_global_tile_level_(starting_tile_level),
    local_spatial_idx_(local_spatial_idx),
    size_(sz/fill_granularity),
    fill_granularity_(fill_granularity),
    access_granularity_(access_granularity)
  {
    SetAccessGranularity(access_granularity_);
  }
  
  void LogActivity(std::ostream& ostr)
  {
    if (binding_.IsDisabled()) return;
    command_log_.Dump(ostr, Traceable::GetName() + "_commands", "symphony::modules::LogicalBuffet");
    ostr << "," << std::endl;
    read_pgen_log_.Dump(ostr, Traceable::GetName() + "_reads", "symphony::modules::LogicalGatedPatternGenerator");
    ostr << "," << std::endl;
    destination_pgen_log_.Dump(ostr, Traceable::GetName() + "_read_destinations", "symphony::modules::LogicalGatedPatternGenerator");
    ostr << "," << std::endl;
    update_pgen_log_.Dump(ostr, Traceable::GetName() + "_updates", "symphony::modules::LogicalGatedPatternGenerator");
    ostr << "," << std::endl;
    updaters_pgen_log_.Dump(ostr, Traceable::GetName() + "_update_sources", "symphony::modules::LogicalGatedPatternGenerator");
    ostr << "," << std::endl;
    shrink_pgen_log_.Dump(ostr, Traceable::GetName() + "_shrinks", "symphony::modules::LogicalGatedPatternGenerator");
  }

  void LogTopologyModule(std::ostream& ostr, int expansion_factor)
  {  
    if (binding_.IsDisabled()) return;
    ostr << "  - type: logical_component" << std::endl;
    ostr << "    class: symphony::modules::LogicalBuffetComplex" << std::endl;
    ostr << "    base_name: " << Traceable::GetName() << std::endl;
    ostr << "    configuration:" << std::endl;
    ostr << "      knobs_use_prefix: true" << std::endl;
    ostr << "      knobs:" << std::endl;
    ostr << "        - \"size_ = " <<  size_ * (fill_granularity_ / access_granularity_) << "\"" << std::endl;
    ostr << "        - \"base_ = " <<  0 << "\"" << std::endl;
    ostr << "        - \"bound_ = " <<  size_ * (fill_granularity_ / access_granularity_) << "\"" << std::endl;
    ostr << "        - \"use_external_fills_ = " <<  (starting_global_tile_level_ != 0) << "\"" << std::endl;
    ostr << "        - \"use_absolute_address_mode_ = false\"" << std::endl;
    ostr << "        - \"automatically_handle_fills_ = true\"" << std::endl;
    ostr << "        - \"automatically_handle_updates_ = true\"" << std::endl;
    ostr << "        - \"shrink_requires_data_up_to_date_ = false\"" << std::endl;
    ostr << "        - \"buffet_fill_line_size_in_bytes_ = " << fill_granularity_ * 4 << "\"" << std::endl;
    ostr << "        - \"buffet_read_update_line_size_in_bytes_ = " << access_granularity_ * 4 << "\"" << std::endl;
    if (binding_.GetLevel() == 0)
    {
      ostr << "        - \"use_memory_interface_ = true\"" << std::endl;
    }
    ostr << std::endl;
  }

  void LogTopologyConnections(std::ostream& ostr, int id, int spatial_idx, int expansion_factor)
  {
    if (binding_.IsDisabled()) return;
    for (int dst_idx = 0; dst_idx < fronting_buffers_.size(); dst_idx++)
    {
      if (fronting_buffers_[dst_idx]->binding_.IsDisabled()) continue;
      // The buffer feeds another (usually smaller) buffer.
      ostr << "      - src:" << std::endl;
      ostr << "        - name: " << Traceable::GetName() << std::endl;
      ostr << "          port-name: read_data_out_" << dst_idx << std::endl;
      ostr << "        dst:" << std::endl;
      ostr << "          - name: " << fronting_buffers_[dst_idx]->Traceable::GetName() << std::endl;
      ostr << "            port-name: fill_data_in_0" << std::endl;
      ostr << "        configuration:" << std::endl;
      ostr << "          credited: true" << std::endl;
      ostr << "      - src:" << std::endl;
      ostr << "        - name: " << fronting_buffers_[dst_idx]->Traceable::GetName() << std::endl;
      ostr << "          port-name: drain_data_out_0" << std::endl;
      ostr << "        dst:" << std::endl;
      ostr << "        - name: " << Traceable::GetName() << std::endl;
      ostr << "          port-name: update_data_in_" << dst_idx << std::endl;
    }
    // The buffer feeds datapaths between it and the next tile level 
    // where this tensor participates.
    // If there is no such tile level then it covers all the remaining levels.
    int end_level = ending_global_tile_level_ == starting_global_tile_level_ ? compute_logs.size() : ending_global_tile_level_;
    int local_dp_index = 0;
    for (int tile_level = starting_global_tile_level_; tile_level < end_level; tile_level++)
    {
      int num_dpaths = compute_logs[tile_level].size() / expansion_factor;
      for (int x = 0; x < num_dpaths; x++)
      {
        int dp_idx = spatial_idx * num_dpaths + x;
        if (compute_bindings[tile_level][dp_idx].IsDisabled()) continue;
        ostr << "      - src:" << std::endl;
        ostr << "        - name: " << Traceable::GetName() << std::endl;
        ostr << "          port-name: read_data_out_" << fronting_buffers_.size() + local_dp_index << std::endl;
        ostr << "        dst:" << std::endl;
        ostr << "        - name: " << GetComputeBoundName(tile_level, dp_idx) << std::endl;
        ostr << "          port-name: input_" << id << std::endl; // Note: This index could be smarter.
        ostr << "      - src:" << std::endl;
        ostr << "        - name: " << GetComputeBoundName(tile_level, dp_idx) << std::endl;
        ostr << "          port-name: output_" << id << std::endl;
        ostr << "        dst:" << std::endl;
        ostr << "        - name: " << Traceable::GetName() << std::endl;
        ostr << "          port-name: update_data_in_" << fronting_buffers_.size() + local_dp_index << std::endl;
        local_dp_index++;
      }
    }
  }

  void LogTopologyRoutes(std::ostream& ostr, int id, int spatial_idx, int expansion_factor)
  {
    if (binding_.IsDisabled()) return;
    for (int dst_idx = 0; dst_idx < fronting_buffers_.size(); dst_idx++)
    {
      if (fronting_buffers_[dst_idx]->binding_.IsDisabled()) continue;
      // The buffer feeds another (usually smaller) buffer.
      // Does it go to a direct connection, or to the network?
      int phys_idx = GetPhysicalIndex(binding_, fronting_buffers_[dst_idx]->binding_);
      ostr << " - Route:" << std::endl;
      ostr << "   - Circuit_id: 0" << std::endl;
      ostr << "     type: unicast" << std::endl;
      ostr << "     logical_src:" << std::endl;
      ostr << "       logical_name: " << Traceable::GetName() << std::endl;
      ostr << "       logical_connection: read_data_out_" << dst_idx << std::endl;
      ostr << "     logical_dst:" << std::endl;
      ostr << "       logical_name: " << fronting_buffers_[dst_idx]->Traceable::GetName() << std::endl;
      ostr << "       logical_connection: fill_data_in_0" << std::endl;
      ostr << "     physical_data_path:" << std::endl;
      ostr << "       - physical_module_name: system:" << binding_.ToString() << "-BCC" << std::endl;
      ostr << "         physical_connection_name: read_out_" << phys_idx << std::endl;
      ostr << "       - physical_module_name: system:" << fronting_buffers_[dst_idx]->binding_.ToString() << "-BCC" << std::endl;
      ostr << "         physical_connection_name: fill_in_0" << std::endl;
      ostr << std::endl;
      ostr << " - Route:" << std::endl;
      ostr << "   - Circuit_id: 1" << std::endl;
      ostr << "     type: unicast" << std::endl;
      ostr << "     logical_src:" << std::endl;
      ostr << "       logical_name: " << fronting_buffers_[dst_idx]->Traceable::GetName() << std::endl;
      ostr << "       logical_connection: drain_data_out_0" << std::endl;
      ostr << "     logical_dst:" << std::endl;
      ostr << "       logical_name: " << Traceable::GetName() << std::endl;
      ostr << "       logical_connection: update_data_in_" << dst_idx << std::endl;
      ostr << "     physical_data_path:" << std::endl;
      ostr << "       - physical_module_name: system:" << fronting_buffers_[dst_idx]->binding_.ToString() << "-BCC" << std::endl;
      ostr << "         physical_connection_name: drain_out_0" << std::endl;
      ostr << "       - physical_module_name: system:" << binding_.ToString() << "-BCC" << std::endl;
      ostr << "         physical_connection_name: update_in_" << phys_idx << std::endl;
      ostr << std::endl;
    }
    
    // The buffer feeds datapaths between it and the next tile level 
    // where this tensor participates.
    // If there is no such tile level then it covers all the remaining levels.
    int end_level = ending_global_tile_level_ == starting_global_tile_level_ ? compute_logs.size() : ending_global_tile_level_;
    int local_dp_index = 0;
    for (int tile_level = starting_global_tile_level_; tile_level < end_level; tile_level++)
    {
      int num_dpaths = compute_logs[tile_level].size() / expansion_factor;
      for (int x = 0; x < num_dpaths; x++)
      {
        int dp_idx = spatial_idx * num_dpaths + x;
        if (compute_bindings[tile_level][dp_idx].IsDisabled()) continue;
        BindingTarget dst_binding = compute_bindings[tile_level][dp_idx];
        ASSERT_WARNING(binding_.GetLevel() == dst_binding.GetLevel()) << "Attempted route to non-local CECC. Logical Src: " << Traceable::GetName() << ", Physical Src: " << binding_.ToString() << ", Logical Dst: " << GetComputeBoundName(tile_level, dp_idx) << ", Physical Dst: " << dst_binding.ToString() << ". YAML may be incorrect." << EndT;
        if (binding_.GetLevel() != dst_binding.GetLevel())
        {
          local_dp_index++;
          continue;
        }
        // Does it go to the local datapath, or to the network?
        int phys_idx = compute_bindings[tile_level][dp_idx] == binding_ ? 0 : fronting_buffers_.size();
        ostr << " - Route:" << std::endl;
        ostr << "   - Circuit_id: 2" << std::endl;
        ostr << "     type: unicast" << std::endl;
        ostr << "     logical_src:" << std::endl;
        ostr << "       logical_name: " << Traceable::GetName() << std::endl;
        ostr << "       logical_connection: read_data_out_" << fronting_buffers_.size() + local_dp_index << std::endl;
        ostr << "     logical_dst:" << std::endl;
        ostr << "       logical_name: " << GetComputeBoundName(tile_level, dp_idx) << std::endl;
        ostr << "       logical_connection: input_" << id << std::endl;
        ostr << "     physical_data_path:" << std::endl;
        ostr << "       - physical_module_name: system:" << binding_.ToString() << "-BCC" << std::endl;
        ostr << "         physical_connection_name: read_out_" << phys_idx <<  std::endl;
        ostr << "       - physical_module_name: system:" << compute_bindings[tile_level][dp_idx].ToString() << "-CECC" << std::endl;
        ostr << "         physical_connection_name: data_in_0" << std::endl;
        ostr << std::endl;
        ostr << " - Route:" << std::endl;
        ostr << "   - Circuit_id: 3" << std::endl;
        ostr << "     type: unicast" << std::endl;
        ostr << "     logical_src:" << std::endl;
        ostr << "       logical_name: " << GetComputeBoundName(tile_level, dp_idx) << std::endl;
        ostr << "       logical_connection: output_" << id << std::endl;
        ostr << "     logical_dst:" << std::endl;
        ostr << "       logical_name: " << Traceable::GetName() << std::endl;
        ostr << "       logical_connection: update_data_in_" << fronting_buffers_.size() + local_dp_index << std::endl;
        ostr << "     physical_data_path:" << std::endl;
        ostr << "       - physical_module_name: system:" << compute_bindings[tile_level][dp_idx].ToString() << "-CECC" << std::endl;
        ostr << "         physical_connection_name: data_out_0" << std::endl;
        ostr << "       - physical_module_name: system:" << binding_.ToString() << "-BCC" << std::endl;
        ostr << "         physical_connection_name: update_in_" << phys_idx << std::endl;
        ostr << std::endl;
        local_dp_index++;
      }
    }
  }

  void Resize(int size)
  {
    size_ = size / fill_granularity_;
    command_log_.Resize(size_);
  }

  int GetNumReceivers(int expansion_factor)
  {
    int flat_datapaths = 0;
    auto end_it = ending_global_tile_level_ == starting_global_tile_level_ ? compute_logs.end() : compute_logs.begin() + ending_global_tile_level_;
    for (auto it = compute_logs.begin() + starting_global_tile_level_; it != end_it; it++)
    {
      flat_datapaths += (*it).size();
    }
    return fronting_buffers_.size() + (flat_datapaths/expansion_factor);
  }
  
  int GetComputeIndex(int tile_level, int local_spatial_idx, int expansion_factor)
  {
    int flat_datapaths = 0;
    assert(tile_level < compute_logs.size());
    for (auto it = compute_logs.begin() + starting_global_tile_level_; it != compute_logs.begin() + tile_level; it++)
    {
      flat_datapaths += (*it).size() / expansion_factor;
    }
    return fronting_buffers_.size() + flat_datapaths + local_spatial_idx;
  }
 
  virtual void Access(int address, int requestor_idx) = 0;
  virtual void Update(int address, int requestor_idx) = 0;
  virtual bool DrainOne() = 0;
  virtual void DrainAll() = 0;
  virtual void FlushBuffet() = 0;
  virtual void FixupLevelNumbering(int final_num_levels) = 0;
  virtual void SetBufferWidth(int width) = 0;
  
  virtual void Bind(const BindingTarget& target) 
  {
    ASSERT(binding_.IsUnbound()) << "Multiple bindings received. Old binding: " << binding_.ToString() << ", new: " << target.ToString() << EndT;
    binding_ = target;
    Traceable::PrependToName(binding_.ToString() + "_");
    StatsCollection::PrependToName(binding_.ToString() + "_");
    AddPhysicalBufferMap(binding_, Traceable::GetName());
  }
  virtual void SetAccessGranularity(int g)
  {
    assert(g > 0);
    ASSERT(level_ == 0 || fill_granularity_ % g == 0) << "Access granularity must be an even divisor of fill granularity. Fill: " << fill_granularity_ << ", access: " << g << EndT;
    access_granularity_ = g;
    command_log_.SetAccessThreshold(fill_granularity_ /  access_granularity_);
  }
  
};

class BackingBufferModel : public BufferModel
{
 private:
  int GetIdx(int address)
  {
    return address / rowbuffer_width_;
  }
 protected:
  void CheckRowBufferHit(int address)
  {
    int addr_idx = GetIdx(address);

    if(rowbuffer_active_idx_ == addr_idx)
    {
      IncrStat("Backing row buffer hit");
      T(4) << "Row buffer hit for read to address " << address << EndT;
    }
    else
    {
      rowbuffer_active_idx_ = addr_idx;
      IncrStat("Backing row buffer miss");
      T(4) << "Row buffer miss for read to address " << address << EndT;
    }
  }

 public:
  int rowbuffer_active_idx_ = -1;
  int rowbuffer_width_ = 16; //This is a default value; users can modify it via "rowbuffer_width_(name)" option
  
  BackingBufferModel(const std::string& nm, int size, int access_granularity) : 
    BufferModel(0, 0, 0, nm, size, access_granularity, access_granularity)
  {
    AddOption(&rowbuffer_width_, "rowbuffer_width_" + nm, "The width of row buffer in DRAM (backing memory)" + nm);
    command_log_.Init(size, false);
  }

  virtual void Access(int addr, int requestor_idx)
  {
    int address = AlignAddress(addr);
    if (CheckCoalescing(address, requestor_idx, false))
    {  
      IncrStat("Backing multicasts");
      T(4) << "Read (coalesced), address: " << addr << "(line_address: " << address << ") by requestor: " << requestor_idx  << EndT;
    }
    else
    {
      CheckRowBufferHit(address);
      IncrStat("Backing reads");
      T(4) << "Read, address: " << addr << " (line_address: " << address << ") by requestor: " << requestor_idx << EndT;
      LogAccess(address, requestor_idx, address);
      TryToDrainOldestAccesses();
    }
  }
  
  virtual void Update(int addr, int requestor_idx)
  {
    int address = AlignAddress(addr);
    if (CheckCoalescing(address, requestor_idx, true))
    {  
      IncrStat("Backing multi-reduces");
      T(4) << "Update (coalesced), address: " << addr << "(line_address: " << address << ") by requestor: " << requestor_idx  << EndT;
    }
    else
    {
      CheckRowBufferHit(address);

      IncrStat("Backing updates");
      T(4) << "Update, address: " << addr  << " (line_address: " << address << ") by requestor: " << requestor_idx << EndT;
      LogUpdate(address, requestor_idx, address);
      TryToDrainOldestAccesses();
    }
  }

  virtual void FlushBuffet()
  {
  }
  virtual bool DrainOne()
  {
    return true;
  }
  virtual void DrainAll()
  {
    DrainLog();
  }
  virtual void FixupLevelNumbering(int final_num_levels)
  {
    // Intentional no-op.
  }

  virtual void SetBufferWidth(int buffer_width)
  {
    rowbuffer_width_ = buffer_width;
  }
};

class AssociativeBufferModel : public BufferModel
{
 public:
 
  class EntryInfo
  {
   public:
    bool modified_ = false;
    std::deque<BuffetLogInfo*> accesses_{};
  };
  std::unordered_map<int, EntryInfo> presence_info_;
  std::list<int> lru_cache_; // list occupancy should never exceed size_. Back = least recently used.
  int occupancy_ = 0;
  
  void SetRecentlyUsed(int address)
  {
    auto it = std::find(lru_cache_.begin(), lru_cache_.end(), address);
    if (it != lru_cache_.end())
    {
      lru_cache_.push_front(*it); // Does not invalidate iterator in std::list
      lru_cache_.erase(it);
    }
  }
    
  explicit AssociativeBufferModel(int size, int level, int starting_tile_level, int local_spatial_idx, const std::string& nm, int shrink_threshold, int access_granularity, int fill_granularity) :
    BufferModel(level, starting_tile_level, local_spatial_idx, nm, size, access_granularity, fill_granularity) 
  {
    command_log_.Init(size_);
    int final_shrink_thresh = shrink_threshold == 0 ? size_ : shrink_threshold/fill_granularity;
    command_log_.SetShrinkThreshold(final_shrink_thresh);
    shrink_pgen_log_.SetShrinkThreshold(final_shrink_thresh);
  }

  void EvictLRU()
  {
    // Determine victim
    int victim_addr = lru_cache_.back();
    EntryInfo victim = presence_info_[victim_addr];

    // We now have enough info to determine the victim's buffet slot.
    int slot = num_evicts_ % size_;
    //T(0) << "Slot: " << slot << ", num_evicts: " << num_evicts_ << ", size:" << size_ << EndT;
    num_evicts_++;

    // Update all logged accesses to have the slot info.
    for (auto access : victim.accesses_)
    {
      access->index_ = slot;
    }
    // Mark the last access so we can send a shrink.
    if (victim.accesses_.size() > 0)
    {
      victim.accesses_.back()->is_last_access_ = true;
    }

    // Check if the above completed any of the oldest accesses.
    TryToDrainOldestAccesses();

    // remove victim from unordered_map
    presence_info_.erase(victim_addr);

    // We now have enough info to back-log the fill.
    backing_buffer_->Access(victim_addr, local_spatial_idx_);
    if (victim.modified_)
    {
      backing_buffer_->Update(victim_addr, local_spatial_idx_);
    }

    // Final book-keeping updates
    occupancy_--;
    lru_cache_.pop_back();
  }

    // Note that the address coming in is just the index within the tensor, not
    // the actual address
  virtual void Access(int addr, int requestor_idx)
  {
    int address = AlignAddress(addr);
    
    if (CheckCoalescing(address, requestor_idx, false))
    {
      IncrStat("L" + std::to_string(level_) + " multicasts");
      T(4) << "Read (coalesced), address: " << addr << "(line_address: " << address << ") by requestor: " << requestor_idx << EndT;
      SetRecentlyUsed(address);
      return;
    }
    
    IncrStat("L" + std::to_string(level_) + " reads");
    T(4) << "Read, address: " << addr << "(line_address: " << address << ") by requestor: " << requestor_idx << EndT;

    bool found        = false;
    bool caused_evict = false;
    
    auto it = presence_info_.find(address);
    if (it != presence_info_.end())
    {
      found = true;
      T(4) << "    (Already present in buffer)" << EndT;
      SetRecentlyUsed(address);
    }
    
    if (!found)
    {
      IncrStat("L" + std::to_string(level_) + " fills");
      num_fills_++;
      if (occupancy_ == size_)
      {
        EvictLRU();
        caused_evict = true;
      }
      T(4) << "    (Filling from next level, caused_evict:" << caused_evict << ")" << EndT;

      // Start out as most recently used.
      lru_cache_.emplace_front(address);
      occupancy_++;

      // insert into unordered_map with default info (e.g., clean)
      presence_info_[address] = EntryInfo();
    }
    presence_info_[address].accesses_.push_back(LogAccess(address, requestor_idx));
  }

  // Note that the address coming in is just the index within the tensor, not
  // the actual address
  virtual void Update(int addr, int requestor_idx)
  {
    int address = AlignAddress(addr);

    if (CheckCoalescing(address, requestor_idx, true))
    {
      IncrStat("L" + std::to_string(level_) + " multi-reduces");
      T(4) << "Update (coalesced), address: " << addr << "(line_address: " << address << ") by requestor: " << requestor_idx << EndT;
      presence_info_[address].modified_ = true;
      return;
    }
    
    IncrStat("L" + std::to_string(level_) + " updates");
    T(4) << "Update, address: " << addr << "(line address: " << address << ") by requestor: " << requestor_idx << EndT;

    bool found        = false;
    bool caused_evict = false;

    auto it = presence_info_.find(address);
    if (it != presence_info_.end())
    {
      found = true;
      presence_info_[address].modified_ = true;
      T(4) << "    (Already present in buffer)" << EndT;
      SetRecentlyUsed(address);
    }
    
    if (!found)
    {
      num_fills_++;
      if (occupancy_ == size_)
      {
        EvictLRU();
        caused_evict = true;
      }
      T(4) << "    (Allocating on write, caused_evict:" << caused_evict << ")" << EndT;

      // Start out as most recently used.
      lru_cache_.emplace_front(address);
      occupancy_++;

      // insert into unordered_map with default info (e.g., clean)
      presence_info_[address] = EntryInfo();
      presence_info_[address].modified_ = true;
    }
    presence_info_[address].accesses_.push_back(LogUpdate(address, requestor_idx));
    command_log_.SetModified();
  }

  virtual bool DrainOne()
  {
    if (occupancy_ != 0)
    {
      EvictLRU();
    }
    return (occupancy_ == 0);
  }

  virtual void DrainAll()
  {
    // Drain in LRU order.
    while (!lru_cache_.empty())
    {
      EvictLRU();
    }
    assert(occupancy_ == 0);
    DrainLog();
  }

  virtual void FlushBuffet()
  {
      presence_info_.clear();
      DrainAll();
  }
  
  virtual void FixupLevelNumbering(int final_num_levels)
  {
    if (!binding_.IsUnbound()) return;
    level_ = final_num_levels - level_;
    std::string prefix = "l" + std::to_string(level_) + "_";
    Traceable::PrependToName(prefix);
    StatsCollection::PrependToName(prefix);
  }

  virtual void SetBufferWidth(int buffer_width)
  {
    //not supported in Buffet
    return;
  }
};

} // namespace buff

namespace ast
{

using VarName = std::string;

class PrimVar : public StatsCollection
{
 public:
  
  std::vector<int> vals_;
  
  PrimVar() = default;

  PrimVar(const std::string& nm) : 
    StatsCollection(nm, &top_stats)
  {
  }

  std::shared_ptr<timewhoop::Container> ConvertExpression()
  {
    std::shared_ptr<timewhoop::Container> converted_var = std::shared_ptr<timewhoop::Variable>(new timewhoop::Variable(timewhoop::Type::INTEGER, this->name_));

    return converted_var;
  }
  
  void InitializePartitioning(const int& flattened_num_partitions)
  {
    vals_.resize(flattened_num_partitions, 0xAAAAAAAA);
  }
  
  void Update(const int& flat_base, const int& flat_bound, const int& new_val)
  {
    IncrStat("Var updates");
    for (int x = flat_base; x < flat_bound; x++)
    {
      vals_[x] = new_val;
    }
  }
  
  int Access(const int& flat_base, const int& flat_bound)
  {
    IncrStat("Var reads");
    int result = vals_[flat_base];
    for (int x = flat_base; x < flat_bound; x++)
    {
      assert(result == vals_[x]);
    }
    return result;
  }
};


class PrimTensor : public StatsCollection
{

  friend class boost::serialization::access;

//  protected:
  public:
    
  UINT64 FlattenSizes(const std::vector<int>& sizes)
  {
    UINT64 res = 1;
    for (auto it = sizes.begin(); it != sizes.end(); it++)
    {
      res *= (*it);
    }
    return res;
  }
  
  UINT64 FlattenIndices(const std::vector<int>& idxs) const
  {
    //std::cout << "  Flattening for: " <<  name_ << " " << idxs.size() << " " <<  dim_sizes_.size() << std::endl;
    

    UINT64 res = 0;
    UINT64 amplification_factor = 1;
    assert(idxs.size() == dim_sizes_.size());
    auto size_it = dim_sizes_.begin();
    for (auto rit = idxs.rbegin(); rit != idxs.rend(); rit++)
    {
      res += (*rit) * amplification_factor;
      amplification_factor *= (*size_it);
      size_it++;
    }
    return res;
  }
  
  std::vector<int> UnflattenIndex(UINT64 x)
  {
    UINT64 cur = x;
    UINT64 num_dims = dim_sizes_.size();
    std::vector<int> res(num_dims);
    for (auto k = 0; k < num_dims; k++)
    {
      res[k] = cur % dim_sizes_[k];
      cur = cur / dim_sizes_[k];
    }
    return res;
  }
  
  void PrimInitialize(const UINT64& start_idx, 
                      const UINT64& end_idx,
                      int (*func)(const std::vector<int>& idxs))
  {
    assert(start_idx <= end_idx);
    for (auto x = start_idx; x < end_idx; x++)
    {
      auto idxs = UnflattenIndex(x);
      vals_[x] = func(idxs);
    }
  }
  
  void PrimTraverse(const UINT64& start_idx, 
                    const UINT64& end_idx,
                    void (*func)(const int& val, const std::vector<int>& idxs))
  {
    assert(start_idx <= end_idx);
    for (auto x = start_idx; x < end_idx; x++)
    {
      auto idxs = UnflattenIndex(x);
      func(vals_[x], idxs);
    }
  }
  
 public:
  
  std::vector<int> dim_sizes_; // 0 == innermost, N-1 == outermost
  std::vector<int> vals_;
  // A global id for this tensor in the list of all tensors.
  int id_;
  // The following information is important both for optimizations
  // and for correct tracefile generation.
  bool is_updated_dynamically_ = false;
  
  // We use a stack to represent the tile levels so that we can
  // add to it as we go through loops. (Impelemented as a deque for iterators.)
  std::vector<std::shared_ptr<std::deque<std::shared_ptr<std::vector<std::shared_ptr<buff::BufferModel>>>>>> buffer_levels_{};
  
  explicit PrimTensor(const std::string& nm = "") :
    StatsCollection(nm, &top_stats)
  {
  }
  
  PrimTensor(const std::vector<int>& dim_sizes, int (*init_func)(const std::vector<int>& idxs), const std::string& nm = "") :
    dim_sizes_(dim_sizes.rbegin(), dim_sizes.rend()), // Reverse the dim sizes
    vals_(FlattenSizes(dim_sizes)), 
    StatsCollection(nm, &top_stats)
  {
    InitializeAll(init_func);
  }
  
  PrimTensor(const std::vector<int>& dim_sizes, const int& init_val, const std::string& nm = "") :
    dim_sizes_(dim_sizes.rbegin(), dim_sizes.rend()), // Reverse the dim sizes
    vals_(FlattenSizes(dim_sizes), init_val),
    StatsCollection(nm, &top_stats)
  {
  }
  
  explicit PrimTensor(const std::vector<int>& dim_sizes, const std::string& nm = "") :
    PrimTensor(dim_sizes, 0xAAAAAAAA, nm)
  {
  }

  void FillVal( int fill_val_ )
  {
    std::fill(vals_.begin(), vals_.end(), fill_val_);
  }
    
  void InitializeAll(int (*func)(const std::vector<int>& idxs))
  {
    PrimInitialize(0, PrimSize(), func);
  }
  
  void Initialize(std::vector<int>& start_idx, std::vector<int>& end_idx, int (*func)(const std::vector<int>& idxs))
  {
    PrimInitialize(FlattenIndices(start_idx), FlattenIndices(end_idx), func);
  }

  void TraverseAll(void (*func)(const int& val, const std::vector<int>& idxs))
  {
    PrimTraverse(0, PrimSize(), func);
  }
  
  void Traverse(std::vector<int>& start_idx, std::vector<int>& end_idx, void (*func)(const int& val, const std::vector<int>& idxs))
  {
    PrimTraverse(FlattenIndices(start_idx), FlattenIndices(end_idx), func);
  }

  void Update(const std::vector<int>& idxs, const int& new_val, const int& access_tile_level, const int& compute_tile_level, const int& spatial_part_idx = 0, const int& num_spatial_partitions = 1,  const int& port_idx = 0)
  {
    is_updated_dynamically_ = true;
    // TODO: better error messages.
    int buffer_spatial_part_idx = whoop::buff::GetBufIndex( spatial_part_idx, (*buffer_levels_[port_idx])[access_tile_level]->size(), num_spatial_partitions );
    int num_buffers = (*buffer_levels_[port_idx])[access_tile_level]->size();
    int local_spatial_idx = spatial_part_idx % (num_spatial_partitions / num_buffers);

    IncrStat("Tensor updates");
    UINT64 idx = FlattenIndices(idxs);
    vals_[idx] = new_val;
    auto buffer_to_access = (*(*buffer_levels_[port_idx])[access_tile_level])[buffer_spatial_part_idx];
    // Datapath access ids are offset by the fronting buffers and any previous datapaths.
    int my_final_idx = buffer_to_access->GetComputeIndex(compute_tile_level, local_spatial_idx, num_buffers);
    buffer_to_access->Update(idx, my_final_idx);
  }
  
  int Access(const std::vector<int>& idxs, const int& access_tile_level, const int& compute_tile_level, const int& spatial_part_idx = 0, const int& num_spatial_partitions = 1, const int& port_idx = 0)
  {
    int buffer_spatial_part_idx = whoop::buff::GetBufIndex( spatial_part_idx, (*buffer_levels_[port_idx])[access_tile_level]->size(), num_spatial_partitions );      
    int num_buffers = (*buffer_levels_[port_idx])[access_tile_level]->size();
    int local_spatial_idx = spatial_part_idx % (num_spatial_partitions / num_buffers);

    IncrStat("Tensor reads");
    UINT64 idx = FlattenIndices(idxs);
    auto buffer_to_access = (*(*buffer_levels_[port_idx])[access_tile_level])[buffer_spatial_part_idx];
    // Datapath access ids are offset by the fronting buffers and any previous datapaths.
    int my_final_idx = buffer_to_access->GetComputeIndex(compute_tile_level, local_spatial_idx, num_buffers);
    buffer_to_access->Access(idx, my_final_idx);
    return vals_[idx];
  }

  int& At(const std::vector<int>& idxs)
  {
    UINT64 idx = FlattenIndices(idxs);
    return vals_[idx];
  }

  const int& At(const std::vector<int>& idxs) const
  {
    UINT64 idx = FlattenIndices(idxs);
    return vals_[idx];
  }
  
  int& PrimAt(const int& idx)
  {
    return vals_[idx];
  }

  const int& PrimAt(const int& idx) const
  {
    return vals_[idx];
  }

  void PrimPushBack(const int& val, const int& dim)
  {
    vals_.push_back(val);
    dim_sizes_[dim]++;
  }

  void PrimShrinkToFit()
  {
    vals_.shrink_to_fit();
  }

  UINT64 size()
  {
      return vals_.size();
  }
  
  UINT64 PrimSize()
  {
    return vals_.size();
  }
  
  std::vector<int> PrimUnflattenIndex(UINT64 x)
  {
    return UnflattenIndex(x);
  }

  void Resize(const std::vector<int>& dim_sizes)
  {
    // Reverse the dimension sizes.
    dim_sizes_.assign(dim_sizes.rbegin(), dim_sizes.rend());
    UINT64 s = FlattenSizes(dim_sizes_);
    vals_.resize(s);
    // Tell the backing buffer the new size.
    (*buffer_levels_[0])[0]->at(0)->Resize(s);
  }
  
  // Used when loading tensors from files to cleanup buffer model.
  void FixupSize()
  {
    // Tell the backing buffer the new size.
    UINT64 s = FlattenSizes(dim_sizes_);
    (*buffer_levels_[0])[0]->at(0)->Resize(s);
  }
  
  int Size(int dim)
  {
    // Note: No reversal here since we have reversed already.
    return dim_sizes_[dim];
  }
  
  int NumDimensions()
  {
    return dim_sizes_.size();
  }
  
  std::vector<int> DimensionSizes()
  {
    return dim_sizes_;
  }
  
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & dim_sizes_;
    ar & vals_;
  }

  void FixupBufferLevelNumbering()
  {
    for (auto port_it : buffer_levels_)
    {
      int final_num_levels = (*port_it).size();
      for (auto level_it : *port_it)
      {
        for (auto buf_it : *level_it)
        {
          (*buf_it).FixupLevelNumbering(final_num_levels);
        }
      }
    }
  }

  void DrainAll()
  {
    // Purposely iterate ports backwards since backing is always port 0.
    for (auto port_rit = buffer_levels_.rbegin(); port_rit != buffer_levels_.rend(); port_rit++)
    {
      // Purposely iterate from closer levels on down, for maximum locality.
      // Skip level 0 since it is the same backing for each port and handled specially
      for (auto level_rit = (*port_rit)->rbegin(); level_rit != std::prev((*port_rit)->rend()); level_rit++)
      {
        // yuhsinc: at each level, evict 1 entry from each buffet at a time in a round robin fashion
        bool all_empty = false;
        while (!all_empty)
        {
          all_empty = true;
          for (auto buf : *(*level_rit))
          {
            all_empty &= buf->DrainOne();
          }
        }
        // Clean out the logs and any final activity.
        for (auto buf : *(*level_rit))
        {
           buf->DrainAll();
        }
      }
    }
    // Every tensor has a backing level.
    (*buffer_levels_[0])[0]->at(0)->DrainAll();
  }
  
  void SetTraceLevel(int level)
  {
    if (!options::kShouldTraceBuffers) return;
    for (auto port_it : buffer_levels_)
    {
      // Purposely iterate from closer levels on down, for maximum locality.
      for (auto level_it : *port_it)
      {
        for (auto buf_it : *level_it)
        {
          (*buf_it).SetTraceLevel(level);
        }
      }
    }
  }
 
  void LogActivity(std::ostream& ostr)
  {
    for (auto& port : buffer_levels_)
    {
      // Skip the backing level as it is repeated across ports.
      for (auto level_it = std::next(port->begin()); level_it != port->end(); level_it++)
      {
        for (auto& buff : *(*level_it))
        {
          buff->LogActivity(ostr);
          ostr << "," << std::endl;
        }
      }
    }
    // Every tensor has a backing level.
    (*buffer_levels_[0])[0]->at(0)->LogActivity(ostr);
  }

  void LogTopologyModules(std::ostream& ostr)
  {
    for (auto& port : buffer_levels_)
    {
      // Skip the backing level as it is repeated across ports.
      for (auto level_it = std::next(port->begin()); level_it != port->end(); level_it++)
      {
        for (auto& buff : *(*level_it))
        {
          buff->LogTopologyModule(ostr, (*level_it)->size());
        }
      }
    }
    // Every tensor has a backing level.
    (*buffer_levels_[0])[0]->at(0)->LogTopologyModule(ostr, 1);
  }

  void LogTopologyConnections(std::ostream& ostr)
  {
    for (auto& port : buffer_levels_)
    {
      // Skip the backing level as it is repeated across ports.
      for (auto level_it = std::next(port->begin()); level_it != port->end(); level_it++)
      {
        int local_spatial_idx = 0;
        for (auto& buff : *(*level_it))
        {
          buff->LogTopologyConnections(ostr, id_, local_spatial_idx, (*level_it)->size());
          local_spatial_idx++;
        }
      }
    }
    // Every tensor has a backing level.
    (*buffer_levels_[0])[0]->at(0)->LogTopologyConnections(ostr, id_, 0, 1);
  }

  void LogTopologyRoutes(std::ostream& ostr)
  {
    for (auto& port : buffer_levels_)
    {
      // Skip the backing level as it is repeated across ports.
      for (auto level_it = std::next(port->begin()); level_it != port->end(); level_it++)
      {
        int local_spatial_idx = 0;
        for (auto& buff : *(*level_it))
        {
          buff->LogTopologyRoutes(ostr, id_, local_spatial_idx, (*level_it)->size());
          local_spatial_idx++;
        }
      }
    }
    // Every tensor has a backing level.
    (*buffer_levels_[0])[0]->at(0)->LogTopologyRoutes(ostr, id_, 0, 1);
  }

  void BindToDefaults()
  {
    for (auto port_it : buffer_levels_)
    {
      for (int level = 0; level < port_it->size(); level++)
      {
        for (auto spatial_idx = 0; spatial_idx < (*port_it)[level]->size(); spatial_idx++)
        {
          auto cur_buff = (*(*port_it)[level])[spatial_idx];
          if (cur_buff->binding_.IsUnbound())
          {
            cur_buff->Bind(GetDefaultBinding(cur_buff->starting_global_tile_level_, spatial_idx, (*port_it)[level]->size()));
          }
        }
      }
    }
  }

};

class Statement;
  
class ExecutionContext
{
 public:
  class PartitionContext
  {
   public:
    // These are stored flattened (meaning expanded across all nested s_fors)
    int current_spatial_partition_ = 0;
    int num_spatial_partitions_ = 1;
    // Sometimes we need to access things using flattened indices
    // rather than relative.
    int flat_base_ = 0;
    int flat_stride_ = 1;
    int flat_cur_ = 0;
  };
  std::deque<PartitionContext> partition_stack_{};
  PartitionContext active_;
  
  ExecutionContext(int num_flat_partitions)
  {
    active_.flat_stride_ = num_flat_partitions;
  }
    
  void BeginSpatialPartitioning(int num_partitions, int flat_expansion)
  {
    // Copy the current info and save it.
    partition_stack_.push_back(active_);
    // Update the current info.
    active_.current_spatial_partition_ *= num_partitions;
    active_.num_spatial_partitions_ *= num_partitions;
    active_.flat_stride_ = flat_expansion;
    active_.flat_base_ = active_.current_spatial_partition_ * flat_expansion;
    active_.flat_cur_ = active_.flat_base_;
  }

  void EndSpatialPartitioning()
  {
    // Restore the old info.
    active_ = partition_stack_.back();
    partition_stack_.pop_back();
  }

  void NextSpatialPartition()
  {
    active_.current_spatial_partition_++;
    active_.flat_cur_ += active_.flat_stride_;
  } 

  int CurrentSpatialPartition()
  {
    return active_.current_spatial_partition_;
  }

  int NumSpatialPartitions()
  {
    return active_.num_spatial_partitions_;
  }

  void RestartCurrentPartitions()
  {
    active_.current_spatial_partition_ = partition_stack_.back().current_spatial_partition_ * partition_stack_.back().num_spatial_partitions_;
    active_.flat_cur_ = active_.flat_base_;
  }
  
  int FlatBegin()
  {
    return active_.flat_cur_;
  }

  int FlatEnd()
  {
    return active_.flat_cur_ + active_.flat_stride_;
  }
  
};


class ExecTraceable : public Traceable
{
 public:
  ExecTraceable(const std::string& nm = "") :
    Traceable(nm)
  {
    if (options::kShouldTraceExecution)
    {
      SetTraceLevel(options::kCurrentTraceLevel);
    }
    else
    {
      SetTraceLevel(0);
    }
  }
};

class Expression : public ExecTraceable
{
 public:
  virtual std::shared_ptr<timewhoop::Expression> ConvertExpression()
  {
    std::shared_ptr<timewhoop::Expression> expr = std::shared_ptr<timewhoop::Expression> (new timewhoop::Expression());
    return expr;
  }

  virtual int Evaluate(ExecutionContext& ctx)
  {
    return 0;
  }
};

std::list<int> EvaluateAll(const std::list<Expression*>& exprs, ExecutionContext& ctx);

std::string ShowIndices(const std::vector<int>& idxs);

class VarAccess : public Expression
{
 public:
  PrimVar& target_;
  
  VarAccess(PrimVar& v) : target_(v)
  {
  }

  virtual std::shared_ptr<timewhoop::Expression> ConvertExpression()
  {
    std::shared_ptr<timewhoop::Expression> expr = std::shared_ptr<timewhoop::Variable> (new timewhoop::Variable(timewhoop::Type::INTEGER, target_.name_ ));
    return expr;
  }
  
  virtual int Evaluate(ExecutionContext& ctx)
  {
    return target_.Access(ctx.FlatBegin(), ctx.FlatEnd());
  }
};


class TensorAccess : public Expression
{
 public:

  PrimTensor& target_;
  std::list<Expression*> idx_exprs_;
  int compute_level_ = 0;
  int tile_level_ = 0;
  int port_ = 0;
  
  TensorAccess(PrimTensor& v) : target_(v) {}

  TensorAccess(PrimTensor& v, const std::list<Expression*>& e, const int tile_level, const int compute_level, const int port = 0) : target_(v), idx_exprs_(e), tile_level_(tile_level), compute_level_(compute_level), port_(port) {}

  virtual std::shared_ptr<timewhoop::Expression> ConvertExpression()
  {
    std::shared_ptr<std::list<std::shared_ptr<timewhoop::Expression>>> converted_idx_exprs = std::shared_ptr<std::list<std::shared_ptr<timewhoop::Expression>>> (new std::list<std::shared_ptr<timewhoop::Expression>>());

    for(auto& idx_e : idx_exprs_)
    {
      std::shared_ptr<timewhoop::Expression> converted_idx_expr = std::shared_ptr<timewhoop::Expression>(idx_e->ConvertExpression());
      converted_idx_exprs->push_back(converted_idx_expr);
    }

    std::shared_ptr<timewhoop::Expression> converted_expr = std::shared_ptr<timewhoop::TensorAccess> (new timewhoop::TensorAccess(timewhoop::Type::INTEGER, target_.name_, converted_idx_exprs));
    return converted_expr;
  }
  
  virtual int Evaluate(ExecutionContext& ctx)
  {
    auto idxs = EvaluateAll(idx_exprs_, ctx);
    std::vector<int> v(idxs.begin(), idxs.end());
    compute_logs[compute_level_][ctx.CurrentSpatialPartition()]->LogInputTensor(target_.id_);
    return target_.Access(v, tile_level_, compute_level_, ctx.CurrentSpatialPartition(), ctx.NumSpatialPartitions(), port_);
  }
};


class BinaryOp : public Expression
{
 public:
  Expression* src1_;
  Expression* src2_;
  int (*op_)(const int& s1, const int& s2);
  
  BinaryOp(int (*o)(const int& s1, const int& s2), Expression* s1, Expression* s2) :
    src1_(s1), src2_(s2), op_(o)
  {
  }

  virtual std::shared_ptr<timewhoop::Expression> ConvertExpression()
  {
    timewhoop::BinaryOperator op;
    if(op_ == &PlusOp)
      op = timewhoop::BinaryOperator::PLUS;
    else if(op_ == &MinusOp)
      op = timewhoop::BinaryOperator::MINUS;
    else if(op_ == &MulOp)
      op = timewhoop::BinaryOperator::MULT;
    else if(op_ == &DivOp)
      op = timewhoop::BinaryOperator::DIV;
    else if(op_ == &EQOp)
      op = timewhoop::BinaryOperator::EQ;
    else if(op_ == &NEQOp)
      op = timewhoop::BinaryOperator::NEQ;
    else if(op_ == &GTEOp)
      op = timewhoop::BinaryOperator::GEQ;
    else if(op_ == &LTEOp)
      op = timewhoop::BinaryOperator::LEQ;
    else if(op_ == &GTOp) 
      op = timewhoop::BinaryOperator::GT;
    else if(op_ == &LTOp)
      op = timewhoop::BinaryOperator::LT;
    else if(op_ == &ANDOp)
      op = timewhoop::BinaryOperator::LOGICAL_AND;
    else if(op_ == &OROp)
      op = timewhoop::BinaryOperator::LOGICAL_OR;
    else if(op_ == &BWANDOp)
      op = timewhoop::BinaryOperator::BITWISE_AND;
    else if(op_ == &BWOROp)
      op = timewhoop::BinaryOperator::BITWISE_OR;
    else
      op = timewhoop::BinaryOperator::INVALID_OP;

    std::shared_ptr<timewhoop::Expression> expr_src1 = std::shared_ptr<timewhoop::Expression>(src1_->ConvertExpression());
    std::shared_ptr<timewhoop::Expression> expr_src2 = std::shared_ptr<timewhoop::Expression>(src2_->ConvertExpression());

    std::shared_ptr<timewhoop::Expression> expr = std::shared_ptr<timewhoop::BinaryOp> (new timewhoop::BinaryOp(op, expr_src1, expr_src2));
    return expr;
  }

  virtual int Evaluate(ExecutionContext& ctx)
  {
    int v1 = src1_->Evaluate(ctx);
    int v2 = src2_->Evaluate(ctx);
    return op_(v1, v2);
  }
};


class UnaryOp : public Expression
{
 public:
  Expression* src1_;
  int (*op_)(const int& s);

  UnaryOp(int (*o)(const int& s1), Expression* s1) :
    src1_(s1), op_(o)
  {
  }

  virtual std::shared_ptr<timewhoop::Expression> ConvertExpression()
  {
    timewhoop::UnaryOperator op;
    if(op_ == &POSTINCOp)
      op = timewhoop::UnaryOperator::POST_INCREMENT;
    else if(op_ == &PREINCOp)
      op = timewhoop::UnaryOperator::PRE_INCREMENT;
    else
      op = timewhoop::UnaryOperator::INVALID_OP;

    std::shared_ptr<timewhoop::Expression> expr_src = std::shared_ptr<timewhoop::Expression>(src1_->ConvertExpression());
    std::shared_ptr<timewhoop::Expression> expr = std::shared_ptr<timewhoop::UnaryOp> (new timewhoop::UnaryOp(op, expr_src));

    return expr;    
  }

  virtual int Evaluate(ExecutionContext& ctx)
  {
    int v1 = src1_->Evaluate(ctx);
    return op_(v1);
  }
};


class Constant : public Expression
{
 public:
  const int val_;
  
  Constant(const int& v) : val_(v) {}

  virtual std::shared_ptr<timewhoop::Expression> ConvertExpression()
  {
    std::shared_ptr<timewhoop::Expression> expr = std::shared_ptr<timewhoop::Integer>(new timewhoop::Integer(val_));
    return expr;
  }

  virtual int Evaluate(ExecutionContext& ctx)
  {
    return val_;
  }
};



class Statement : public ExecTraceable
{
 public:
  Statement* next_ = NULL;
  Statement* inner_ = NULL;
  Statement* else_ = NULL;
  std::vector<Statement*> partition_continuations_;

  virtual std::shared_ptr<timewhoop::Statement> ConvertStatement()
  {
    std::shared_ptr<timewhoop::Statement> converted_statement = std::shared_ptr<timewhoop::Statement>(new timewhoop::Statement);

    auto next_s = next_;
    if(next_s)
    {
      converted_statement->SetNextStmt(std::shared_ptr<timewhoop::Statement>(next_s->ConvertStatement()));
    }

    return converted_statement;
  }
  
  virtual Statement* Execute(ExecutionContext& ctx)
  {
    if (next_)
    {
      return next_->Execute(ctx);
    }
    else
    {
      // We did our last statment. We are done...
      return NULL;
    }
  }
  
  virtual void Init(ExecutionContext& ctx)
  {
    partition_continuations_.resize(ctx.NumSpatialPartitions(), NULL);
    if (inner_)
    {
      inner_->Init(ctx);
    }
    if (next_)
    {
      next_->Init(ctx);
    }
    if (else_)
    {
      else_->Init(ctx);
    }
  }
    
  bool AllPartitionsDone()
  {
    auto it = std::find_if(partition_continuations_.begin(),
                           partition_continuations_.end(),
                           [](Statement* s) { return s != NULL; });
    return it == partition_continuations_.end();
  }

  bool AnyPartitionIsPaused(int base, int num)
  {
    if (partition_continuations_.size() == 0) return false;
    for (int x = base; x < base + num; x++)
    {
      if (partition_continuations_[x] != NULL) return true;
    }
    return false;
  }
  
  Statement* ExecuteCurrentInner(ExecutionContext& ctx)
  {
    if (inner_)
    {
      partition_continuations_[ctx.CurrentSpatialPartition()] = inner_->Execute(ctx);
    }
    return partition_continuations_[ctx.CurrentSpatialPartition()];
  }
  
  Statement* ExecuteCurrentElse(ExecutionContext& ctx)
  {
    if (else_)
    {
      partition_continuations_[ctx.CurrentSpatialPartition()] = else_->Execute(ctx);
    }
    return partition_continuations_[ctx.CurrentSpatialPartition()];
  }

  bool CurrentIsPaused(ExecutionContext& ctx)
  {
    return partition_continuations_[ctx.CurrentSpatialPartition()];
  }

  Statement* ResumeCurrent(ExecutionContext& ctx)
  {
    assert(CurrentIsPaused(ctx));
    partition_continuations_[ctx.CurrentSpatialPartition()] = partition_continuations_[ctx.CurrentSpatialPartition()]->Execute(ctx);
    return partition_continuations_[ctx.CurrentSpatialPartition()];
  }

};

class VarAssignment : public Statement
{
 public:
  PrimVar& target_;
  Expression* body_ = NULL;
  
  VarAssignment(PrimVar& t) : 
    target_(t)
  {
  }
  
  VarAssignment(PrimVar& t, Expression* e) : 
    target_(t), body_(e)
  {
  }

  virtual std::shared_ptr<timewhoop::Statement> ConvertStatement()
  {
    std::shared_ptr<timewhoop::Container> converted_target = std::shared_ptr<timewhoop::Container>(target_.ConvertExpression());
    std::shared_ptr<timewhoop::Expression> converted_expr = std::shared_ptr<timewhoop::Expression>(body_->ConvertExpression());
    std::shared_ptr<timewhoop::Statement> converted_statement = std::shared_ptr<timewhoop::VariableAssignment> (new timewhoop::VariableAssignment(converted_target, converted_expr));

    auto next_s = next_;
    if(next_s)
    {
      converted_statement->SetNextStmt(std::shared_ptr<timewhoop::Statement>(next_s->ConvertStatement()));
    }

    return converted_statement;
  }

  virtual Statement* Execute(ExecutionContext& ctx)
  {
    T(4) << "Entering: variable assignment." << EndT;
    if (body_)
    {
      int res = body_->Evaluate(ctx);
      T(3) << "Updating variable " << target_.name_ << " value to: " << res << " (partitions: " << ctx.FlatBegin() <<  ".." << ctx.FlatEnd() - 1 << ")" << EndT;
      target_.Update(ctx.FlatBegin(), ctx.FlatEnd(), res);
    }
    T(4) << "Done: variable assignment." << EndT;
    return Statement::Execute(ctx);
  }

};


class TensorAssignment : public Statement
{
 public:
  PrimTensor& target_;
  std::list<Expression*> idx_exprs_;
  Expression* body_ = NULL;
  int tile_level_ = 0;
  int compute_level_ = 0;
  int port_ = 0;
  
  TensorAssignment(PrimTensor& t) : 
    target_(t)
  {
  }
  
  TensorAssignment(PrimTensor& t, const std::list<Expression*>& e) : 
    target_(t), idx_exprs_(e)
  {
  }

  TensorAssignment(PrimTensor& t, const std::list<Expression*> idx_e, Expression* body_e, const int& tile_level, const int& compute_level, const int& port = 0) : 
    target_(t), idx_exprs_(idx_e), body_(body_e), tile_level_(tile_level), compute_level_(compute_level), port_(port)
  {
  }

  virtual std::shared_ptr<timewhoop::Statement> ConvertStatement()
  {
    std::shared_ptr<timewhoop::Container> converted_target = std::shared_ptr<timewhoop::Tensor>(new timewhoop::Tensor(timewhoop::Type::INTEGER, target_.name_, target_.dim_sizes_));
    std::shared_ptr<timewhoop::Expression> converted_body = std::shared_ptr<timewhoop::Expression> (body_->ConvertExpression());
    std::shared_ptr<std::list<std::shared_ptr<timewhoop::Expression>>> converted_idx_exprs = std::shared_ptr<std::list<std::shared_ptr<timewhoop::Expression>>> (new std::list<std::shared_ptr<timewhoop::Expression>>());

    for(auto& idx_e : idx_exprs_)
    {
      std::shared_ptr<timewhoop::Expression> converted_idx_expr = std::shared_ptr<timewhoop::Expression>(idx_e->ConvertExpression());
      converted_idx_exprs->push_back(converted_idx_expr);
    }

    std::shared_ptr<timewhoop::Statement> converted_statement = std::shared_ptr<timewhoop::TensorAssignment>(new timewhoop::TensorAssignment(converted_target, converted_body, converted_idx_exprs));

    auto next_s = next_;
    if(next_s)
    {
      converted_statement->SetNextStmt(std::shared_ptr<timewhoop::Statement>(next_s->ConvertStatement()));
    }

    return converted_statement;
  }

  
  virtual Statement* Execute(ExecutionContext& ctx)
  {
    T(4) << "Entering: tensor assignment." << EndT;
    auto idxs = EvaluateAll(idx_exprs_, ctx);
    int res = body_->Evaluate(ctx);
    std::vector<int> vidxs(idxs.begin(), idxs.end());
    T(3) << "Updating tensor " << target_.name_ << " index: " << ShowIndices(vidxs) << " value to: " << res << EndT;
    target_.Update(vidxs, res, tile_level_, compute_level_, ctx.CurrentSpatialPartition(), ctx.NumSpatialPartitions(), port_);
    compute_logs[compute_level_][ctx.CurrentSpatialPartition()]->LogOutputTensor(1 << target_.id_);
    T(4) << "Done: tensor assignment." << EndT;
    return Statement::Execute(ctx);
  }
};

// TODO:  timewhoop will need support for While -ajaleel
class While : public Statement
{
 public:
  Expression* test_expr_;

  std::vector<bool> active_;
  
  While(Expression* test_e) :
    test_expr_(test_e)
  {
  }

  virtual std::shared_ptr<timewhoop::Statement> ConvertStatement()
  {
      assert(0);
  }
  
  virtual void Init(ExecutionContext& ctx)
  {
    active_.resize(ctx.NumSpatialPartitions(), false);
    Statement::Init(ctx);  
  }

  virtual Statement* Execute(ExecutionContext& ctx)
  {
    if (CurrentIsPaused(ctx))
    {
      T(4) << "Bypassing: while-loop." << EndT;
      if (ResumeCurrent(ctx))
      {
        // It didn't finish... try again in the future...
        T(4) << "Pausing: while-loop." << EndT;    
        return this;
      }
    }
    else
    {
      if (!active_[ctx.CurrentSpatialPartition()])
      {
        // We are at a beginning of a new body invocation.
        T(4) << "Starting: while-loop." << EndT;
      }

      // See if we need more iterations
      active_[ctx.CurrentSpatialPartition()] = test_expr_->Evaluate(ctx);

      if (active_[ctx.CurrentSpatialPartition()])
      {
        // Invoke the body (if any)
        T(4) << "Executing body: while-loop." << EndT;
        if (ExecuteCurrentInner(ctx))
        {
          // It didn't finish... try again in the future...
          T(4) << "Pausing: while-loop." << EndT;    
          return this;
        }
      }
    }
    
    if (active_[ctx.CurrentSpatialPartition()])
    {
      // Check for more iterations next time around.
      T(4) << "Pausing: while-loop." << EndT;    
      return this;
    }
    else
    {
      T(4) << "Done: while-loop." << EndT;
      // Move on to the next statement (if any)
      // Note: we purposely remove ourselves from the callstack in the sequential case.
      return Statement::Execute(ctx); 
    }
  }
};

// TODO:  timewhoop will need support for Flush -ajaleel
class Flush : public Statement
{
 public:
  // include tensor pointer
  PrimTensor*                                 tensor_ptr_;
  int                                         level_;
  std::shared_ptr<std::vector<std::shared_ptr<buff::BufferModel>>>  buffs_;

    Flush( ast::PrimTensor *ptr_in, int level, std::shared_ptr<std::vector<std::shared_ptr<buff::BufferModel>>>  buffs_in)
  {
      tensor_ptr_ = ptr_in;
      level_      = level;
      buffs_      = buffs_in;
  }

  virtual Statement* Execute(ExecutionContext& ctx)
  {
    T(4) << "Entering: Flush Buffer." << EndT;
    T(4) << "  Tensor: "<<tensor_ptr_->name_<< EndT;
    T(4) << "  Spatial Partition: "<<ctx.CurrentSpatialPartition()<< EndT;

    
    int num_buffs  = buffs_->size();

    auto buf_id  = whoop::buff::GetBufIndex( ctx.CurrentSpatialPartition(), num_buffs, ctx.NumSpatialPartitions() );
    
    (*buffs_)[buf_id]->FlushBuffet();
    
    T(4) << "Done: Flush Bufer." << EndT;
    return Statement::Execute(ctx);
  }
};

class TemporalFor : public Statement
{
 public:
  Statement* init_stmt_;
  Expression* test_expr_;
  Statement* incr_stmt_;
  std::vector<bool> active_;

  TemporalFor(Statement* init_s, Expression* test_e, Statement* incr_s) :
    init_stmt_(init_s),
    test_expr_(test_e),
    incr_stmt_(incr_s)
  {
  }

  virtual std::shared_ptr<timewhoop::Statement> ConvertStatement()
  {
    std::shared_ptr<timewhoop::Statement> converted_init_stmt = std::shared_ptr<timewhoop::Statement>(init_stmt_->ConvertStatement());
    std::shared_ptr<timewhoop::Expression> converted_test_expr = std::shared_ptr<timewhoop::Expression>(test_expr_->ConvertExpression());
    std::shared_ptr<timewhoop::Statement> converted_incr_stmt = std::shared_ptr<timewhoop::Statement>(incr_stmt_->ConvertStatement());
    std::shared_ptr<timewhoop::Statement> converted_body_stmt = std::shared_ptr<timewhoop::Statement>(this->inner_->ConvertStatement());
    std::shared_ptr<std::list<std::string>> buffered_tensor_name_list = std::make_shared<std::list<std::string>>();

    std::shared_ptr<timewhoop::Statement> converted_statement = std::shared_ptr<timewhoop::TemporalFor>(new timewhoop::TemporalFor(converted_init_stmt, converted_test_expr, converted_incr_stmt, converted_body_stmt, buffered_tensor_name_list));

    auto next_s = this->next_;
    if(next_s)
    {
      converted_statement->SetNextStmt(std::shared_ptr<timewhoop::Statement>(next_s->ConvertStatement()));
      next_s = next_s->next_;
    }

    return converted_statement;
  }
  
  virtual void Init(ExecutionContext& ctx)
  {
    active_.resize(ctx.NumSpatialPartitions(), false);
    Statement::Init(ctx);  
  }
  
  virtual Statement* Execute(ExecutionContext& ctx)
  {
    if (CurrentIsPaused(ctx))
    {
      T(4) << "Bypassing: temporal for-loop." << EndT;
      if (ResumeCurrent(ctx))
      {
        // It didn't finish... try again in the future...
        T(4) << "Pausing: temporal for-loop." << EndT;    
        return this;
      }
    }
    else
    {
      if (!active_[ctx.CurrentSpatialPartition()])
      {
        // We are at a beginning of a new body invocation.
        T(4) << "Starting: temporal for-loop." << EndT;
        init_stmt_->Execute(ctx); 
      }

      // See if we need more iterations
      active_[ctx.CurrentSpatialPartition()] = test_expr_->Evaluate(ctx);

      if (active_[ctx.CurrentSpatialPartition()])
      {
        // Invoke the body (if any)
        T(4) << "Executing body: temporal for-loop." << EndT;
        if (ExecuteCurrentInner(ctx))
        {
          // It didn't finish... try again in the future...
          T(4) << "Pausing: temporal for-loop." << EndT;    
          return this;
        }
      }
    }
    
    if (active_[ctx.CurrentSpatialPartition()])
    {
      T(4) << "Incrementing: temporal for-loop." << EndT;
      incr_stmt_->Execute(ctx);
      // Check for more iterations next time around.
      T(4) << "Pausing: temporal for-loop." << EndT;    
      return this;
    }
    else
    {
      T(4) << "Done: temporal for-loop." << EndT;
      // Move on to the next statement (if any)
      // Note: we purposely remove ourselves from the callstack in the sequential case.
      return Statement::Execute(ctx); 
    }
  }

  virtual void PrependToName(const std::string& p) 
  {
    init_stmt_->PrependToName(p);
    incr_stmt_->PrependToName(p);
    ExecTraceable::PrependToName(p);
  }
};


class SpatialFor : public Statement
{
 public:
  int num_partitions_;
  int flat_expansion_factor_ = 1;

  Statement* init_stmt_;
  Expression* test_expr_;
  Statement* incr_stmt_;
  

  SpatialFor(const int& num_parts, Statement* init_s, Expression* test_e, Statement* incr_s) :
    num_partitions_(num_parts),
    init_stmt_(init_s),
    test_expr_(test_e),
    incr_stmt_(incr_s)
  {
  }
 
  virtual std::shared_ptr<timewhoop::Statement> ConvertStatement()
  {
    std::shared_ptr<timewhoop::Statement> converted_init_stmt = std::shared_ptr<timewhoop::Statement>(init_stmt_->ConvertStatement());
    std::shared_ptr<timewhoop::Expression> converted_test_expr = std::shared_ptr<timewhoop::Expression>(test_expr_->ConvertExpression());
    std::shared_ptr<timewhoop::Statement> converted_incr_stmt = std::shared_ptr<timewhoop::Statement>(incr_stmt_->ConvertStatement());
    std::shared_ptr<timewhoop::Statement> converted_body_stmt = std::shared_ptr<timewhoop::Statement>(this->inner_->ConvertStatement());

    std::shared_ptr<timewhoop::Statement> converted_statement = std::shared_ptr<timewhoop::SpatialFor>(new timewhoop::SpatialFor(num_partitions_, converted_init_stmt, converted_test_expr, converted_incr_stmt, converted_body_stmt));

    auto next_s = next_;
    if(next_s)
    {
      converted_statement->SetNextStmt(std::shared_ptr<timewhoop::Statement>(next_s->ConvertStatement()));
    }

    return converted_statement;
  }
  
  virtual void Init(ExecutionContext& ctx)
  {
    // Purposely don't call superclass method.
    ctx.BeginSpatialPartitioning(num_partitions_, flat_expansion_factor_);
    
    partition_continuations_.resize(ctx.NumSpatialPartitions(), NULL);
    if (inner_)
    {
      inner_->Init(ctx);
    }
    ctx.EndSpatialPartitioning();
    if (next_)
    {
      next_->Init(ctx);
    }
  };
    

 
  virtual Statement* Execute(ExecutionContext& ctx)
  {
    // Make a copy of the existing context so that we can use it for statements in
    // the loop itself. This is because the s_for loop statements themselves are
    // non-parallel.
    ExecutionContext old_ctx = ctx;
    
    ctx.BeginSpatialPartitioning(num_partitions_, flat_expansion_factor_);
    
    // Remember the starting partition.
    int base_part = ctx.CurrentSpatialPartition();
    bool did_something = false;

    if (!AnyPartitionIsPaused(base_part, num_partitions_))
    {
      // We are a fresh loop.
      T(4) << "Entering: spatial for-loop: " << num_partitions_ << EndT;

      // Do all the statements in sequence, remembering where to continue them if needed.
      for (init_stmt_->Execute(old_ctx); test_expr_->Evaluate(old_ctx); incr_stmt_->Execute(old_ctx))
      {
        if (inner_)
        {      
          T(4) << "Entering: spatial partition: " << ctx.CurrentSpatialPartition() << EndT;
          ExecuteCurrentInner(ctx);
          // Never pause here.
          did_something = true;
        }
        ctx.NextSpatialPartition();
      }
    }
    else
    {
      // Go over all the partitions instead of just the paused ones because
      // they may reference the variables altered by this for-loop.
      T(4) << "Re-entering: spatial for-loop: " << num_partitions_ << EndT;
      for (init_stmt_->Execute(old_ctx); test_expr_->Evaluate(old_ctx); incr_stmt_->Execute(old_ctx))
      {
        // Only actually descend into the paused ones.
        Statement* cont = partition_continuations_[ctx.CurrentSpatialPartition()];
        if (cont != NULL)
        {
          did_something = true;
          T(4) << "Resuming: spatial partition: " << ctx.CurrentSpatialPartition() << EndT;
          ResumeCurrent(ctx);
          // Never pause here.
        }
        ctx.NextSpatialPartition();
      }
    }
    
    assert(did_something);

    // If any partitions still have unfinished work, then come back to us.
    if (AnyPartitionIsPaused(base_part, num_partitions_))
    {
      T(4) << "Pausing: spatial for-loop." << EndT;
      // Pop the spatial context stack.
      ctx.EndSpatialPartitioning();
      return this;
    }

    T(4) << "Done: spatial for-loop." << EndT;
    // Pop the spatial context stack.
    ctx.EndSpatialPartitioning();
    // Purposely remove ourselves from the callstack by going on to the next
    // sequential statement.
    return Statement::Execute(ctx);
  }

  virtual void PrependToName(const std::string& p) 
  {
    init_stmt_->PrependToName(p);
    incr_stmt_->PrependToName(p);
    ExecTraceable::PrependToName(p);
  }
};

using EndLoop = Statement;


class If : public Statement
{
 public:
  Expression* test_expr_;
  double exec_probability_ = 1.0; //For timewhoop

  If(Expression* test_e, double exec_prob) :
    test_expr_(test_e),
    exec_probability_(exec_prob)
  {
  }
  
  If(Expression* test_e) :
    test_expr_(test_e)
  {
  }

  virtual std::shared_ptr<timewhoop::Statement> ConvertStatement()
  {
    std::shared_ptr<timewhoop::Expression> converted_test_expr = std::shared_ptr<timewhoop::Expression>(test_expr_->ConvertExpression());
    std::shared_ptr<timewhoop::Statement> converted_body_stmt = std::shared_ptr<timewhoop::Statement>(this->inner_->ConvertStatement());

    std::shared_ptr<timewhoop::Statement> converted_statement = std::shared_ptr<timewhoop::If>(new timewhoop::If(converted_test_expr, converted_body_stmt,exec_probability_));


    auto next_s = this->next_;
    if(next_s)
    {
      converted_statement->SetNextStmt(std::shared_ptr<timewhoop::Statement>(next_s->ConvertStatement()));
    }

    return converted_statement;
  }

  virtual Statement* Execute(ExecutionContext& ctx)
  {
  
    if (CurrentIsPaused(ctx))
    {
      T(4) << "Bypassing: dynamic if." << EndT;
      if (ResumeCurrent(ctx))
      {
        // It didn't finish... try again in the future...
        T(4) << "Pausing: dynamic if." << EndT;    
        return this;
      }
    }
    else
    {
      T(4) << "Entering: dynamic if." << EndT;
      int res = test_expr_->Evaluate(ctx);

      if (res)
      {
        T(4) << "    (condition evaluated to true)" << EndT;
        if (ExecuteCurrentInner(ctx))
        {
          // It didn't finish... try again in the future...
          T(4) << "Pausing: dynamic if." << EndT;    
          return this;
        }
      }
      else
      {
        T(4) << "    (condition evaluated to false)" << EndT;
        if (ExecuteCurrentElse(ctx))
        {
          // It didn't finish... try again in the future...
          T(4) << "Pausing: dynamic if." << EndT;    
          return this;
        }
      }
    }
  
    // If we got here, nothing paused...
    T(4) << "Done: dynamic if." << EndT;

    // Note: we explicitly don't call Statement::Execute(ctx) if we have an else.
    // Instead we jump to whatever is after the else (even if the else itself
    // wasn't executed).
    if (else_)
    {
      if (else_->next_)
      {
        return else_->next_->Execute(ctx);
      }
    }
    else
    {
      return Statement::Execute(ctx);
    }
  }
};


class Else : public Statement
{
 public:


  virtual std::shared_ptr<timewhoop::Statement> ConvertStatement()
  {
    std::shared_ptr<timewhoop::Statement> converted_body_stmt = std::shared_ptr<timewhoop::Statement>(this->inner_->ConvertStatement());

    std::shared_ptr<timewhoop::Statement> converted_statement = std::shared_ptr<timewhoop::Else>(new timewhoop::Else(converted_body_stmt));

    auto next_s = this->next_;
    if(next_s)
    {
      converted_statement->SetNextStmt(std::shared_ptr<timewhoop::Statement>(next_s->ConvertStatement()));
    }

    return converted_statement;
  }

  virtual Statement* Execute(ExecutionContext& ctx)
  {
    T(4) << "Entering: else." << EndT;
    if (inner_)
    {
      return ExecuteCurrentInner(ctx);
    }
    T(4) << "Done: else." << EndT;
    // Note: we explicitly don't call Statement::Execute(ctx) here.
    // It will be called through If...
    return NULL; // We are done.
  }
};

}  // namespace ast

}  // namespace whoop

#endif /* WHOOP_ABSTRACT_SYNTAX_HPP_ */
