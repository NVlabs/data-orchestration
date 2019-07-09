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

// top_stats
// Global variable to serve as top-of-tree for stats dump
extern StatsCollection top_stats;

extern std::vector<std::vector<activity::ComputeEngineLog*>> compute_logs;
void InitializeComputeLogs(const std::vector<int>& flattened_tile_level_spatial_expansions);
void LogComputeActivity(std::ostream& ostr);
void LogComputeTopology(std::ostream& ofile, int num_tensors);

namespace buff
{

int GetBufIndex( int curr_spatial_idx, int buffs_at_level, int max_spatial_partitions );

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
    int index_ = 0;
    std::bitset<64> receiver_mask_ = 0;
    std::bitset<64> updater_mask_ = 0;
    bool caused_evict_ = false;
    bool is_dirty_ = false;
  };
  activity::BuffetCommandLog command_log_;
  activity::PatternGeneratorLog read_pgen_log_;
  activity::PatternGeneratorLog update_pgen_log_;
  activity::PatternGeneratorLog destination_pgen_log_;
  activity::PatternGeneratorLog updaters_pgen_log_;
  activity::ShrinkPatternGeneratorLog shrink_pgen_log_;
  std::deque<std::pair<int, BuffetLogInfo>> coalescing_buffer_;
  int coalescing_window_size_ = 32; // XXX TODO: MAKE DYNAMIC
  
  int num_fills_ = 0;
  int num_shrinks_ = 0;
  
  bool CheckCoalescing(int address, int requestor_idx, bool is_update, int* entry = NULL)
  {
    // Purposely search backwards to find the most recent entry.
    // This way even if the previous entry wasn't evicted will we find the latest.
    auto it = coalescing_buffer_.rbegin();
    for ( ; it != coalescing_buffer_.rend(); it++)
    {
      if ((*it).first == address) break;
    }
    if (it != coalescing_buffer_.rend())
    {
      assert(requestor_idx < 64);
      // Don't coalesce your own requests. Can't multicast to yourself!
      if (!is_update && (*it).second.receiver_mask_[requestor_idx])
      {
        return false;
      }
      (*it).second.receiver_mask_[requestor_idx] = true;
      if (is_update)
      {
        // For now, don't coalesce more than one update per updater on at a time....
        if ((*it).second.updater_mask_[requestor_idx])
        {
          return false;
        }
        (*it).second.updater_mask_[requestor_idx] = true;
        (*it).second.is_dirty_ = true;
      }
      if (entry)
      {
        *entry = (*it).second.index_;
      }
      return true;
    }
    else
    {
      return false;
    }
  }
  
  void LogAccess(int address, int curr_index, int requestor_idx, bool caused_evict)
  {
    // Keep track of last accessed address.
    BuffetLogInfo info;
    info.index_ = curr_index;
    assert(requestor_idx < 64);
    info.receiver_mask_[requestor_idx] = true;
    info.caused_evict_ = caused_evict;
    if (coalescing_buffer_.size() == coalescing_window_size_)
    {
      // Evict the oldest from the window.
      RecordInfo(coalescing_buffer_.front().second);
      coalescing_buffer_.pop_front();
    }
    coalescing_buffer_.push_back(std::make_pair(address, info));
  }

  void LogUpdate(int address, int curr_index, int requestor_idx, bool caused_evict)
  {
    T(4) << "Logging implicit RMW to address: " << address << " by requestor: " << requestor_idx << EndT;
    // Keep track of last accessed address.
    BuffetLogInfo info;
    info.index_ = curr_index;
    assert(requestor_idx < 64);
    info.receiver_mask_[requestor_idx] = true;
    info.updater_mask_[requestor_idx] = true;
    info.caused_evict_ = caused_evict;
    info.is_dirty_ = true;
    if (coalescing_buffer_.size() == coalescing_window_size_)
    {
      // Evict the oldest from the window.
      RecordInfo(coalescing_buffer_.front().second);
      coalescing_buffer_.pop_front();
    }
    coalescing_buffer_.push_back(std::make_pair(address, info));
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
      RecordInfo((*it).second);
    }
    coalescing_buffer_.clear();

    // Shrink off anything remaining in the buffet.
    if (num_fills_ > num_shrinks_)
    {
      if (is_modified)
      {
        command_log_.SetModified();
      }
      command_log_.Shrink(num_fills_ - num_shrinks_);
      shrink_pgen_log_.Shrink(num_fills_ - num_shrinks_);
    }
  }
  
  void RecordInfo(const BuffetLogInfo& info)
  {
    if (!options::kShouldLogActivity) return;

    // Record any shrinks.
    if (info.caused_evict_)
    {
      if (info.is_dirty_)
      {
        command_log_.SetModified(); // XXX This is not quite right... it should be that the evicted line was dirty... this is a bit of a hack...
      }
      command_log_.Shrink();
      shrink_pgen_log_.Shrink();
      num_shrinks_++;
    }

    // Record accces.
    command_log_.Read(info.is_dirty_);
    read_pgen_log_.Send(command_log_.GetRelativeIndex(info.index_));
    destination_pgen_log_.Send(info.receiver_mask_.to_ulong());

    // Record any updates
    if (info.is_dirty_) {
      // NOTE: Update indexes are purposely absolute in the buffer.
      update_pgen_log_.Send(info.index_);
      // Minor hack: the datapath doesn't show up as a source.
      // So if the mask is 0, make it 1.
      int final_mask = info.updater_mask_.to_ulong();
      updaters_pgen_log_.Send(final_mask == 0 ? 1 : final_mask);
    }
  }
 public:
  
  std::shared_ptr<BufferModel> backing_buffer_ = NULL;
  std::vector<std::shared_ptr<BufferModel>> fronting_buffers_;

  int level_;
  // Individual tensors may not participate in all global tile levels.
  // E.g. the L2 weights tile may be the last tile of weights.
  int starting_global_tile_level_;
  int ending_global_tile_level_;
  std::vector<int> num_datapaths_;
  int local_spatial_idx_;
  
  explicit BufferModel(int level, int starting_tile_level, int local_spatial_idx, const std::string& nm = "") :
    StatsCollection(nm, &top_stats),
    TraceableBuffer(nm),
    level_(level),
    starting_global_tile_level_(starting_tile_level),
    ending_global_tile_level_(starting_tile_level + 1),
    num_datapaths_(1, 1),
    local_spatial_idx_(local_spatial_idx),
    fronting_buffers_{}
  {
  }
  
  void LogActivity(std::ostream& ostr)
  {
    command_log_.Dump(ostr, Traceable::GetName() + "_commands", "symphony::modules::LocalBuffet");
    ostr << "," << std::endl;
    read_pgen_log_.Dump(ostr, Traceable::GetName() + "_reads", "symphony::modules::LocalPatternGenerator");
    ostr << "," << std::endl;
    destination_pgen_log_.Dump(ostr, Traceable::GetName() + "_read_destinations", "symphony::modules::LocalPatternGenerator");
    ostr << "," << std::endl;
    update_pgen_log_.Dump(ostr, Traceable::GetName() + "_updates", "symphony::modules::LocalPatternGenerator");
    ostr << "," << std::endl;
    updaters_pgen_log_.Dump(ostr, Traceable::GetName() + "_update_sources", "symphony::modules::LocalPatternGenerator");
    ostr << "," << std::endl;
    shrink_pgen_log_.Dump(ostr, Traceable::GetName() + "_shrinks", "symphony::modules::LocalPatternGenerator");
  }

  void LogTopologyModule(std::ostream& ostr)
  {  
    ostr << "  - type: module" << std::endl;
    ostr << "    class: symphony::modules::BuffetComplex" << std::endl;
    ostr << "    base_name: " << Traceable::GetName() << std::endl;
    ostr << "    configuration:" << std::endl;
    ostr << "      knobs_use_prefix: false" << std::endl;
    ostr << "      knobs:" << std::endl;
    ostr << "        - \"num_receivers = " << GetNumReceivers() << "\""  << std::endl;
    ostr << "        - \"num_updaters = " << GetNumReceivers() << "\""  << std::endl;
    ostr << std::endl;
  }

  void LogTopologyConnections(std::ostream& ostr, int id, int local_spatial_idx, int max_tile_level)
  {
    for (int dst_idx = 0; dst_idx < fronting_buffers_.size(); dst_idx++)
    {
      // The buffer feeds another (usually smaller) buffer.
      ostr << "      - src:" << std::endl;
      ostr << "        - name: " << Traceable::GetName() << std::endl;
      ostr << "          port-name: read_data_out_" << dst_idx << std::endl;
      ostr << "        dst:" << std::endl;
      ostr << "          - name: " << fronting_buffers_[dst_idx]->Traceable::GetName() << std::endl;
      ostr << "            port-name: fill_data_in_0" << std::endl;
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
    int end_level = ending_global_tile_level_ == starting_global_tile_level_ ? max_tile_level : ending_global_tile_level_;
    for (int tile_level = starting_global_tile_level_; tile_level < end_level; tile_level++)
    {
      int num_dpaths = num_datapaths_[tile_level - starting_global_tile_level_];
      for (int x = 0; x < num_dpaths; x++)
      {
        int dp_idx = local_spatial_idx * num_dpaths + x;
        ostr << "      - src:" << std::endl;
        ostr << "        - name: " << Traceable::GetName() << std::endl;
        ostr << "          port-name: read_data_out_" << fronting_buffers_.size() + x << std::endl;
        ostr << "        dst:" << std::endl;
        ostr << "        - name: compute_engine_" << tile_level << "_" << dp_idx << std::endl;
        ostr << "          port-name: input_" << id << std::endl; // Note: This index could be smarter.
        ostr << "      - src:" << std::endl;
        ostr << "        - name: compute_engine_" << tile_level << "_" << dp_idx << std::endl;
        ostr << "          port-name: output_" << id << std::endl;
        ostr << "        dst:" << std::endl;
        ostr << "        - name: " << Traceable::GetName() << std::endl;
        ostr << "          port-name: update_data_in_" << fronting_buffers_.size() + x << std::endl;
      }
    }
  }

  void Resize(int size)
  {
    command_log_.Resize(size);
  }
 
  void ExpandDatapaths(int global_tile_level, int num)
  {
    int tile_level = global_tile_level - starting_global_tile_level_;
    if (tile_level >= num_datapaths_.size())
    {
      // Any unmentioned layers must have 1 datapath.
      num_datapaths_.resize(tile_level+1, 1);
    }
    num_datapaths_[tile_level] *= num;
  }

  int GetNumReceivers()
  {
    int flat_datapaths = std::accumulate(num_datapaths_.begin(), num_datapaths_.end(), 0);
    return fronting_buffers_.size() + flat_datapaths;
  }
 
  virtual void Access(int address, int requestor_idx) = 0;
  virtual void Update(int address, int requestor_idx) = 0;
  virtual void DrainAll() = 0;
  virtual void FlushBuffet() = 0;
  virtual void FixupLevelNumbering(int final_num_levels) = 0;
  virtual void SetBufferWidth(int width) = 0;
};

class OffChipBufferModel : public BufferModel
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
      IncrStat("Offchip row buffer hit");
      T(4) << "Row buffer hit for read to address " << address << EndT;
    }
    else
    {
      rowbuffer_active_idx_ = addr_idx;
      IncrStat("Offchip row buffer miss");
      T(4) << "Row buffer miss for read to address " << address << EndT;
    }
  }

 public:
  int rowbuffer_active_idx_ = -1;
  int rowbuffer_width_ = 16; //This is a default value; users can modify it via "rowbuffer_width_(name)" option
  
  OffChipBufferModel(const std::string& nm, int size) : 
    BufferModel(0, 0, 0, nm)
  {
    AddOption(&rowbuffer_width_, "rowbuffer_width_" + nm, "The width of row buffer in DRAM (offchip memory)" + nm);
    command_log_.Init(size, 0, false);
  }

  virtual void Access(int address, int requestor_idx)
  {
    
    if (CheckCoalescing(address, requestor_idx, false))
    {  
      IncrStat("Offchip multicasts");
      T(4) << "Read (coalesced), address: " << address << " by requestor: " << requestor_idx  << EndT;
    }
    else
    {
      CheckRowBufferHit(address);
      IncrStat("Offchip reads");
      T(4) << "Read, address: " << address << " by requestor: " << requestor_idx << EndT;
      LogAccess(address, address, requestor_idx, false);
    }
  }
  
  virtual void Update(int address, int requestor_idx)
  {
    if (CheckCoalescing(address, requestor_idx, true))
    {  
      IncrStat("Offchip multi-reduces");
      T(4) << "Update (coalesced), address: " << address << " by requestor: " << requestor_idx  << EndT;
    }
    else
    {
      CheckRowBufferHit(address);

      IncrStat("Offchip updates");
      LogUpdate(address, address, requestor_idx, false);
      T(4) << "Update, address: " << address  << " by requestor: " << requestor_idx << EndT;
    }
  }

  virtual void FlushBuffet()
  {
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
  std::unordered_map<int, int> presence_info_;
  std::vector<int> cache_;
  std::vector<int> modified_; // NOTE: We had some problems with std::vector<bool>
  int head_ = 0;
  int occupancy_ = 0;
  int size_;
  int granularity_;
  int buffet_size_;
  int extra_buffering_ = 0;
    
  explicit AssociativeBufferModel(int size, int level, int starting_tile_level, int local_spatial_idx, const std::string& nm = "", int shrink_granularity = 0, int granularity = 1) : 
    cache_(size/granularity), 
    modified_(size/granularity), 
    size_(size/granularity), 
    BufferModel(level, starting_tile_level, local_spatial_idx, nm), 
    granularity_(granularity), 
    buffet_size_(size)
  {
    command_log_.Init(size/granularity, extra_buffering_/granularity);
    command_log_.SetShrinkGranularity(shrink_granularity == 0 ? size/granularity : shrink_granularity);
    shrink_pgen_log_.SetShrinkGranularity(shrink_granularity == 0 ? size/granularity : shrink_granularity);
  }
  
  void ModIncr(int& v)
  {
    if (v + 1 == size_)
    {
      v = 0;
    }
    else
    {
      v++;
    }
  }

  int ModAdd(const int& x, const int& y)
  {
    int res = (x + y) % size_;
    return res;
  }

    // Note that the address coming in is just the index within the tensor, not
    // the actual address
  virtual void Access(int addr, int requestor_idx)
  {
    int address = addr - (addr % granularity_);
    
    if (CheckCoalescing(address, requestor_idx, false))
    {
      IncrStat("L" + std::to_string(level_) + " multicasts");
      T(4) << "Read (coalesced), address: " << address << " by requestor: " << requestor_idx << EndT;
      return;
    }
    
    IncrStat("L" + std::to_string(level_) + " reads");
    T(4) << "Read, address: " << address << " by requestor: " << requestor_idx << EndT;

    bool found        = false;
    bool caused_evict = false;
    int  curr_index = -1;
    
    auto it = presence_info_.find(address);
    if (it != presence_info_.end())
    {
      found = true;
      curr_index = it->second;
      T(4) << "    (Already present in buffer at index: " << curr_index << ")" << EndT;
    }
    
    if (!found)
    {
      IncrStat("L" + std::to_string(level_) + " fills");
      num_fills_++;
      backing_buffer_->Access(address, local_spatial_idx_);
      if (occupancy_ == size_)
      {
        if (modified_[head_] == 1)
        {
          backing_buffer_->Update(cache_[head_], local_spatial_idx_);
        }

        // remove victim from unordered_map
        presence_info_.erase(cache_[head_]);

        ModIncr(head_);
        occupancy_--;
        caused_evict = true;
      }
      int tail = ModAdd(head_, occupancy_);
      T(4) << "    (Filling from next level to index: " << tail << ", caused_evict:" << caused_evict << ")" << EndT;

      cache_[tail] = address;
      modified_[tail] = 0;
      occupancy_++;

      // insert into unordered_map, and store pointer to modified should we update
      presence_info_[address] = tail;
      curr_index             = tail;
    }
    LogAccess(address, curr_index, requestor_idx, caused_evict);
  }

  // Note that the address coming in is just the index within the tensor, not
  // the actual address
  virtual void Update(int addr, int requestor_idx)
  {
    int address = addr - (addr % granularity_);

    int entry;
    if (CheckCoalescing(address, requestor_idx, true, &entry))
    {
      IncrStat("L" + std::to_string(level_) + " multi-reduces");
      T(4) << "Update (coalesced), address: " << address << " by requestor: " << requestor_idx << EndT;
      modified_[entry] = true;
      return;
    }
    
    IncrStat("L" + std::to_string(level_) + " updates");
    T(4) << "Update, address: " << address << " by requestor: " << requestor_idx << EndT;

    bool found        = false;
    bool caused_evict = false;
    int  curr_index = -1;

    auto it = presence_info_.find(address);
    if (it != presence_info_.end())
    {
      curr_index = it->second;
      modified_[curr_index] = 1;
      T(4) << "    (Already present in buffer at index: " << curr_index << ")" << EndT;
      found = true;
    }

    if (!found)
    {
      if (occupancy_ == size_)
      {
        if (modified_[head_] == 1)
        {
          backing_buffer_->Update(cache_[head_], local_spatial_idx_);
        }

        // remove victim from unordered_map
        presence_info_.erase(cache_[head_]);

        ModIncr(head_);
        occupancy_--;
        caused_evict = true;
      }

      int tail = ModAdd(head_, occupancy_);
      T(4) << "    (Allocating on write to index: " << tail << ", caused_evict:" << caused_evict << ")" << EndT;
      cache_[tail] = address;
      modified_[tail] = 1;
      occupancy_++;

      // insert into unordered_map, and store pointer to modified should we update
      presence_info_[address] = tail;
      curr_index              = tail;
    }
    LogUpdate(address, curr_index, requestor_idx, caused_evict);
    command_log_.SetModified();
  }
        
  virtual void DrainAll()
  {
    for (int x = 0; x < occupancy_; x++)
    {
      int entry = ModAdd(head_, x);
      if (modified_[entry] == 1)
      {
        backing_buffer_->Update(cache_[entry], local_spatial_idx_);

        // since it has been drained, no longer modified
        modified_[entry] = 0;
      }
    }
    occupancy_ = 0;
    DrainLog();
  }

  virtual void FlushBuffet()
  {
      presence_info_.clear();
      DrainAll();
  }
  
  virtual void FixupLevelNumbering(int final_num_levels)
  {
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
  
  int val_ = 0xAAAAAAAA;
  
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
  
  void Update(const int& new_val)
  {
    IncrStat("Var updates");
    val_ = new_val;
  }
  
  int Access()
  {
    IncrStat("Var reads");
    return val_;
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

  void Update(const std::vector<int>& idxs, const int& new_val, const int& tile_level, const int& spatial_part_idx = 0, const int& max_spatial_partitions = 0,  const int& port_idx = 0)
  {
    is_updated_dynamically_ = true;
    // TODO: better error messages.
    int buffer_spatial_part_idx = whoop::buff::GetBufIndex( spatial_part_idx, (*buffer_levels_[port_idx])[tile_level]->size(), max_spatial_partitions );
    int local_spatial_idx = spatial_part_idx % (max_spatial_partitions / (*buffer_levels_[port_idx])[tile_level]->size());

    IncrStat("Tensor updates");
    UINT64 idx = FlattenIndices(idxs);
    vals_[idx] = new_val;
    auto buffer_to_access = (*(*buffer_levels_[port_idx])[tile_level])[buffer_spatial_part_idx];
    // Datapath access ids are offset by the fronting buffers.
    buffer_to_access->Update(idx, buffer_to_access->fronting_buffers_.size() + local_spatial_idx);
  }
  
  int Access(const std::vector<int>& idxs, const int& tile_level, const int& spatial_part_idx = 0, const int& max_spatial_partitions = 0, const int& port_idx = 0)
  {
    int buffer_spatial_part_idx = whoop::buff::GetBufIndex( spatial_part_idx, (*buffer_levels_[port_idx])[tile_level]->size(), max_spatial_partitions );      
    int local_spatial_idx = spatial_part_idx % (max_spatial_partitions / (*buffer_levels_[port_idx])[tile_level]->size());

    IncrStat("Tensor reads");
    UINT64 idx = FlattenIndices(idxs);
    auto buffer_to_access = (*(*buffer_levels_[port_idx])[tile_level])[buffer_spatial_part_idx];
    // Datapath access ids are offset by the fronting buffers.
    buffer_to_access->Access(idx, buffer_to_access->fronting_buffers_.size() + local_spatial_idx);
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

  void PrimPushBack(const int& val)
  {
    vals_.push_back(val);
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
    // Tell the offchip buffer the new size.
    (*buffer_levels_[0])[0]->at(0)->Resize(s);
  }
  
  // Used when loading tensors from files to cleanup buffer model.
  void FixupSize()
  {
    // Tell the offchip buffer the new size.
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

  void ExpandDatapaths(int global_tile_level, int num)
  {
    for (auto port_it : buffer_levels_)
    {
      auto last_level = (*port_it).back();
      for (auto buf_it : *last_level)
      {
        (*buf_it).ExpandDatapaths(global_tile_level, num);
      }
    }
  }

  void DrainAll()
  {
    // Purposely iterate ports backwards since offchip is always port 0.
    for (auto port_rit = buffer_levels_.rbegin(); port_rit != buffer_levels_.rend(); port_rit++)
    {
      // Purposely iterate from closer levels on down, for maximum locality.
      // Skip level 0 since it is the same offchip for each port and handled specially
      for (auto level_rit = (*port_rit)->rbegin(); level_rit != std::prev((*port_rit)->rend()); level_rit++)
      {
        for (auto buf : *(*level_rit))
        {
          buf->DrainAll();
        }
      }
    }
    // Every tensor has an offchip.
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
      // Skip the offchip level as it is repeated across ports.
      for (auto level_it = std::next(port->begin()); level_it != port->end(); level_it++)
      {
        for (auto& buff : *(*level_it))
        {
          buff->LogActivity(ostr);
          ostr << "," << std::endl;
        }
      }
    }
    // Every tensor has an offchip.
    (*buffer_levels_[0])[0]->at(0)->LogActivity(ostr);
  }

  void LogTopologyModules(std::ostream& ostr)
  {
    for (auto& port : buffer_levels_)
    {
      // Skip the offchip level as it is repeated across ports.
      for (auto level_it = std::next(port->begin()); level_it != port->end(); level_it++)
      {
        for (auto& buff : *(*level_it))
        {
          buff->LogTopologyModule(ostr);
        }
      }
    }
    // Every tensor has an offchip.
    (*buffer_levels_[0])[0]->at(0)->LogTopologyModule(ostr);
  }

  void LogTopologyConnections(std::ostream& ostr, int max_tile_level)
  {
    for (auto& port : buffer_levels_)
    {
      // Skip the offchip level as it is repeated across ports.
      for (auto level_it = std::next(port->begin()); level_it != port->end(); level_it++)
      {
        int local_spatial_idx = 0;
        for (auto& buff : *(*level_it))
        {
          buff->LogTopologyConnections(ostr, id_, local_spatial_idx, max_tile_level);
          local_spatial_idx++;
        }
      }
    }
    // Every tensor has an offchip.
    (*buffer_levels_[0])[0]->at(0)->LogTopologyConnections(ostr, id_, 0, max_tile_level);
  }
};

  
class ExecutionContext
{
 public:
  std::deque<std::pair<int, int>> partition_stack_{};
  // These are stored flattened (meaning expanded across all nested s_fors)
  int current_spatial_partition_ = 0;
  int max_spatial_partitions_ = 1;
    
  void BeginSpatialPartitioning(int num_partitions)
  {
    partition_stack_.push_back({current_spatial_partition_, num_partitions});
    // Find the absolute id in the space using an equation of this form:
    // x3 * X2 * X1 * X0 + x2 * X1 * X0 + x1 * X0 + x0
    // Note that x0 == 0 here since we are starting this dimension.
    int current_base = 0;
    for (auto it = partition_stack_.begin(); it != partition_stack_.end(); it++)
    {
      int expansion_factor = (*it).second;
      for (auto it2 = it+1; it2 != partition_stack_.end(); it2++)
      {
        expansion_factor *= (*it2).second;
      }
      current_base += (*it).first * expansion_factor;
    }
    if( num_partitions == 0 ) 
    {
        std::cout<<"Why is num partitions zero?"<<std::endl;
        exit(0);
    }
    max_spatial_partitions_ *= num_partitions;
    current_spatial_partition_ = current_base;
  }
  
  void EndSpatialPartitioning()
  {
    current_spatial_partition_ = partition_stack_.back().first;
    max_spatial_partitions_ /= partition_stack_.back().second;
    partition_stack_.pop_back();
  }

  void NextSpatialPartition()
  {
    current_spatial_partition_++;
  } 

  int CurrentSpatialPartition()
  {
    return current_spatial_partition_;
  }

  int MaxSpatialPartitions()
  {
    return max_spatial_partitions_;
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
    return target_.Access();
  }
};


class TensorAccess : public Expression
{
 public:

  PrimTensor& target_;
  std::list<Expression*> idx_exprs_;
  int tile_level_ = 0;
  int port_ = 0;
  
  TensorAccess(PrimTensor& v) : target_(v) {}

  TensorAccess(PrimTensor& v, const std::list<Expression*>& e, const int tile_level, const int port = 0) : target_(v), idx_exprs_(e), tile_level_(tile_level), port_(port) {}

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
    compute_logs[tile_level_][ctx.CurrentSpatialPartition()]->LogInputTensor(target_.id_);
    return target_.Access(v, tile_level_, ctx.CurrentSpatialPartition(), ctx.MaxSpatialPartitions(), port_);
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
  
  virtual void Execute(ExecutionContext& ctx)
  {
    if (next_)
    {
      next_->Execute(ctx);
    }
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

  virtual void Execute(ExecutionContext& ctx)
  {
    T(4) << "Entering: variable assignment." << EndT;
    if (body_)
    {
      int res = body_->Evaluate(ctx);
      T(3) << "Updating variable " << target_.name_ << " value to: " << res << EndT;
      target_.Update(res);
    }
    T(4) << "Done: variable assignment." << EndT;
    Statement::Execute(ctx);
  }
};


class TensorAssignment : public Statement
{
 public:
  PrimTensor& target_;
  std::list<Expression*> idx_exprs_;
  Expression* body_ = NULL;
  int tile_level_ = 0;
  int port_ = 0;
  
  TensorAssignment(PrimTensor& t) : 
    target_(t)
  {
  }
  
  TensorAssignment(PrimTensor& t, const std::list<Expression*>& e) : 
    target_(t), idx_exprs_(e)
  {
  }

  TensorAssignment(PrimTensor& t, const std::list<Expression*> idx_e, Expression* body_e, const int& tile_level, const int& port = 0) : 
    target_(t), idx_exprs_(idx_e), body_(body_e), tile_level_(tile_level), port_(port)
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

  
  virtual void Execute(ExecutionContext& ctx)
  {
    T(4) << "Entering: tensor assignment." << EndT;
    auto idxs = EvaluateAll(idx_exprs_, ctx);
    int res = body_->Evaluate(ctx);
    std::vector<int> vidxs(idxs.begin(), idxs.end());
    T(3) << "Updating tensor " << target_.name_ << " index: " << ShowIndices(vidxs) << " value to: " << res << EndT;
    target_.Update(vidxs, res, tile_level_, ctx.CurrentSpatialPartition(), ctx.MaxSpatialPartitions(), port_);
    // TODO: Capture multicasting to datapaths.
    compute_logs[tile_level_][ctx.CurrentSpatialPartition()]->LogOutputTensor(1 << target_.id_);
    T(4) << "Done: tensor assignment." << EndT;
    Statement::Execute(ctx);
  }
};

// TODO:  timewhoop will need support for While -ajaleel
class While : public Statement
{
 public:
  Expression* test_expr_;
  
  While(Expression* test_e) :
    test_expr_(test_e)
  {
  }

  virtual std::shared_ptr<timewhoop::Statement> ConvertStatement()
  {
      assert(0);
  }
  
  virtual void Execute(ExecutionContext& ctx)
  {
    T(4) << "Entering: while loop." << EndT;
    while (test_expr_->Evaluate(ctx))
    {
      if (inner_)
      {
        inner_->Execute(ctx);
      }
    }
    T(4) << "Done: while loop." << EndT;
    Statement::Execute(ctx);
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

  virtual void Execute(ExecutionContext& ctx)
  {
    T(4) << "Entering: Flush Buffer." << EndT;
    T(4) << "  Tensor: "<<tensor_ptr_->name_<< EndT;
    T(4) << "  Spatial Partition: "<<ctx.current_spatial_partition_<< EndT;

    
    int num_buffs  = buffs_->size();

    auto buf_id  = whoop::buff::GetBufIndex( ctx.CurrentSpatialPartition(), num_buffs, ctx.MaxSpatialPartitions() );
    
    (*buffs_)[buf_id]->FlushBuffet();
    
    T(4) << "Done: Flush Bufer." << EndT;
    Statement::Execute(ctx);
  }
};

class TemporalFor : public Statement
{
 public:
  Statement* init_stmt_;
  Expression* test_expr_;
  Statement* incr_stmt_;
  
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
  
  virtual void Execute(ExecutionContext& ctx)
  {
    T(4) << "Entering: temporal for-loop." << EndT;
    for (init_stmt_->Execute(ctx); test_expr_->Evaluate(ctx); incr_stmt_->Execute(ctx))
    {
      if (inner_)
      {
        inner_->Execute(ctx);
      }
    }
    T(4) << "Done: temporal for-loop." << EndT;
    Statement::Execute(ctx);
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

 
  virtual void Execute(ExecutionContext& ctx)
  {
    T(4) << "Entering: spatial for-loop: " << num_partitions_ << EndT;
    ctx.BeginSpatialPartitioning(num_partitions_);
    for (init_stmt_->Execute(ctx); test_expr_->Evaluate(ctx); incr_stmt_->Execute(ctx))
    {
      if (inner_)
      {
        inner_->Execute(ctx);
      }
      ctx.NextSpatialPartition();
    }
    ctx.EndSpatialPartitioning();
    T(4) << "Done: temporal spatial-loop." << EndT;
    Statement::Execute(ctx);
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

  
  virtual void Execute(ExecutionContext& ctx)
  {
    T(4) << "Entering: dynamic if." << EndT;
    int res = test_expr_->Evaluate(ctx);
    if (res)
    {
      T(4) << "    (condition evaluated to true)" << EndT;
      if (inner_)
      {
        inner_->Execute(ctx);
      }
    }
    else 
    {
      T(4) << "    (condition evaluated to false)" << EndT;
      if (else_)
      {
        T(4) << "    (switching to else-branch)" << EndT;
        else_->Execute(ctx);
      }
    }
    T(4) << "Done: dynamic if." << EndT;
    // Note: we explicitly don't call Statement::Execute(ctx) if we have an else.
    // Instead we jump to whatever is after the else (even if the else itself
    // wasn't executed).
    if (else_)
    {
      if (else_->next_)
      {
        else_->next_->Execute(ctx);
      }
    }
    else
    {
      Statement::Execute(ctx);
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

  virtual void Execute(ExecutionContext& ctx)
  {
    T(4) << "Entering: else." << EndT;
    if (inner_)
    {
      inner_->Execute(ctx);
    }
    T(4) << "Done: else." << EndT;
    // Note: we explicitly don't call Statement::Execute(ctx) here.
    // It will be called through ExecuteNext
  }
};

}  // namespace ast

}  // namespace whoop

#endif /* WHOOP_ABSTRACT_SYNTAX_HPP_ */
