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
 
#ifndef INCLUDE_STATS_H_
#define INCLUDE_STATS_H_

#include <list>
#include <map>

namespace whoop
{

class StatsCollection
{
 public:
  std::string name_;
  std::map<std::string, uint64_t> stats_;
  StatsCollection* parent_ = NULL;
  std::list<StatsCollection*> children_{};
  
  StatsCollection() = default;

  StatsCollection(StatsCollection* p) :
    parent_(p)
  {
    if (parent_)
    {
      parent_->AddChild(this);
    }
  }
  
  explicit StatsCollection(const std::string& nm, StatsCollection* parent = NULL) :
    name_(nm), parent_(parent)
  {
    if (parent_)
    {
      parent_->AddChild(this);
    }
  }  
  
  ~StatsCollection()
  {
    if( parent_)
    {
       parent_->RemoveChild(this);
    }
  }
    

  StatsCollection* GetParent()
  {
    return parent_;
  }

  void AddChild(StatsCollection* s)
  {
    children_.push_back(s);
  }

  void RemoveChild(StatsCollection* s)
  {
    children_.remove(s);
  }
  
  void IncrStat(const std::string& name, unsigned int num = 1)
  {
    stats_[name] = stats_[name] + num;
  }
  
  void RecordEvent(const std::string& name, uint64_t param = 0)
  {
    // Param is ignored fore now.
    IncrStat(name);
  }

  const std::string& GetName() { return name_; }
  virtual void PrependToName(const std::string& p) { name_ = p + name_; }
  virtual void AppendToName(const std::string& s) { name_ = name_ + s; }

  bool HasStats()
  {
    return (stats_.size() != 0);
  }

  bool ChildrenHaveStats()
  {
    bool has_stats = false;
    for (auto it = children_.begin(); it != children_.end(); it++)
    {
      has_stats |= (*it)->HasStats();
      has_stats |= (*it)->ChildrenHaveStats();
    }
    return has_stats;
  }

  void Indent(std::ostream& ofile, unsigned int lvl)
  {
    for (unsigned int x = 0; x < lvl; x++)
    {
      ofile << "  ";
    }
  }

  void PrintStats(std::ostream& ofile, unsigned int lvl, StatsCollection* aggregator)
  {
    for (std::map<std::string, uint64_t>::iterator it = stats_.begin(); it != stats_.end(); it++)
    {
      Indent(ofile, lvl);
      ofile << it->first << ": " << it->second << std::endl;
      // Add stat into subtotal if necessary.
      if (aggregator) aggregator->IncrStat(it->first, it->second);
    }
  }


  void DumpStats(std::ostream& ofile, unsigned int lvl, StatsCollection* aggregator)
  {
    Indent(ofile, lvl);
    ofile << name_ << ":" << std::endl;
    StatsCollection* final_aggregator = ChildrenHaveStats() ? NULL : aggregator;
    PrintStats(ofile, lvl + 1, final_aggregator);
    for (auto it = children_.begin(); it != children_.end(); it++)
    {
      (*it)->DumpStats(ofile, lvl + 1, this);
    }
    if (ChildrenHaveStats())
    {
      if (lvl == 0)
      {
        Indent(ofile, 0);
        ofile << "Totals:" << std::endl;
        PrintStats(ofile, 1, aggregator);
      }
      else
      {
        Indent(ofile, lvl + 1);
        ofile << "Sub-totals (" << name_ << "):" << std::endl;
        PrintStats(ofile, lvl + 2, aggregator);
      }
    }
  }
};

}  // namespace whoop

#endif  // INCLUDE_STATS_H_
