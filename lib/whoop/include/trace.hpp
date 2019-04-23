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
 
#ifndef INCLUDE_TRACE_H_
#define INCLUDE_TRACE_H_

#include <iostream>
#include <sstream>

namespace whoop
{

extern int num_warnings_;

// Dummy class indicating it's time to flush.
class Flusher
{
 public:
  Flusher() {}
  ~Flusher() {}
};

// Buffers messages locally then dumps its local
// contents to the shared buffer.
// Also supports trace levels.
class Tracer
{
 protected:  
  std::stringstream buff_;
  std::ostream* ostr_;
  int trace_level_;
  int cur_level_;
  bool fatal_;
 public:
  explicit Tracer(std::ostream* o = NULL)  : buff_(), 
  ostr_(o), 
  trace_level_(0), 
  cur_level_(0),
  fatal_(false)
  {    // Default to stdout
    if (!ostr_)
    {
      ostr_ = &std::cout;
    }
  }
  Tracer(const Tracer& other)  : buff_(),
  ostr_(other.ostr_),
  trace_level_(other.trace_level_),
  cur_level_(other.cur_level_),
  fatal_(other.fatal_)
  {
  }
  Tracer& operator =(const Tracer& other)
  {
    ostr_ = other.ostr_;
    trace_level_ = other.trace_level_;
    cur_level_ = other.cur_level_;
    fatal_ = other.fatal_;
  }
  ~Tracer() { }
  template <typename T>
  Tracer& operator <<(T t)
  {
    // Just buffer it for now.
    buff_ << t;
    return *this;
  }

  Tracer& operator<<(std::ostream& (*f)(std::ostream&))
  {
    // Handle all the std::ostream stuff
    buff_ << f;
    return *this;
  }

  Tracer& operator<<(Tracer& (*f)(Tracer&))
  {
    // Just apply the function.
    return f(*this);
  }

  Tracer& operator<<(Flusher& endt)
  {
    // Just flush the buffer.
    return FlushBuffer();
  }

  Tracer& FlushBuffer()
  {
    if (cur_level_ <= trace_level_)
    {
      buff_ << std::endl;
      (*ostr_) << buff_.str() << std::flush;
    }
    if (fatal_) std::exit(1);
    return DiscardBuffer();
  }

  Tracer& DiscardBuffer()
  {
    buff_.str("");
    return *this;
  }

  int GetTraceLevel() 
  {
    return trace_level_;
  }

  void SetTraceLevel(int l) 
  {
    trace_level_ = l;
  }
  void SetCurrentLevel(int l) 
  {
    cur_level_ = l;
  }
  void SetFatal() 
  {
     fatal_ = true; // Exit after next buffer dump. 
  }
};

class Traceable
{
 protected:
  std::string name_;
  Tracer tracer_;
  Flusher EndT;  // Note: this variable is excused from following naming conventions.
 public:
  explicit Traceable(const std::string& name = "") : name_(name)
  {
  }
  const std::string& GetName() { return name_; }
  virtual void PrependToName(const std::string& p) { name_ = p + name_; }
  virtual void AppendToName(const std::string& s) { name_ = name_ + s; }
  virtual Tracer& T(int l = 0, int cc = -1)
  {
    tracer_.SetCurrentLevel(l);
    // Only print the cycle count if provided by the user.
    if (cc >= 0)
    {
      tracer_ << cc << ": ";
    }
    if (name_ != "")
    {
      tracer_ << name_;
      if (name_.back() != ' ')
      {
        tracer_ << ": ";
      }
    }
    return tracer_;
  }
  virtual Tracer& ASSERT(bool cond, int cc = -1)
  {
    if (cond) 
    {
      // It passed, so guarantee it won't print out.
      tracer_.SetCurrentLevel(tracer_.GetTraceLevel() + 1);
      return tracer_;
    }
    // Fatal assertion failure.
    tracer_.SetFatal();
    tracer_.SetCurrentLevel(0);
    // Only print the cycle count if provided by the user.
    if (cc >= 0)
    {
      tracer_ << cc << ": ";
    }
    if (name_ != "")
    {
      tracer_ << name_ << ": ";
    }
    tracer_ << "Assertion failed: ";
    return tracer_;
  }
  virtual Tracer& ASSERT_WARNING(bool cond, int cc = -1)
  {
    if (cond) 
    {
      // It passed, so guarantee it won't print out.
      tracer_.SetCurrentLevel(tracer_.GetTraceLevel() + 1);
      return tracer_;
    }
    num_warnings_++;
    tracer_.SetCurrentLevel(0);
    // Only print the cycle count if provided by the user.
    if (cc >= 0)
    {
      tracer_ << cc << ": ";
    }
    if (name_ != "")
    {
      tracer_ << name_ << ": ";
    }
    tracer_ << "WARNING: ";
    return tracer_;
  }
  void SetTraceLevel(int l) { tracer_.SetTraceLevel(l); }
};

class UserTracer : public Traceable
{
 public:
  UserTracer() :
    Traceable(options::kProgramName)
  {
    SetTraceLevel(options::kUserTraceLevel);
  }
};

}  // namespace whoop

#endif  // INCLUDE_TRACE_H_
