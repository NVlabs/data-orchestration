/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <any>


namespace queueda
{

namespace options
{

// ******* Static options can be changed below *******

#ifdef __CUDACC__

static const int kDefaultQSize = 8;
static const int kMaxQBuffering = 49152;

#else

static const int kDefaultQSize = 1024;
static const int kMaxQBuffering = 10000000;

#endif

static const int kMaxQFanout = 16;

static const int kMaxBlocksPerGPU = 108;
static const int kMaxWarpsPerBlock = 64;
static const int kMaxThreadsPerWarp = 32;

static const int kMaxSequentialNodes = 8;

static const int kMsgBufferSize = 256;

// ******* Dynamic options (changed via environment variable) *******
struct DynamicOptions
{
 public:
  bool kShouldCheckReferenceOutput = true;
  int kCurrentLibraryTraceLevel = 0;
};

// Device copies of dynamic options have to be separate, so 
// make two copies of the struct.

DynamicOptions* host_;
__CUDA_DEVICE__ DynamicOptions* device_;

}

std::vector<std::any> options__;

template 
<
  typename T 
>
class OptionInfo
{
 public:
  std::string name_;
  std::string desc_;
  T* target_;
  
  OptionInfo(const char* name, const char* desc, T* target) :
    name_(name),
    desc_(desc),
    target_(target)
  {
  }
  
  void Parse(const char* str) {
    std::stringstream s(str);
    s >> (*target_);
  }
};


template <
  typename T
>
void AddOption(const char* name, const char* desc, const T& default_val, T* target)
{
  *target = default_val;
  OptionInfo<T> opt(name, desc, target);
  std::any wrapper(std::make_any<OptionInfo<T>>(opt));
  options__.push_back(wrapper);
}

void AddOptions()
{
  options::host_ = new options::DynamicOptions();
  AddOption("QUEUEDA_CHECK_REFERENCE","Should check reference output?", true, &options::host_->kShouldCheckReferenceOutput);
  AddOption("QUEUEDA_TRACE_LEVEL", "Current library trace level of detail.", 0, &options::host_->kCurrentLibraryTraceLevel);
}

template
<
  typename T
>
void ParseOptions()
{
  for (auto option : options__)
  {
    try
    {
      auto foption = std::any_cast<OptionInfo<T>>(option);
      //printf("CHECKING %s\n", foption.name_.c_str());
      auto val = std::getenv(foption.name_.c_str());
      if (val)
      {
        //printf("Setting %s to %s\n", foption.name_.c_str(), val);
        foption.Parse(val);
      }
    }
    catch (const std::bad_any_cast& e)
    {
      // No problem, just continue.
    }
  }
}

void ReadOptions()
{
  ParseOptions<bool>();
  ParseOptions<int>();
  ParseOptions<float>();
  
#ifdef __CUDACC__

  auto dev_opt = AllocOnDevice<options::DynamicOptions>();
  SetDeviceValue(dev_opt, options::host_);
  gpuErrchk(cudaMemcpyToSymbol(options::device_, &dev_opt, sizeof(options::DynamicOptions*)));

#else

  options::device_ = options::host_;

#endif

}


}  // namespace queuda

