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

#ifndef WHOOP_OPTIONS_HPP_
#define WHOOP_OPTIONS_HPP_

#include <boost/program_options.hpp>
#include <string>
#include <iostream>

namespace whoop
{

namespace options
{

// ******* Static options can be changed below *******
// No static options yet.

// ******* Dynamic options (see options.cpp to change default value) *******
extern bool kShouldCheckReferenceOutput;
extern bool kShouldFlushTiles;
extern bool kShouldTraceExecution;
extern bool kShouldTraceBuffers;
extern bool kShouldLogActivity;
extern int kCurrentTraceLevel;
extern int kUserTraceLevel;
extern int kCoalescingWindowSize;
extern std::string kPhysicalPath;
extern std::string kPhysicalFile;
extern std::string kProgramName;
extern std::string kStatsFileName;


// This function parses the user's environment variables and over-rides the
// default values as appropriate.
void SetOverrides();

std::string GetHelpMessage();

// These are implementation specific and should not be accessed elsewhere.
extern boost::program_options::options_description desc_;

}  // namespace options

enum
{
  kOptionNotRequired = 0,
  kOptionRequired = 1
}
OptionRequirements;

extern boost::program_options::options_description user_desc;
extern bool parsed_users_options;

void AddOption(int* var, const std::string& name, const std::string& desc);
void AddOption(std::vector<int>* vec, const std::string& name, const std::string& desc);
void AddOption(std::string* var, const std::string& name, const std::string& desc);

void ParseOptions(int argc, char** argv);


}  // namespace whoop

#endif  // LITTLE_OPTIONS_HPP

