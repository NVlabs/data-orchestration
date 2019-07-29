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

#include <list>
#include <string>
#include <thread>
#include "options.hpp"

namespace po = boost::program_options;

namespace whoop
{

namespace options
{

// ******* Dynamic Options and Default Values ********
bool kShouldCheckReferenceOutput(true);
bool kShouldTraceExecution(false);
bool kShouldTraceBuffers(true);
bool kShouldLogActivity(false);
bool kShouldFlushTiles(false);
int kCurrentTraceLevel(0);
int kUserTraceLevel(0);
int kCoalescingWindowSize(32);
std::string kPhysicalPath(".");
std::string kPhysicalFile("mini-vivaldi.physical.yaml");
std::string kProgramName;
std::string kStatsFileName; 
    
po::options_description desc_{"Options for whoop"};

// ******** Dynamic Options Over-ride parsing information ********
void SetOverrides()
{
  desc_.add_options()
    ("check_reference,c",
      po::value<bool>(&kShouldCheckReferenceOutput)->implicit_value(true),
      "Check VecOuts against reference output.")
    ("trace_exec,e",
      po::value<bool>(&kShouldTraceExecution)->implicit_value(false),
      "Trace the execution of the abstract syntax tree.")
    ("flush_tiles,f",
      po::value<bool>(&kShouldFlushTiles)->implicit_value(false),
      "User Specified Manual Buffet Flushes.")
    ("trace_buffers,b",
      po::value<bool>(&kShouldTraceBuffers)->implicit_value(true),
      "Trace the execution of the tile buffer accesses.")
    ("log_activity,l",
      po::value<bool>(&kShouldLogActivity)->implicit_value(false),
      "Log activity for cycle-accurate simulation.")
    ("trace_level,t",
      po::value<int>(&kCurrentTraceLevel),
      "Level of trace detail (0=none, 1=some, 2=detailed, 3=system, 4=runtime)."),
    ("window_size,w",
      po::value<int>(&kCoalescingWindowSize),
      "Coalescing window size (default=32)."),
    ("physical_path,pp",
      po::value<std::string>(&kPhysicalPath),
      "Path to physical topology file."),
    ("physical_file,pf",
      po::value<std::string>(&kPhysicalFile),
      "Physical topology file name.");
  try
  {
    po::variables_map vm;
    // For now we only take variables from the environment. TODO: config files.
    // po::store(po::command_line_parser(argc, argv).options(desc_).run(), vm);
    po::store(po::parse_environment(desc_, "WHOOP_"), vm);
    po::notify(vm);
  }
  catch(po::unknown_option& err)
  {
    std::cerr << "ERROR: whoop environment variable parsing: " << err.what() << std::endl;
    std::exit(1);
  }
}

std::string GetHelpMessage()
{
  std::stringstream ss;
  ss << desc_;
  return ss.str();
}

}  // namespace options

// What follows are functions that help users easily create options for their own programs.


boost::program_options::options_description user_desc{"Options for whoop user program."};
bool parsed_users_options = false;

void AddOption(int* var, const std::string& name, const std::string& desc)
{
  user_desc.add_options()
    (name.c_str(),
      po::value<int>(var)->implicit_value(*var),
      desc.c_str());
}

void AddOption(std::vector<int>* vec, const std::string& name, const std::string& desc)
{
  user_desc.add_options()
    (name.c_str(),
      po::value<std::vector<int>>(vec)->multitoken(),
      desc.c_str());
}

void AddOption(std::string* var, const std::string& name, const std::string& desc)
{
  user_desc.add_options()
    (name.c_str(),
      po::value<std::string>(var)->implicit_value(*var),
      desc.c_str());
}

void ParseOptions(int argc, char** argv)
{

  // by default, what is the prefix for the whoop output file called?
  AddOption( &options::kStatsFileName, "stats", "Stats File Name");

  assert(!parsed_users_options);

  user_desc.add_options()
    ("help", "Print this help message.")
    ("trace_level,t",
      po::value<int>(&options::kUserTraceLevel),
      "Level of trace detail (0=none, 1=some, 2=detailed).");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(user_desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
      std::cout << user_desc << std::endl;
      std::exit(1);
  }

  options::kProgramName = std::string(argv[0]);

  // if no prefix provided, use the program name as the prefix
  if( options::kStatsFileName == "" )
  {
      options::kStatsFileName = options::kProgramName;
  }

  parsed_users_options = true;
}

}  // namespace whoop
