# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

env = Environment(ENV = os.environ)

debug = ARGUMENTS.get('debug',0)

env.Append(CCFLAGS = ['-fmax-errors=1', '-std=c++14'])
env.Append(LIBPATH = ['#lib/whoop'])
env.Append(CPPPATH = ['#lib/whoop/include'])
env.Append(LIBPATH = ['#lib/timewhoop'])
env.Append(CPPPATH = ['#lib/timewhoop/include'])
env.Append(CPPPATH = ['loop-nests/conv6d/lib'])

if int(debug):
  env.Append(CCFLAGS = ['-g'])
else:
  env.Append(CCFLAGS = ['-O3'])

if os.environ.get('BOOSTDIR'):
    env.Append(CPPPATH = [os.environ['BOOSTDIR'] + '/include'])
    env.Append(LIBPATH = [os.environ['BOOSTDIR'] + '/lib'])

env.Append(LINKFLAGS = ['-std=c++11', '-pthread', '-static-libgcc', '-static-libstdc++'])

env.Append(LIBS = ['whoop', 'timewhoop', 'boost_program_options', 'boost_serialization'])

subdirs = ['lib', 'tools', 'tests', 'loop-nests']

def PhonyTargets(env = None, **kw):
    if not env: env = DefaultEnvironment()
    for target,action in kw.items():
        env.AlwaysBuild(env.Alias(target, [], action))

pycmd = "python"
if os.environ.get('PYTHON'):
    pycmd = os.environ['PYTHON'] 

PhonyTargets(lint = 'find . -name "*.cpp" -or -name "*.hpp" | grep -v opencv | xargs ' + pycmd + ' util/cpplint.py --linelength=256 2> cpplint.out')

for subdir in subdirs:
    env.SConscript(subdir + '/SConscript', {'env': env})

