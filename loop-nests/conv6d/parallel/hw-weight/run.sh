#!/bin/sh

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

# Stop on errors
set -e
# build the executable
scons -u -Q

executable=./conv6d-parallel-hw-weight.bin

if [ $# -eq 1 ]; then
    infile=$1
	tlevel=0
    user_tile_size_h=4
    user_tile_size_w=4
elif [ $# -eq 2 ]; then
    infile=$1
	tlevel=$2
    user_tile_size_h=4
    user_tile_size_w=4
elif [ $# -eq 3 ]; then
    infile=$1
	tlevel=$2
    user_tile_size_h=$3
    user_tile_size_w=4
elif [ $# -eq 4 ]; then
    infile=$1
	tlevel=$2
    user_tile_size_h=$3
    user_tile_size_w=$4
else
    infile=A
	tlevel=0
    user_tile_size_h=4
    user_tile_size_w=4
fi

# Run the program with some interesting (and legal) default settings
WHOOP_CHECK_REFERENCE=1 ${executable} \
  --tensor_inputs_file=../../data-sets/inputs_${infile}.in.txt \
  --tensor_weights_file=../../data-sets/weights_${infile}.in.txt \
  --ref_outputs_file=../../data-sets/outputs_${infile}.ref.txt \
  --tensor_outputs_file=./outputs_${infile}.out.txt \
  --tile_size_h=${user_tile_size_h} \
  --tile_size_w=${user_tile_size_w} \
  --trace_level=${tlevel}

