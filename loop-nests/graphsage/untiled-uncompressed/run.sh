#!/bin/sh
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
scons -uj32 -Q

if [ $# -eq 1 ]; then
    tlevel=$1
else
    tlevel=3
fi

vertices=64
features=8
hidden=16
intermed=$[2*$hidden]

in_dir=../input/
out_dir=../output/

# Run the program with some interesting (and legal) default settings
WHOOP_CHECK_REFERENCE=1 ./graphsage.bin \
  --tensor_adj_matrix.$vertices.$vertices_file=$in_dir/adj_matrix.$vertices.$vertices.in.txt \
  --tensor_neib_sample.$vertices.$vertices_file=$in_dir/neib_sample.$vertices.$vertices.in.txt \
  --tensor_features.$vertices.$features_file=$in_dir/features.$vertices.$features.in.txt \
  --tensor_W_in.$hidden.$features_file=$in_dir/W_in.$hidden.$features.in.txt \
  --tensor_W_1.$hidden.$intermed_file=$in_dir/W_1.$hidden.$intermed.in.txt \
  --tensor_b_1.$hidden_file=$in_dir/b_1.$hidden.in.txt \
  --tensor_W_2.$hidden.$hidden_file=$in_dir/W_2.$hidden.$hidden.in.txt \
  --tensor_b_2.$hidden_file=$in_dir/b_2.$hidden.in.txt \
  --tensor_W_out.$hidden_file=$in_dir/W_out.$hidden.in.txt \
  --ref_prediction.$vertices_file=$out_dir/prediction.$vertices.ref.txt \
  --trace_level=${tlevel}
