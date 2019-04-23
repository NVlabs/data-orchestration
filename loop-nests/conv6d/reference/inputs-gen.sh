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

# Test vector generator
VECGENDIR=../../../tools/bin/

if [ $# -eq 6 ]; then
  K=$1
  C=$2
  R=$3
  S=$4
  W=$5
  H=$6
else
  K=5
  C=3
  R=3
  S=3
  W=15
  H=15
fi

INSIZE=$(($C*$W*$H))
WTSIZE=$(($K*$C*$R*$S))

echo "InputSize:$INSIZE"
echo "WeightSize:$WTSIZE"

# Generate Input
WHOOP_CHECK_REFERENCE=0 $VECGENDIR/random-vec-gen.bin \
  --size=$INSIZE \
  --seed=1717 \
  --min=0 \
  --max=255 \
  --vec_output_file=inputs_A.in.txt

# Generate Weight
WHOOP_CHECK_REFERENCE=0 $VECGENDIR/random-vec-gen.bin \
  --size=$WTSIZE \
  --seed=1717 \
  --min=0 \
  --max=255 \
  --vec_output_file=weights_A.in.txt

PRELUDE_DIM="22 serialization::archive 15 0 0 6 0"

echo $PRELUDE_DIM $K $C $R $S $W $H> dimensions_A.in.txt

