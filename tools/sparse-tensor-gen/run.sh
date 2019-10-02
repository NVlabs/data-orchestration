#!/bin/bash
set -e
scons -u -Q
./sparse-tensor-gen.bin --dim_sizes 8 8 --tensor_output_file=sparse2D.in.txt
