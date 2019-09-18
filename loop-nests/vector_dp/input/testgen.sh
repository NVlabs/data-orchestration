#!/bin/bash
rnz=300
vocabsize=100000

./random-tensor-gen.bin --dim_sizes=$rnz $vocabsize --seed=1717 --min=1 --max=2 --ref_output_file=test.ref.txt --tensor_output_file=test.in.txt 
