#!/bin/bash
rnz=19
word2vec=300
vocabsize=100000

./random-tensor-gen.bin --dim_sizes=$rnz $word2vec --seed=1717 --min=1 --max=2 --ref_output_file=swmd_RF_small.in.txt --tensor_output_file=swmd_RD_small.in.txt 
./random-tensor-gen.bin --dim_sizes=$word2vec $vocabsize --seed=1717 --min=1 --max=2 --ref_output_file=swmd_VD_small.in.txt --tensor_output_file=swmd_VD_small.in.txt
