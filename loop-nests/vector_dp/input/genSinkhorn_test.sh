#!/bin/bash
rnz=4
dbsize=8
vocabsize=16

./random-tensor-gen.bin --dim_sizes=$rnz $dbsize --seed=1717 --min=1 --max=2 --ref_output_file=swmd_X.in.txt --tensor_output_file=swmd_X.in.txt 
./random-tensor-gen.bin --dim_sizes=$rnz $vocabsize --seed=1717 --min=1 --max=2 --ref_output_file=swmd_X.in.txt --tensor_output_file=swmd_K.in.txt
./random-tensor-gen.bin --dim_sizes=$vocabsize $rnz --seed=1717 --min=1 --max=2 --ref_output_file=swmd_X.in.txt --tensor_output_file=swmd_KT.in.txt
./random-tensor-gen.bin --dim_sizes=$rnz --seed=1717 --min=1 --max=2 --ref_output_file=swmd_X.in.txt --tensor_output_file=swmd_R.in.txt 
#sparse
./random-tensor-gen.bin --dim_sizes=$dbsize $vocabsize --seed=1717 --min=0 --max=2 --ref_output_file=swmd_C.in.txt --tensor_output_file=swmd_C.in.txt 

./encode-tensor-no-zeros.bin --tensor_input_file=swmd_C.in.txt  --tensor_output_file=swmd_C_sparse.in.txt
