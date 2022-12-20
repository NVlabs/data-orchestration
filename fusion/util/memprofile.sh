profile_name=${1}
gpuid=${2}
cmd=${3}

ncu_memory_metrics_L1_to_global="l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"
ncu_memory_metrics_global_to_L1="l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"
ncu_memory_metrics_L1_to_local="l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum"
ncu_memory_metrics_local_to_L1="l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum"
ncu_memory_metrics_to_shared_mem="smsp__sass_data_bytes_mem_shared_op_ld.sum"
ncu_memory_metrics_from_shared_mem="smsp__sass_data_bytes_mem_shared_op_st.sum"
ncu_memory_metrics_L1_to_L2="lts__t_sectors_srcunit_l1_op_write.sum"
ncu_memory_metrics_L2_to_L1="lts__t_sectors_srcunit_l1_op_read.sum"
ncu_memory_metrics_L2_to_sys_mem="lts__t_sectors_aperture_sysmem_op_write.sum"
ncu_memory_metrics_sys_mem_to_L2="lts__t_sectors_aperture_sysmem_op_read.sum"
ncu_memory_metrics_L2_to_device_mem="dram__bytes_write.sum"
ncu_memory_metrics_device_mem_to_L2="dram__bytes_read.sum"



if [[ $m == "all" ]]; then
    ncu_memory_metrics="${ncu_memory_metrics_L1_to_global},${ncu_memory_metrics_global_to_L1},${ncu_memory_metrics_L1_to_local},${ncu_memory_metrics_local_to_L1},${ncu_memory_metrics_to_shared_mem},${ncu_memory_metrics_from_shared_mem},${ncu_memory_metrics_L1_to_L2},${ncu_memory_metrics_L2_to_L1},${ncu_memory_metrics_L2_to_sys_mem},${ncu_memory_metrics_sys_mem_to_L2},${ncu_memory_metrics_L2_to_device_mem},${ncu_memory_metrics_device_mem_to_L2}"
else 
    ncu_memory_metrics="${ncu_memory_metrics_L2_to_device_mem},${ncu_memory_metrics_device_mem_to_L2}"
fi 

metrics="--metrics ${ncu_memory_metrics}"

echo ncu --csv -f --devices ${gpuid} --target-processes all --fp --print-kernel-base demangled --print-units base --profile-from-start on --clock-control base  --log-file ${profile_name}_ncu_metrics.csv ${metrics} ${cmd}
ncu --csv -f --devices ${gpuid} --target-processes all --fp --print-kernel-base demangled --print-units base --profile-from-start on --clock-control base  --log-file ${profile_name}_ncu_metrics.csv ${metrics} ${cmd}
