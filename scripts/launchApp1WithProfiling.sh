#!/bin/bash

#SBATCH --job-name="MyProjectWithProfiling"
#SBATCH --time=00:05:00
#SBATCH --constraint=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=debug

unset COMPUTE_PROFILE 
export PMI_NO_FORK=1 

#Go check http://docs.nvidia.com/cuda/profiler-users-guide/index.html#metrics-reference-6x
#For a list of metrics
#======START=====
srun nvprof -o app1.%h.%p.nvvp\
  --metrics \
flops_dp,\
flop_count_dp_add,\
flop_count_dp_mul,\
flop_count_dp_fma,\
flop_dp_efficiency,\
gld_throughput,\
gld_transactions,\
gst_throughput,\
gst_transactions,\
inst_compute_ld_st,\
inst_control,\
inst_executed,\
inst_fp_64\
  ./app1
#=====END====

