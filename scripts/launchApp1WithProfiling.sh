#SBATCH --job-name="MyProjectWithProfiling"
#SBATCH --time=00:05:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=debug
#SBATCH --output=app1Nvprof.%h.%p.o
#SBATCH --error=app1.%j.e

#======START=====
srun nvprof --metrics \
  flop_count_sp_add,\
  flop_count_sp_fma,\
  flop_count_sp_mul,\
  flop_count_sp_special,\
  inst_fp_32,\
  gst_transactions,\
  gld_transactions,\
  tex_cache_transactions,\
  l2_atomic_transactions,\
  l1_cache_local_hit_rate,\
  tex_cache_hit_rate,\
  l2_l1_read_hit_rate,\
  l2_texture_read_hit_rate\
  --demangling on --csv ./app1
#=====END====

