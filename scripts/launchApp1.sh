#!/bin/bash -l
#SBATCH --job-name="example_thbt"
#SBATCH --time=00:05:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=debug
##SBATCH --output=app1.%j.o
##SBATCH --error=app1.%j.e

#======START=====
srun ./app1
#=====END====

