#!/bin/bash -l
#SBATCH --partition=p9gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=4
#SBATCH --gres=gpu:$number_of_gpus
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=4G
#SBATCH --output=%x-%j.out
#SBATCH --nodelist=p9-03

env | grep CUDA
scontrol --details show jobs $SLURM_JOBID | grep RES
  
cd $data_path
 
mpiexec -n $SLURM_NTASKS /sware/exp/pynx/devel.p9/bin/pynx-cdi-id01 pynx-cdi-inputs.txt