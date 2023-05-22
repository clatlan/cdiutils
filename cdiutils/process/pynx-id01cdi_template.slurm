#!/bin/bash -l
#SBATCH --partition=p9gpu
#SBATCH --nodes=$number_of_nodes
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=4G
#SBATCH --output=%x-%j.out

env | grep CUDA
scontrol --details show jobs $SLURM_JOBID | grep RES
  
cd $data_path

mpiexec -n $SLURM_NTASKS /sware/exp/pynx/2023.1/p9/bin/pynx-cdi-id01 pynx-cdi-inputs.txt