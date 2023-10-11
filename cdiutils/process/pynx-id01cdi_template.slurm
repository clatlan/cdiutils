#!/bin/bash -l
#SBATCH --partition=gpu,p9gpu
#SBATCH --nodes=$number_of_nodes
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=4G
#SBATCH --output=%x-%j.out

module purge

scontrol --details show jobs $SLURM_JOBID | grep RES

cd $data_path

module load pynx
mpiexec -n $SLURM_NTASKS pynx-cdi-id01 pynx-cdi-inputs.txt