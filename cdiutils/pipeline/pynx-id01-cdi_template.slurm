#!/bin/bash -l
#SBATCH --partition=gpu,p9gpu
#SBATCH --nodes=$number_of_nodes
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=4G
#SBATCH --output=slurm-%j.out

scontrol --details show jobs $SLURM_JOBID | grep RES

module purge
module load pynx

cd $data_path

mpiexec -n $SLURM_NTASKS pynx-cdi-id01 pynx-cdi-inputs.txt