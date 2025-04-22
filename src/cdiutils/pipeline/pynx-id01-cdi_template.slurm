#!/bin/bash -l
#SBATCH --partition=gpu,p9gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out

scontrol --details show jobs $SLURM_JOBID | grep RES

module load pynx

cd $data_path

pynx-cdi-id01 pynx-cdi-inputs.txt