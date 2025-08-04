#!/bin/bash -l
#SBATCH --account=m4639
#SBATCH --job-name PhaseRetrieval
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --qos=regular
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out

# Load MPI
module use /global/common/software/m3169/perlmutter/modulefiles
module load openmpi

scontrol --details show jobs $SLURM_JOBID | grep RES

cd $data_path

mpiexec -n $SLURM_NTASKS /global/common/software/m4639/pynx-env/bin/pynx-cdi-id01 pynx-cdi-inputs.txt