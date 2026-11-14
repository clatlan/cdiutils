#!/bin/bash -l
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
# Start from a clean login environment on the node: the environment of
# the (x86_64) submission host would otherwise be exported, and its
# MODULEPATH would load x86_64 modules on the ppc64le (p9gpu) nodes.
#SBATCH --export=NONE

# Template filled by BcdiPipeline._make_slurm_file (string.Template):
# a doubled dollar sign is a bash variable, a single one a value set by
# cdiutils (partition, data_path, pynx_env_x86_64, pynx_env_ppc64le).

scontrol --details show jobs $SLURM_JOBID | grep RES

# The partitions can mix architectures (gpu: x86_64, p9gpu: ppc64le)
arch=$$(uname -m)
echo "Node $$(hostname), architecture $$arch"

# PyNX environment (prefix) given by cdiutils for this architecture
case "$$arch" in
    x86_64) pynx_env="$pynx_env_x86_64" ;;
    ppc64le) pynx_env="$pynx_env_ppc64le" ;;
    *) pynx_env="" ;;
esac

# Find a pynx-cdi-id01 that runs on this node: first the environment
# given by cdiutils, then the ESRF pynx module of this architecture.
if [ -n "$$pynx_env" ]; then
    export PATH="$$pynx_env/bin:$$PATH"
    # pycuda compiles its kernels with nvcc, which a conda environment
    # may not provide: use the CUDA module of this architecture then
    if ! command -v nvcc > /dev/null 2>&1; then
        echo "No nvcc in $$pynx_env, loading the cuda module."
        module load cuda
    fi
fi
if ! pynx-cdi-id01 --help > /dev/null 2>&1; then
    echo "No working pynx-cdi-id01 in the environment, loading the pynx module."
    module load pynx
fi
if ! pynx-cdi-id01 --help > /dev/null 2>&1; then
    echo "ERROR: no working pynx-cdi-id01 on this $$arch node."
    echo "Give a PyNX environment for $$arch with the pynx_envs argument of"
    echo "BcdiPipeline.phase_retrieval, e.g. pynx_envs={'$$arch': '/path/to/env'},"
    echo "or restrict the partition (slurm_partition) to nodes where it works."
    exit 1
fi
echo "Using $$(command -v pynx-cdi-id01)"

cd "$data_path" || exit 1

pynx-cdi-id01 pynx-cdi-inputs.txt
