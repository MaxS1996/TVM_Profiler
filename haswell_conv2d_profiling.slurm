#!/bin/bash
#SBATCH --partition=haswell64
#SBATCH --job-name=conv2d_TVMquick
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s0144002@msx.tu-dresden.de
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2541
#SBATCH --time=33:00:00
#SBATCH --output=template_runner_haswell_conv2d_%j.log
pwd; hostname; date

echo "Running TVM conv2d grid search profiler on $SLURM_CPUS_ON_NODE CPU cores"
echo ""
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"

module load LLVM/11.0.0 Python/3.8.6 CMake/3.21.1-GCCcore-11.2.0
source ~/DIR/ssd/s0144002-TVMMapper/python_envs/haswell/bin/activate
export TVM_HOME=/home/s0144002/tvm_haswell_power
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}


cd ~/DIR/ssd/s0144002-TVMMapper/TVM_Profiler
python3 template_profiling.py -t haswell -w conv2d
