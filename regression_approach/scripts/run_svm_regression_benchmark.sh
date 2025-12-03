#!/bin/bash
#SBATCH --job-name=svm_1_02
#SBATCH --output=svm_1_02.out
#SBATCH --error=svm_1_02.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --ntasks=2          # Number of tasks
#SBATCH --cpus-per-task=1   # Number of CPU cores per task
#SBATCH --mem=200GB         # Memory allocation
#SBATCH --gres=gpu:1

# Ensure Conda is in the PATH
export PATH=~/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

# Initialize Conda in the script 
source ~/anaconda3/etc/profile.d/conda.sh
echo "Using Conda:"
conda -V 

# Activate the Conda environment
echo "Using Python:"
/home/haicu/lisa.barros/anaconda3/envs/rogue_wave/bin/python -V
/home/haicu/lisa.barros/anaconda3/envs/rogue_wave/bin/python -u run_svm_regression_benchmark.py