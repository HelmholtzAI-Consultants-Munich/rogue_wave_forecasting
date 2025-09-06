#!/bin/bash
#SBATCH --job-name=rw_shap
#SBATCH --output=rw_shap.out
#SBATCH --error=rw_shap.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --ntasks=16          # Number of tasks
#SBATCH --cpus-per-task=1   # Number of CPU cores
#SBATCH --mem=100GB           # Memory allocation

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
/home/haicu/lisa.barros/anaconda3/envs/rogue_wave/bin/python -u shap_rf.py
