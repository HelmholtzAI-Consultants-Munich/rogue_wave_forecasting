#!/bin/bash
#SBATCH --job-name=train_xgb
#SBATCH --output=train_xgb.out
#SBATCH --error=train_xgb.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=50    
#SBATCH --mem=500GB   

# Ensure Conda is in the PATH
export PATH=~/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

# Initialize Conda in the script 
source ~/anaconda3/etc/profile.d/conda.sh
echo "Using Conda:"
conda -V    

# Activate the Conda environment
echo "Using Python:"
~/anaconda3/envs/rogue_wave/bin/python -V
~/anaconda3/envs/rogue_wave/bin/python -u train_model.py  \
    --model_type xgb  \
    --file_data ../data/data_train_test.pickle  \
    --dir_output ../results/xgb/  \
    --n_jobs 50