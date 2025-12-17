#!/bin/bash
#SBATCH --job-name=train_lm
#SBATCH --output=train_lm.out
#SBATCH --error=train_lm.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=50    
#SBATCH --mem=100GB   

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
/home/haicu/lisa.barros/anaconda3/envs/rogue_wave/bin/python -u train_model.py --model_type lm --file_data ../data/data_train_test.pickle --dir_output ../results/lm/ --n_jobs 50