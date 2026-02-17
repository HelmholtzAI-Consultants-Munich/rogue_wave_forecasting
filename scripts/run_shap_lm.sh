#!/bin/bash
#SBATCH --job-name=shap_lm
#SBATCH --output=shap_lm.out
#SBATCH --error=shap_lm.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1    
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
~/anaconda3/envs/rogue_wave/bin/python -V
~/anaconda3/envs/rogue_wave/bin/python -u run_shap.py  \
    --batch_size 40000  \
    --batch_multiprocessing  \
    --dataset test  \
    --n_dataset 40000  \
    --n_background 2000  \
    --model_type Linear  \
    --file_data_model ../results/lm/model_and_data.pkl  \
    --dir_output ../results/lm/shap_lm/  \
    --n_jobs 1
~/anaconda3/envs/rogue_wave/bin/python -u run_shap.py  \
    --batch_size 160000  \
    --batch_multiprocessing  \
    --dataset train  \
    --n_dataset 160000  \
    --n_background 2000  \
    --model_type Linear  \
    --file_data_model ../results/lm/model_and_data.pkl  \
    --dir_output ../results/lm/shap_lm/  \
    --n_jobs 1