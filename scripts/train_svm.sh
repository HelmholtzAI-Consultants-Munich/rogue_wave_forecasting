#!/bin/bash
#SBATCH --job-name=train_svm
#SBATCH --output=train_svm.out
#SBATCH --error=train_svm.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1   
#SBATCH --mem=200GB         
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
/home/haicu/lisa.barros/anaconda3/envs/rogue_wave/bin/python -u train_model.py --model_type svm --file_data ../data/data_train_test.pickle --dir_output ../results/svm_new/ --n_jobs 1