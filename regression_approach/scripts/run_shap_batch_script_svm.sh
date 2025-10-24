#!/bin/bash
#SBATCH --job-name=svm_shap
#SBATCH --output=svm_shap.out
#SBATCH --error=svm_shap.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --ntasks=1          # Number of tasks
#SBATCH --cpus-per-task=1   # Number of CPU cores
#SBATCH --mem=300GB         # Memory allocation
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
/home/haicu/lisa.barros/anaconda3/envs/rogue_wave/bin/python -u run_shap.py --batch_size 100 --last_batch -1 --dataset test --n_dataset 4000 --n_background 200 --model_type Kernel --file_data_model ../results/svm/model_and_data.pickle --dir_output /lustre/groups/aiconsultants/workspace/lisa.barros/shap/svm/
/home/haicu/lisa.barros/anaconda3/envs/rogue_wave/bin/python -u run_shap.py --batch_size 100 --last_batch -1 --dataset train --n_dataset 16000 --n_background 200 --model_type Kernel --file_data_model ../results/svm/model_and_data.pickle --dir_output /lustre/groups/aiconsultants/workspace/lisa.barros/shap/svm/
