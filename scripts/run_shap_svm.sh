#!/bin/bash
#SBATCH --job-name=shap_svm_40
#SBATCH --output=shap_svm_40.out
#SBATCH --error=shap_svm_40.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=supergpu[02-33]
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1   
#SBATCH --mem=300GB         

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
/home/haicu/lisa.barros/anaconda3/envs/rogue_wave/bin/python -u run_shap.py --batch_size 100 --last_batch 39000 --dataset test --n_dataset 40000 --n_background 1000 --model_type Kernel --file_data_model ../results/svm/model_and_data.pkl --dir_output ../results/svm/
