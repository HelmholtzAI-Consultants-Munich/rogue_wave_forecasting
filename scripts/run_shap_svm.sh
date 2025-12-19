#!/bin/bash
#SBATCH --job-name=shap_svm_train_%a
#SBATCH --output=shap_svm_train_%a.out
#SBATCH --error=shap_svm_train_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=supergpu[02-33]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300GB
#SBATCH --array=0-159%5   # 40 bins total, max 5 running at once

# Determine last_batch based on SLURM_ARRAY_TASK_ID
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    LAST_BATCH=-1
else
    LAST_BATCH=$(( SLURM_ARRAY_TASK_ID * 1000 ))
fi

echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Using last_batch = $LAST_BATCH"

# Ensure Conda is in the PATH
export PATH=~/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

# Initialize Conda in the script 
source ~/anaconda3/etc/profile.d/conda.sh
echo "Using Conda:"
conda -V 

# Use your env's Python
echo "Using Python:"
/home/haicu/lisa.barros/anaconda3/envs/rogue_wave/bin/python -V
/home/haicu/lisa.barros/anaconda3/envs/rogue_wave/bin/python -u run_shap.py \
  --batch_size 100 \
  --last_batch "${LAST_BATCH}" \
  --dataset train \
  --n_dataset 160000 \
  --n_background 1000 \
  --model_type Kernel \
  --file_data_model ../results/svm/model_and_data.pkl \
  --dir_output /lustre/groups/aiconsultants/workspace/lisa.barros/shap_svm/
