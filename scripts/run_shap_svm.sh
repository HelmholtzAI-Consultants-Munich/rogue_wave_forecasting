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
#SBATCH --array=0-1599%10   # 1600 batches × 100 = 160000 samples

# -------------------------------
# Batch configuration
# -------------------------------
BATCH_SIZE=100
DATASET="train"
N_DATASET=160000
OUTPUT_DIR="/lustre/groups/aiconsultants/workspace/lisa.barros/shap_svm"

BATCH_START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))

echo "Array task ID     : $SLURM_ARRAY_TASK_ID"
echo "Batch start index : $BATCH_START"

# -------------------------------
# Skip if batch already computed
# -------------------------------
OUTPUT_FILE="${OUTPUT_DIR}/${DATASET}_shap_batch${BATCH_START}.pkl"

if [ -f "$OUTPUT_FILE" ]; then
    echo "$(date) | Output already exists, skipping batch ${BATCH_START}"

    rm -f "shap_svm_${DATASET}_${SLURM_ARRAY_TASK_ID}.out" \
          "shap_svm_${DATASET}_${SLURM_ARRAY_TASK_ID}.err"

    exit 0
fi

# -------------------------------
# Conda setup
# -------------------------------
export PATH=~/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

source ~/anaconda3/etc/profile.d/conda.sh
echo "Using Conda:"
conda -V

echo "Using Python:"
/home/haicu/lisa.barros/anaconda3/envs/rogue_wave/bin/python -V

# -------------------------------
# Run SHAP computation (ONE batch)
# -------------------------------
/home/haicu/lisa.barros/anaconda3/envs/rogue_wave/bin/python -u run_shap.py \
  --batch_size "${BATCH_SIZE}" \
  --last_batch "${BATCH_START}" \
  --multi_batch False \
  --dataset "${DATASET}" \
  --n_dataset "${N_DATASET}" \
  --n_background 1000 \
  --model_type Kernel \
  --file_data_model ../results/svm/model_and_data.pkl \
  --dir_output "${OUTPUT_DIR}"
