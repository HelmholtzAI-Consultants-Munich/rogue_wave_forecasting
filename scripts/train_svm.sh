#!/bin/bash
#SBATCH --job-name=train_svm
#SBATCH --output=train_svm.out
#SBATCH --error=train_svm.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=supergpu[02-33]
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1   
#SBATCH --mem=300GB  

# uv lives in ~/.local/bin after the one-time cluster setup (see README)
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy

# train_model.py resolves ../data and ../results, so run from scripts/
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "Using uv:"
uv --version
echo "Using Python:"
uv run --project .. --no-sync python -V

# --no-sync: use the .venv built on the login node, never hit the network here
uv run --project .. --no-sync python -u train_model.py  \
    --model_type svm  \
    --file_data ../data/data_train_test_new.pickle  \
    --dir_output ../results/svm_new/  \
    --n_jobs 1 
