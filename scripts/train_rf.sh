#!/bin/bash
#SBATCH --job-name=train_rf
#SBATCH --output=train_rf.out
#SBATCH --error=train_rf.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --ntasks=1            
#SBATCH --cpus-per-task=50    
#SBATCH --mem=200GB   

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
uv run --project .. python -u train_model.py  \
    --model_type rf  \
    --cv_type grouped \
    --n_jobs 50

uv run --project .. python -u train_model.py  \
    --model_type rf  \
    --cv_type time_series \
    --n_jobs 50