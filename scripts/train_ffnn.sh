#!/bin/bash

########################################################
# batch settings for job on levante at DKRZ, Rogue waves
# Dec, 2025
########################################################
#SBATCH --job-name=ffnn         # Job name
#SBATCH --partition=gpu       # Partition name
#SBATCH --mem=100G                      # Memory per node
#SBATCH --gres=gpu:a100_40
#SBATCH --qos=24hours                  #had to make a special request. valid till 20.12.2025
#SBATCH --time=24:00:00               # Time limit (hh:mm:ss)
#SBATCH --output=log/job_%j.out           # Standard output log
#SBATCH --error=log/job_%j.err            # Standard error log
#SBATCH --account=     # Account name

module load pytorch

python -u train_ffnn.py
