#!/bin/bash
#SBATCH --job-name=ffnn
#SBATCH --output=log/job_%j.out
#SBATCH --error=log/job_%j.err
#SBATCH --partition=gpu  
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100_40
#SBATCH --qos=24hours
#SBATCH --time=24:00:00

module load pytorch

python -u train_ffnn.py
