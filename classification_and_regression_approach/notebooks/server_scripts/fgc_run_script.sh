#!/bin/bash
#SBATCH --job-name=rw_fgc
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --ntasks=1          # Number of tasks
#SBATCH --cpus-per-task=32   # Number of CPU cores
#SBATCH --mem=300GB           # Memory allocation

# Check for user input
if [ -z "$1" ]; then
  echo "Usage: $0 <k>"
  echo "Error: You must specify a value for k."
  exit 1
fi

K=$1
echo "Running FGC with k = $K"

# Set up environment
export PATH=~/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
echo "Using Conda:"
source ~/anaconda3/etc/profile.d/conda.sh
conda -V

export OMP_NUM_THREADS=25
export OPENBLAS_NUM_THREADS=25
export MKL_NUM_THREADS=25

# Run the Python script
echo "Using Python:"
/home/haicu/lisa.barros/anaconda3/envs/rogue_wave/bin/python -V
/home/haicu/lisa.barros/anaconda3/envs/rogue_wave/bin/python -u xai_fgc.py --k "$K"
