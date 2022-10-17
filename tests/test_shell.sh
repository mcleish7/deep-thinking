#!/bin/bash
#SBATCH --job-name=small_test
#SBATCH --partition=dualgpu-batch
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=small_test_.out
#SBATCH --error=small_test.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 test.py