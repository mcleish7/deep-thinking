#!/bin/bash
#SBATCH --job-name=test_time_2
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=test_time_2.out
#SBATCH --error=test_time_2.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/test_time.py