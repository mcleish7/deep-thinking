#!/bin/bash
#SBATCH --job-name=test_noise_maze
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=test_noise_maze_mismatch.out
#SBATCH --error=test_noise_maze_mismatch.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/test_noise_maze_mismatch.py