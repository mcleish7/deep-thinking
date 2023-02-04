#!/bin/bash
#SBATCH --job-name=test_contecg
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

source /etc/profile.d/modules.sh
module load CUDA

srun python3.9 graph_context.py --dataset=shortestpath