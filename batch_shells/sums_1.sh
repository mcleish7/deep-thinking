#!/bin/bash
#SBATCH --job-name=sums_1
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=sums_1.out
#SBATCH --error=sums_1.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 ~/Documents/deep-thinking/train_model.py problem.hyp.alpha=1 problem.hyp.lr=0.0001 problem/model=dt_net_1d problem=prefix_sums name=prefix_sums_ablation