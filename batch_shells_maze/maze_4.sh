#!/bin/bash
#SBATCH --job-name=maze_4
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=maze_4.out
#SBATCH --error=maze_4.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 ~/Documents/deep-thinking/train_model.py problem.hyp.alpha=1.00 problem/model=dt_net_2d problem=mazes name=mazes_ablation