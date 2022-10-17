#!/bin/bash
#SBATCH --job-name=maze_12
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=shlurm_out_maze/maze_12.out
#SBATCH --error=shlurm_out_maze/maze_12.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=0.00 problem.hyp.epochs=200 problem.hyp.lr=0.0001 problem.hyp.lr_schedule=[175] problem/model=ff_net_recall_2d problem=mazes name=mazes_ablation