#!/bin/bash
#SBATCH --job-name=repo_23
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=out/repo_23.out
#SBATCH --error=out/repo_23.err

srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=1.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation problem.test_data=59 problem.train_data=9
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=1.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation problem.test_data=59 problem.train_data=9
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=0.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation problem.test_data=59 problem.train_data=9
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=0.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation problem.test_data=59 problem.train_data=9