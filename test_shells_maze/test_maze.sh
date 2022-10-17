#!/bin/bash
#SBATCH --job-name=test_maze
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=test_maze.out
#SBATCH --error=test_maze.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-abased-Paden problem=mazes
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-algal-Collyn problem=mazes
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-boorish-Tyronda problem=mazes
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-boughten-Lao problem=mazes
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-brinish-Leanna problem=mazes
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-cany-Bedford problem=mazes
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-described-Allisen problem=mazes
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-distinct-Cornesha problem=mazes
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-leachy-Gale problem=mazes
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-palsied-Zach problem=mazes
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-plumbic-Aleksandra problem=mazes
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-trusty-Ashlynne problem=mazes
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-uncurved-Ibeth problem=mazes
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-unraised-Lucrecia problem=mazes