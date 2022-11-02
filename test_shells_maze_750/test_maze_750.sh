#!/bin/bash
#SBATCH --job-name=test_maze_750
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=test_maze_750.out
#SBATCH --error=test_maze_750.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-abased-Paden problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=750
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-algal-Collyn problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=750
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-boorish-Tyronda problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=750
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-boughten-Lao problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=750
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-brinish-Leanna problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=750
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-cany-Bedford problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=750
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-described-Allisen problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=750
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-distinct-Cornesha problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=750
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-leachy-Gale problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=750
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-palsied-Zach problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=750
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-plumbic-Aleksandra problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=750
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-trusty-Ashlynne problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=750
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-uncurved-Ibeth problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=750
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-unraised-Lucrecia problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=750