#!/bin/bash
#SBATCH --job-name=repo_2_test_2
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=repo_2_test_2.out
#SBATCH --error=repo_2_test_2.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-uncurved-Ibeth problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59 name=testing_59
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_maze/outputs/mazes_ablation/training-unraised-Lucrecia problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59 name=testing_59