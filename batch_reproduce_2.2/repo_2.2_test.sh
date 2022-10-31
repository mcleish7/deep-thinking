#!/bin/bash
#SBATCH --job-name=repo_2.2_test
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=repo_2.2_test.out
#SBATCH --error=repo_2.2_test.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_reproduce_5/outputs/mazes_ablation/training-bubbly-Jackqueline problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_reproduce_5/outputs/mazes_ablation/training-chatty-Luisalberto problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_reproduce_5/outputs/mazes_ablation/training-drizzly-Efren problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_reproduce_5/outputs/mazes_ablation/training-fading-Ilea problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_reproduce_5/outputs/mazes_ablation/training-fluffy-Meena problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_reproduce_5/outputs/mazes_ablation/training-gratis-Dela problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_reproduce_5/outputs/mazes_ablation/training-huger-Tevis problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59