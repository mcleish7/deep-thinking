#!/bin/bash
#SBATCH --job-name=repo_5_test
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=slurm_repo_5/repo_5_test.out
#SBATCH --error=slurm_repo_5/repo_5_test.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-bubbly-Jackqueline problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=13
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-chatty-Luisalberto problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=13
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-drizzly-Efren problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=13
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-fading-Ilea problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=13
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-fluffy-Meena problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=13
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-gratis-Dela problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=13
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-huger-Tevis problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=13
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-leadless-Tempestt problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=13
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-noisy-Agatha problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=13
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-shoeless-Rhet problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=13
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-sissy-Daniell problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=13
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-sparkling-Saquan problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=13
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-uncut-Jaquelyn problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=13
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-unplumbed-Silvana problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=13