#!/bin/bash
#SBATCH --job-name=repo_4_test
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=repo_4_test.out
#SBATCH --error=repo_4_test.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/prefix_sums_ablation/training-advised-Emmanual problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=48
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/prefix_sums_ablation/training-diverse-Shawnta problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=48
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/prefix_sums_ablation/training-fuscous-Laporsha problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=48
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/prefix_sums_ablation/training-osmic-Jenae problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=48
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/prefix_sums_ablation/training-slaty-Ilaisanna problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=48
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/prefix_sums_ablation/training-tensest-Froylan problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=48
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/prefix_sums_ablation/training-thoughtless-Celso problem.model.test_iterations.low=0 problem.model.test_iterations.high=200 problem.test_data=48