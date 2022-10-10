#!/bin/bash
#SBATCH --job-name=test_sum_1
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=shlurm_testing_shells_sums/test_sum_1.out
#SBATCH --error=shlurm_testing_shells_sums/test_sum_1.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 ~/Documents/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-frockless-Verena