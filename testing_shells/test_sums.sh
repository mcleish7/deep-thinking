#!/bin/bash
#SBATCH --job-name=test_sums
#SBATCH --partition=dualgpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=shlurm_testing_shells_sums/test_sums.out
#SBATCH --error=shlurm_testing_shells_sums/test_sums.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-jobless-Shatyra
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-enraged-Jojo
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-peeling-Betzaida
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-noteless-Crystin
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-embowed-Aliza
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-venous-Jayson