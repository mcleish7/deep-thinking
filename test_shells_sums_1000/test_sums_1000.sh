#!/bin/bash
#SBATCH --job-name=test_sums_1000
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=test_sums_1000.out
#SBATCH --error=test_sums_1000.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 ~/Documents/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-frockless-Verena problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-jobless-Shatyra problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-enraged-Jojo problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-peeling-Betzaida problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-noteless-Crystin problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-embowed-Aliza problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-venous-Jayson problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000