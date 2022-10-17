#!/bin/bash
#SBATCH --job-name=repo_4
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=repo_4.out
#SBATCH --error=repo_4.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=1 problem.hyp.lr=0.0001 problem/model=dt_net_1d problem=prefix_sums name=prefix_sums_ablation problem.test_data=48
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=0 problem/model=dt_net_1d problem=prefix_sums name=prefix_sums_ablation problem.test_data=48
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=1 problem/model=dt_net_recall_1d problem=prefix_sums name=prefix_sums_ablation problem.test_data=48
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=0 problem/model=dt_net_recall_1d problem=prefix_sums name=prefix_sums_ablation problem.test_data=48
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=0 problem/model=ff_net_1d problem=prefix_sums name=prefix_sums_ablation problem.test_data=48
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=1 problem/model=ff_net_recall_1d problem=prefix_sums name=prefix_sums_ablation problem.test_data=48
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=0 problem/model=ff_net_recall_1d problem=prefix_sums name=prefix_sums_ablation problem.test_data=48