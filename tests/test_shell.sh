#!/bin/bash
#SBATCH --job-name=test_shell
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=small_test_.out
#SBATCH --error=small_test.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=0 problem/model=dt_gnn_recall problem=graphs name=gnn_test problem.hyp.lr=0.001 problem.hyp.epochs=5 problem.hyp.train_batch_size=1 problem.hyp.test_batch_size=1

# python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=1 problem/model=dt_net_recall_2d problem=cifar10 name=cifar10_ablation

# python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=1 problem/model=gnn problem=graphs_on_fly name=gnn_test