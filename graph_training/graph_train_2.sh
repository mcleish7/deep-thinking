#!/bin/bash
#SBATCH --job-name=graph_train_2
#SBATCH --partition=dualgpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=out/graph_train_2.out
#SBATCH --error=out/graph_train_2.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=0.5 problem/model=ff_net_recall_2d problem=graphs name=graphs_abalation problem.test_data=6 problem.train_data=6 problem.hyp.lr=0.001 problem.hyp.train_batch_size=150 problem.hyp.test_batch_size=150 problem.hyp.epochs=150
# srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=0.5 problem/model=dt_net_recall_2d problem=graphs name=graphs_abalation problem.test_data=6 problem.train_data=6 problem.hyp.lr=0.01 problem.hyp.train_batch_size=150 problem.hyp.test_batch_size=150 problem.hyp.epochs=150 problem.hyp.lr_throttle=False problem.hyp.optimizer=sgd
# srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=0.5 problem/model=dt_net_recall_2d problem=graphs name=graphs_abalation problem.test_data=6 problem.train_data=6 problem.hyp.lr=0.0001 problem.hyp.train_batch_size=150 problem.hyp.test_batch_size=150 problem.hyp.epochs=150 problem.hyp.lr_throttle=False problem.hyp.optimizer=sgd