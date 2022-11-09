#!/bin/bash
#SBATCH --job-name=graph_train
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=out/graph_train.out
#SBATCH --error=out/graph_train.err

source /etc/profile.d/modules.sh
module load CUDA
# srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=1.0 problem/model=dt_net_2d problem=graphs name=graphs_abalation problem.test_data=6 problem.train_data=6 problem.hyp.lr=0.0001 problem.hyp.train_batch_size=200 problem.hyp.test_batch_size=50 problem.hyp.epochs=500
# srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=1.0 problem/model=dt_net_2d problem=graphs name=graphs_abalation problem.test_data=6 problem.train_data=6 problem.hyp.lr=0.0001 problem.hyp.train_batch_size=150 problem.hyp.test_batch_size=50 problem.hyp.epochs=500
# srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=1.0 problem/model=dt_net_2d problem=graphs name=graphs_abalation problem.test_data=6 problem.train_data=6 problem.hyp.lr=0.0001 problem.hyp.train_batch_size=100 problem.hyp.test_batch_size=50 problem.hyp.epochs=500
# srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=1.0 problem/model=dt_net_2d problem=graphs name=graphs_abalation problem.test_data=6 problem.train_data=6 problem.hyp.clip=5 problem.hyp.lr=0.000000001 problem.hyp.train_batch_size=1000 problem.hyp.lr_factor=0.01 problem.hyp.lr_decay=cosine
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=1.0 problem/model=dt_net_recall_2d problem=graphs name=graphs_abalation problem.test_data=6 problem.train_data=6 problem.hyp.lr=0.001 problem.hyp.train_batch_size=100 problem.hyp.test_batch_size=100 problem.hyp.epochs=150