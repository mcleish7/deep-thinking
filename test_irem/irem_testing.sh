#!/bin/bash
#SBATCH --job-name=irem_test
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=irem_testing.out
#SBATCH --error=irem_testing.err

source /etc/profile.d/modules.sh
module load CUDA

python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=irem_best_2/ --rank=10 --gen_rank=10 
# python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=irem_best/ --rank=30 --gen_rank=10 
# python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=irem_best/ --rank=50 --gen_rank=10