#!/bin/bash
#SBATCH --job-name=test_pure_mem
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=test_mem_35.out
#SBATCH --error=test_mem_35.err

source /etc/profile.d/modules.sh
module load CUDA

python3.9 graph_train.py --exp=identity_experiment --train --num_steps=10 --dataset=identity --train --cuda --infinite --num_steps=20 --logdir=rank_35 --mem --rank=35
python3.9 graph_train.py --exp=shortestpath_experiment --train --num_steps=10 --dataset=shortestpath --train --cuda --infinite --num_steps=20 --logdir=rank_35 --mem --rank=35
python3.9 graph_train.py --exp=connected_experiment --train --num_steps=10 --dataset=connected --train --cuda --infinite --num_steps=20 --logdir=rank_35 --mem --rank=35