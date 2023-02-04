#!/bin/bash
#SBATCH --job-name=mem
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=mem.out
#SBATCH --error=mem.err

python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=10 --dataset=shortestpath --train --cuda --infinite --num_steps=20 --logdir=no_mem

python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=10 --dataset=shortestpath --train --cuda --infinite --num_steps=20 --logdir=mem --mem