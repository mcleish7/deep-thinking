#!/bin/bash
#SBATCH --job-name=test_pure
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=test_pure_no_mem.out
#SBATCH --error=test_pure_no_mem.err

source /etc/profile.d/modules.sh
module load CUDA

# python3.9 graph_train.py --exp=identity_experiment --train --num_steps=10 --dataset=identity --train --cuda --infinite --num_steps=20 --logdir=best_paths
# python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=5 --dataset=shortestpath --train --cuda --infinite --logdir=5_steps
# python3.9 graph_train.py --exp=connected_experiment --train --num_steps=10 --dataset=connected --train --cuda --infinite --num_steps=20 --logdir=best_paths

python3.9 graph_train.py --exp=identity_experiment --train --num_steps=10 --dataset=identity --train --cuda --infinite --num_steps=20 --logdir=rank_35 --rank=35
python3.9 graph_train.py --exp=shortestpath_experiment --train --num_steps=10 --dataset=shortestpath --train --cuda --infinite --num_steps=20 --logdir=rank_35 --rank=35
python3.9 graph_train.py --exp=connected_experiment --train --num_steps=10 --dataset=connected --train --cuda --infinite --num_steps=20 --logdir=rank_35 --rank=35