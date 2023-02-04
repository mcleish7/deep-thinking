#!/bin/bash
#SBATCH --job-name=irem_test
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=irem_testing_sp_mem.out
#SBATCH --error=irem_testing.err

source /etc/profile.d/modules.sh
module load CUDA

echo 10
python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=mem/ --rank=10 --gen_rank=1
echo 20
python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=mem/ --rank=20 --gen_rank=1
echo 30
python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=mem/ --rank=30 --gen_rank=1
echo 40
python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=mem/ --rank=40 --gen_rank=1
echo 50
python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=mem/ --rank=50 --gen_rank=1
echo 60
python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=mem/ --rank=60 --gen_rank=1
echo 70
python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=mem/ --rank=70 --gen_rank=1
echo 80
python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=mem/ --rank=80 --gen_rank=1
echo 90
python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=mem/ --rank=9 0 --gen_rank=1
