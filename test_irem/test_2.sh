#!/bin/bash
#SBATCH --job-name=test_2
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=test_2.out
#SBATCH --error=test_2.err

source /etc/profile.d/modules.sh
module load CUDA

# srun python3.9 graph_train.py --exp=shortestpath_experiment --train --num_steps=50 --dataset=shortestpath --cuda --infinite --dt --lr 0.0001 --prog --alpha=0.01 --rank=15 --gen_rank=15 --json_name=shortestpath_dt_rank_15_gen_15 --logdir=15_15_1
# srun python3.9 graph_train.py --exp=shortestpath_experiment --train --num_steps=50 --dataset=shortestpath --cuda --infinite --dt --lr 0.0001 --prog --alpha=0.01 --rank=35 --gen_rank=15 --json_name=shortestpath_dt_rank_10_gen_15 --logdir=10_15_1
# srun python3.9 graph_train.py --exp=shortestpath_experiment --train --num_steps=50 --dataset=shortestpath --cuda --infinite --dt --lr 0.0001 --prog --alpha=0.01 --rank=20 --gen_rank=15 --json_name=shortestpath_dt_rank_20_gen_15 --logdir=20_15_1
# srun python3.9 graph_train.py --exp=shortestpath_experiment --train --num_steps=50 --dataset=shortestpath --cuda --infinite --dt --lr 0.0001 --prog --alpha=0.01 --rank=20 --gen_rank=-10 --json_name=shortestpath_dt_rank_20_gen_-10 --logdir=20_-10_1

# python3.9 graph_train.py --exp=shortestpath_2_experiment_01 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_2_experiment_1 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_2_experiment_2 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_2_experiment_3 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_2_experiment_4 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_2_experiment_5 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_2_experiment_6 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_2_experiment_7 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_2_experiment_8 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_2_experiment_9 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_2_experiment_10 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation

# python3.9 graph_train.py --exp=shortestpath_2_experiment_01 --logdir=alpha_large_25 --train --num_steps=10 --dataset=shortestpath --cuda --infinite --dt --lr 0.0001 --prog --alpha=0.01 --json_name dt --rank=25 --gen_rank=60 --vary

# python3.9 graph_train.py --exp=shortestpath_2_experiment_01 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=65 --plot_name=test_25_2 --plot_folder=extreme_extrapolation

python3.9 graph_train.py --exp=shortestpath_2_experiment_01 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
python3.9 graph_train.py --exp=shortestpath_2_experiment_1 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
python3.9 graph_train.py --exp=shortestpath_2_experiment_2 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
python3.9 graph_train.py --exp=shortestpath_2_experiment_3 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
python3.9 graph_train.py --exp=shortestpath_2_experiment_4 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
python3.9 graph_train.py --exp=shortestpath_2_experiment_5 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
python3.9 graph_train.py --exp=shortestpath_2_experiment_6 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
python3.9 graph_train.py --exp=shortestpath_2_experiment_7 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
python3.9 graph_train.py --exp=shortestpath_2_experiment_8 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
python3.9 graph_train.py --exp=shortestpath_2_experiment_9 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation
python3.9 graph_train.py --exp=shortestpath_2_experiment_10 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_25/ --plot --rank=35 --gen_rank=60 --plot_name=test_25_2 --plot_folder=extreme_extrapolation