#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=test.out
#SBATCH --error=test.err

source /etc/profile.d/modules.sh
module load CUDA

# srun python3.9 graph_train.py --exp=identity_experiment --num_steps=50 --dataset=identity --cuda --infinite  --resume_iter=10000 --logdir=outputs/10_15 --plot --rank=10 --gen_rank=15
# srun python3.9 graph_train.py --exp=identity_experiment --num_steps=50 --dataset=identity --cuda --infinite  --resume_iter=10000 --logdir=outputs/15_15 --plot --rank=15 --gen_rank=15
# srun python3.9 graph_train.py --exp=identity_experiment --num_steps=50 --dataset=identity --cuda --infinite  --resume_iter=10000 --logdir=outputs/20_15 --plot --rank=20 --gen_rank=15
# srun python3.9 graph_train.py --exp=identity_experiment --num_steps=50 --dataset=identity --cuda --infinite  --resume_iter=10000 --logdir=outputs/20_-10 --plot --rank=20 --gen_rank=-10

# srun python3.9 graph_train.py --exp=identity_experiment --train --num_steps=10 --dataset=identity --train --cuda --infinite --recurrent

# python3.9 graph_train.py --exp=identity_experiment --train --num_steps=10 --dataset=identity --cuda --infinite --dt

# python3.9 graph_train.py --exp=shortestpath_experiment_01 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_30/ --plot --rank=35 --gen_rank=60 --plot_name=test_30_0 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_experiment_1 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_30/ --plot --rank=35 --gen_rank=60 --plot_name=test_30_0 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_experiment_2 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_30/ --plot --rank=35 --gen_rank=60 --plot_name=test_30_0 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_experiment_3 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_30/ --plot --rank=35 --gen_rank=60 --plot_name=test_30_0 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_experiment_4 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_30/ --plot --rank=35 --gen_rank=60 --plot_name=test_30_0 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_experiment_5 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_30/ --plot --rank=35 --gen_rank=60 --plot_name=test_30_0 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_experiment_6 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_30/ --plot --rank=35 --gen_rank=60 --plot_name=test_30_0 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_experiment_7 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_30/ --plot --rank=35 --gen_rank=60 --plot_name=test_30_0 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_experiment_8 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_30/ --plot --rank=35 --gen_rank=60 --plot_name=test_30_0 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_experiment_9 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_30/ --plot --rank=35 --gen_rank=60 --plot_name=test_30_0 --plot_folder=extreme_extrapolation
# python3.9 graph_train.py --exp=shortestpath_experiment_10 --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=alpha_large_30/ --plot --rank=35 --gen_rank=60 --plot_name=test_30_0 --plot_folder=extreme_extrapolation

# python3.9 graph_train.py --exp=identity_experiment --train --num_steps=10 --dataset=identity --train --cuda --infinite --logdir=irem_testing

python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=20 --dataset=shortestpath --train --cuda --infinite --logdir=delete_me --rank=10 --gen_rank=30

# python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=20 --dataset=shortestpath --train --cuda --infinite --logdir=irem_best_2 --rank=10 --gen_rank=30
# python3.9 graph_train.py --exp=identity_experiment --num_steps=20 --dataset=identity --train --cuda --infinite --logdir=irem_best --rank=10 --gen_rank=30
# python3.9 graph_train.py --exp=connected_experiment --num_steps=20 --dataset=connected --train --cuda --infinite --logdir=irem_best --rank=10 --gen_rank=30

# python3.9 graph_train.py --exp=shortestpath_experiment --num_steps=30 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --logdir=irem_testing_35_10/ --plot --rank=35 --gen_rank=30 --plot_name=test_irem_35_10 --plot_folder=irem_plots
# python3.9 ../torch_geom_helper.py