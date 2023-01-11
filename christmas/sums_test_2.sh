#!/bin/bash
#SBATCH --job-name=sums_test_1
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=out/test_sums_1.out
#SBATCH --error=out/test_sums_1.err

source /etc/profile.d/modules.sh
module load CUDA

srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.hyp.alpha=0.6 problem/model=dt_net_recall_1d problem=prefix_sums name=test_prefix_alpha_3 problem.test_data=512 problem.train_data=32 problem.model.test_iterations.low=0 problem.model.test_iterations.high=500 problem.model.model_path=../../../outputs/prefix_alpha_3/training-dotted-Jeronica

srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.hyp.alpha=0.7 problem/model=dt_net_recall_1d problem=prefix_sums name=test_prefix_alpha_3 problem.test_data=512 problem.train_data=32 problem.model.test_iterations.low=0 problem.model.test_iterations.high=500 problem.model.model_path=../../../outputs/prefix_alpha_3/training-tinny-Jonel

srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.hyp.alpha=0.8 problem/model=dt_net_recall_1d problem=prefix_sums name=test_prefix_alpha_3 problem.test_data=512 problem.train_data=32 problem.model.test_iterations.low=0 problem.model.test_iterations.high=500 problem.model.model_path=../../../outputs/prefix_alpha_3/training-leary-Leilani

srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.hyp.alpha=0.9 problem/model=dt_net_recall_1d problem=prefix_sums name=test_prefix_alpha_3 problem.test_data=512 problem.train_data=32 problem.model.test_iterations.low=0 problem.model.test_iterations.high=500 problem.model.model_path=../../../outputs/prefix_alpha_3/training-expired-Evyan

srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.hyp.alpha=1.0 problem/model=dt_net_recall_1d problem=prefix_sums name=test_prefix_alpha_3 problem.test_data=512 problem.train_data=32 problem.model.test_iterations.low=0 problem.model.test_iterations.high=500 problem.model.model_path=../../../outputs/prefix_alpha_3/training-wrathful-Demetreus