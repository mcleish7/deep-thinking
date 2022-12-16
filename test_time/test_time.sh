#!/bin/bash
#SBATCH --job-name=test_time
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=test_time.out
#SBATCH --error=test_time.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/test_time.py --which_net=non-prog

srun python3.9 /dcs/large/u2004277/deep-thinking/test_time.py --which_net=prog