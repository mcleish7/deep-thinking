#!/bin/bash
#SBATCH --job-name=chess_3
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=slurm_out_chess/chess_3.out
#SBATCH --error=slurm_out_chess/chess_3.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/train_model.py problem.hyp.alpha=0 problem/model=dt_net_2d problem=chess name=chess_ablation