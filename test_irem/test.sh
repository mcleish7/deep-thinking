#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:3

module load GCCcore/11.2.0 Python/3.9.6
module load CUDA

srun python3.9 graph_train.py --exp=identity_experiment --train --num_steps=10 --dataset=identity --train --cuda --infinite --recurrent

python3.9 graph_train.py --exp=identity_experiment --train --num_steps=10 --dataset=identity --cuda --infinite --dt