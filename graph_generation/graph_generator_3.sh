#!/bin/bash
#SBATCH --job-name=graph_gen_3
#SBATCH --partition=cpu-batch
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1500
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=slurm-outputs/graph_gen_3.out
#SBATCH --error=slurm-outputs/graph_gen_3.err

srun python3.9 core_graph_generation.py --n 12 --x 10000 --sp True --save --embed n2v --train
srun python3.9 core_graph_generation.py --n 12 --x 10000 --sp True --save --embed n2v
srun python3.9 core_graph_generation.py --n 14 --x 10000 --sp True --save --embed n2v --train
srun python3.9 core_graph_generation.py --n 14 --x 10000 --sp True --save --embed n2v
srun python3.9 core_graph_generation.py --n 16 --x 10000 --sp True --save --embed n2v --train
srun python3.9 core_graph_generation.py --n 16 --x 10000 --sp True --save --embed n2v