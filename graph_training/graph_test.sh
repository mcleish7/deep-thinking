#!/bin/bash
#SBATCH --job-name=graph_test
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=out/graph_test.out
#SBATCH --error=out/graph_test.err

#Whitnee = clip of 0.5 for 100 epochs and batch size 200
#Janese = no clip for 100 epochs and batch size 200
#Carly = no clip for 500 epochs and batch size 200
#Asmaa = no clip for 500 epochs and batch size 150
#Latrica = no clip for 500 epochs and batch size 100
#Junie = no clip for 400 epochs and batch size 150 and lr = 0.01
#Love = no clip for 400 epochs and batch size 150 and lr = 0.001
#Aleya = no clip for 400 epochs and batch size 150 and lr = 0.0001
#Brittin = no clip for 400 epochs and batch size 150 and lr = 0.01 and alpha=0 --very bad
#Ricki = no clip for 400 epochs and batch size 150 and lr = 0.001 and alpha=0 --very good run
#Monica = no clip for 400 epochs and batch size 150 and lr = 0.0001 and alpha=0
# = now only 100 in lr-schedule
#dustless-Florance = SGD, no throttle, 0.5 alpha, lr = 0.001
# sporty-Arifa = SGD, no throttle, 0.5 alpha, lr = 0.01 --very bad
#presto-Lekia = SGD, no throttle, 0.5 alpha, lr = 0.0001
#now normalsing outputs
source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/graphs_abalation/training-barish-Rechelle problem=graphs problem.model.test_iterations.low=0 problem.model.test_iterations.high=750 problem.test_data=6 problem.train_data=6
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/graphs_abalation/training-woodless-Vern problem=graphs problem.model.test_iterations.low=0 problem.model.test_iterations.high=750 problem.test_data=6 problem.train_data=6
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/graphs_abalation/training-valiant-Shandel problem=graphs problem.model.test_iterations.low=0 problem.model.test_iterations.high=750 problem.test_data=6 problem.train_data=6