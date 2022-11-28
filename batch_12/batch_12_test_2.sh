#!/bin/bash
#SBATCH --job-name=batch_12_test_2
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=out/batch_12_test_2.out
#SBATCH --error=out/batch_12_test_2.err

srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.hyp.alpha=1.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_testing problem.test_data=59 problem.train_data=9 problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.model.model_path=../../../outputs/mazes_ablation/training-mastless-Adriana
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.hyp.alpha=1.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_testing problem.test_data=59 problem.train_data=9 problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.model.model_path=../../../outputs/mazes_ablation/training-nipping-Armin
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.hyp.alpha=1.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_testing problem.test_data=59 problem.train_data=9 problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.model.model_path=../../../outputs/mazes_ablation/training-noteless-Tashawna
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.hyp.alpha=1.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_testing problem.test_data=59 problem.train_data=9 problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.model.model_path=../../../outputs/mazes_ablation/training-rarest-Jesenia