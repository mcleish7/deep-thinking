#!/bin/bash
#SBATCH --job-name=test_23
#SBATCH --partition=gpu-batch
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=out/test_23.out
#SBATCH --error=out/test_23.err

source /etc/profile.d/modules.sh
module load CUDA
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-blowhard-Thayer problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-bughouse-Lakevia problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-couthy-Tommi problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-goyish-Lorean problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-muckle-Koury problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-pagan-Tiffnay problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-rubbly-Naseem problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59
srun python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../outputs/mazes_ablation/training-unpierced-Sherkia problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=1000 problem.test_data=59