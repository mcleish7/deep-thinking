#!/bin/bash
#SBATCH --job-name=test_fal # Job name for tracking
#SBATCH --partition=falcon     # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12     # Number of CPU threads used by your job
#SBATCH --mem-per-cpu=1500
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)

#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=test_fal.out # Standard out from your job
#SBATCH --error=test_fal.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

module load CUDA

source /dcs/large/u2004277/irem/irem_env/bin/activate
# which python3.9
# srun python3.9 graph_train.py --exp=identity_experiment --num_steps=50 --dataset=identity --cuda --infinite  --resume_iter=10000 --logdir=outputs/10_15 --plot --rank=10 --gen_rank=15
python3.9 graph_train.py --exp=identity_experiment --num_steps=10 --dataset=identity --cuda --infinite  --logdir=test --rank=10 --gen_rank=15

deactivate