#!/bin/sh -l
# FILENAME: job.sh

#SBATCH -A li4578
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1 
#SBATCH --job-name train_safe
#SBATCH --time=4-00:00:00

module load conda
conda activate maskdiffusion
source .venv/bin/activate
python trainer.py
# ./run_grid_search.sh