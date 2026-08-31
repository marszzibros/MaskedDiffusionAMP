#!/bin/sh -l
# FILENAME: job.sh

#SBATCH -A li4578
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1 
#SBATCH --mem=32G
#SBATCH --job-name exp_scales
#SBATCH --time=4-00:00:00

module load conda
conda activate maskdiffusion
source .venv/bin/activate
python trainer.py
# ./run_grid_search.sh