#!/bin/bash

#SBATCH --partition=hgnodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=01-23:59:59
#SBATCH --job-name=DFM_AMP
#SBATCH --mail-user=xzhang31@uvm.edu
#SBATCH --mail-type=ALL

module load cuda/12.2.2
python3 trainer.py
