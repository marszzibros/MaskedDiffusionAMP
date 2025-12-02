#!/bin/bash

#SBATCH --partition=hgnodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=01-23:59:59
#SBATCH --job-name=MASK_AMP
#SBATCH --mail-user=jjung2@uvm.edu
#SBATCH --mail-type=ALL


cd ${SLURM_SUBMIT_DIR}

source ~/.bashrc
module load cuda/12.2.2
conda activate mask_diffusion

cd ${SLURM_SUBMIT_DIR}

python3 trainer.py $FUSION_ARG $CAT_ARG
