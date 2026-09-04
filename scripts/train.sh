#!/bin/bash
#SBATCH --partition=nvgpu
#SBATCH --constraint=GPU_SKU:H200
#SBATCH --exclude=h2node09
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=1-23:59:59
#SBATCH --mail-user=jjung2@uvm.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#
# One arm of the tokenizer sweep.
#
#   sbatch --job-name=safe_ape_1030 train.sh safe_ape_1030
#   sbatch --job-name=quick         train.sh smiles_ape_518 --epochs 150
#
# The arm name is the ONLY thing that should differ between sweep runs -- it
# fixes both the tokenizer and the column (see TOKENIZER_ARMS in dataset.py).
# Model size, LR, batch size and seed are pinned below so the comparison is
# actually controlled; change them here and every arm changes together.
set -euo pipefail

ARM="${1:?usage: sbatch train.sh <arm> [extra trainer.py flags...]}"
shift || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
mkdir -p logs/slurm

module load cuda/12.6.2
source .venv/bin/activate

# --- held constant across every arm ---
HIDDEN=1536
BLOCKS=12
HEADS=12
# 1536/12 = 128 head_dim, one of flash-attn's tuned sizes (96 is not).
# n_heads MUST divide hidden_size or einops fails on the first forward --
# the model still constructs, so the error only appears once a batch runs.
BATCH=16
ACC=8
LR=1e-4
SEED=0
EPOCHS=501
# Sequence length differs ~19x between arms, so scale denoising steps with it
# rather than fixing 100 for all -- otherwise short arms idle and long arms
# are under-resolved. 2.5 steps/token puts smiles_ape_1030 near 40 and
# safe_bpe_159 near 775.
STEPS_PER_TOKEN=2.5
# Per-epoch sampling: monitoring, not measurement. Each point is only n=5
# (+-20pp), but 300 epochs is 1500 samples, so a rolling window over ~20 epochs
# is n=100 and the CURVE is informative even where the points are not. It logs
# sample_validity to wandb, so all eight arms overlay for free -- which is how
# you see whether the ranking is stable across epochs rather than guessing.
# It also surfaces a broken arm at epoch 25 instead of after two days.
# Cost is 0.6% of an epoch for smiles_ape_1030, 10.7% for safe_bpe_159; the
# asymmetry does not confound anything, since validity is the outcome and
# wall-clock is not. The real measurement is eval.sh at n=1000.
NUM_SAMPLES=5

echo "[arm] ${ARM}"
echo "[job] ${SLURM_JOB_ID:-local} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python trainer.py \
    --arm "${ARM}" \
    --tag "${ARM}" \
    --epochs   "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --accumulate "${ACC}" \
    --lr "${LR}" \
    --hidden_size "${HIDDEN}" \
    --n_blocks "${BLOCKS}" \
    --n_heads "${HEADS}" \
    --steps_per_token "${STEPS_PER_TOKEN}" \
    --num_samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    "$@"
