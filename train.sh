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

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
mkdir -p logs/slurm

module load cuda/12.6.2
source .venv/bin/activate

# --- held constant across every arm ---
HIDDEN=1536
BLOCKS=12
HEADS=12
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
# No per-epoch sampling during the sweep. Two reasons: n=5 per epoch is
# statistically useless, and safe_bpe_159 would pay 713 steps x 4 CFG passes
# every epoch while smiles_ape_1030 pays 43 -- which would make wall-clock
# incomparable across arms for a reason that has nothing to do with the
# tokenizer. Score offline from checkpoints with eval.sh instead.
NUM_SAMPLES=0

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
