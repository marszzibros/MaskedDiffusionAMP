#!/bin/bash
# One arm: train, then sample and score the sweep. Submitted by run.sh, from the repo root:
#   sbatch scripts/train.sh TOKENIZER ORDER SCHEDULE EPOCHS BATCH HIDDEN BLOCKS HEADS

#SBATCH --partition=nvgpu
#SBATCH --constraint="GPU_SKU:RTX6000"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=23:59:59
#SBATCH --job-name=AMP_amide

set -e
TOKENIZER=$1; ORDER=$2; SCHEDULE=$3; EPOCHS=$4; BATCH=$5; HIDDEN=$6; BLOCKS=$7; HEADS=$8

VENV=${VENV:-.venv}                           # made once on the login node: uv sync
ATTN=${ATTN:-auto}                            # auto: flash-attn if it runs on this GPU, else sdpa
STEPS=${STEPS:-100 500}                       # the sweep
ETAS=${ETAS:-0 1 2 5 10 50}
N=${N:-256}                                   # samples per cell
OUT_ROOT=${OUT_ROOT:-output}
RUN=$OUT_ROOT/${TOKENIZER}_${ORDER}_${SCHEDULE}

cd "${SLURM_SUBMIT_DIR}"

module load cuda/13.0.2
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1 AMP_ATTN_BACKEND=$ATTN

echo "== $TOKENIZER / $ORDER labels / $SCHEDULE schedule: $EPOCHS epochs, batch $BATCH, ${BLOCKS}x${HEADS}x${HIDDEN} -> $RUN"
echo "== node $(hostname)  job ${SLURM_JOB_ID:-local}  $(date -Is)"
python -c "import torch; (torch.zeros(1, device='cuda') + 1).item(); print('GPU ok:', torch.cuda.get_device_name(0))"

# ---- train (skipped when the run already finished) ----
if [ ! -f "$RUN/model-final.ckpt" ]; then
    python -m training.train \
        --tokenizer "$TOKENIZER" --order "$ORDER" --schedule "$SCHEDULE" \
        --epochs "$EPOCHS" --batch_size "$BATCH" \
        --hidden_size "$HIDDEN" --n_blocks "$BLOCKS" --n_heads "$HEADS" \
        --attn_backend "$ATTN" --save_every 50 --num_samples 0 \
        --out_dir "$RUN" ${TRAIN_EXTRA:-}
fi

# ---- sweep: sample each (steps, eta) cell and score it ----
# "two" schedule: eta moves the topology tokens only. "single": there are no topology tokens, so all three move.
mkdir -p "$RUN/sweep"
for steps in $STEPS; do
    for eta in $ETAS; do
        cell="$RUN/sweep/steps${steps}_eta${eta}"
        [ -f "$cell.txt" ] && continue
        if [ "$SCHEDULE" = two ]; then eta_flags="--topo_eta $eta"; else eta_flags="--topo_eta $eta --chem_eta $eta --base_eta $eta"; fi
        python -m training.sample \
            --checkpoint_path "$RUN/model-final.ckpt" --num_samples "$N" --batch_size 64 \
            --steps "$steps" $eta_flags --seed 0 --attn_backend "$ATTN" --output_file "$cell.csv"
        python evaluation/run.py "$cell.csv" > "$cell.txt"
    done
done

echo "== results $(date -Is)"
for f in "$RUN"/sweep/steps*_eta*.txt; do echo; echo "-- $(basename "$f" .txt)"; tail -n +2 "$f"; done
