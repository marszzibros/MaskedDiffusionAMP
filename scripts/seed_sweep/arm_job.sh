#!/bin/bash
# One arm at one training seed: train, then sample every sweep cell at every sampling seed and score it.
# Submitted by run_all.sh, from the repo root:
#   sbatch scripts/seed_sweep/arm_job.sh TOKENIZER ORDER SCHEDULE TSEED
#
# Resumable: training is skipped when model-final.ckpt exists, and a cell is skipped when its .txt
# exists, so a job that hits the wall clock can simply be resubmitted.

#SBATCH --partition=nvgpu
#SBATCH --constraint="GPU_SKU:RTX6000"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=1-23:59:59
#SBATCH --job-name=AMPss

set -e
TOKENIZER=$1; ORDER=$2; SCHEDULE=$3; TSEED=$4
[ -n "$TSEED" ] || { echo "usage: arm_job.sh TOKENIZER {dfs|raw} {single|two} TSEED"; exit 2; }

VENV=${VENV:-.venv}
ATTN=${ATTN:-auto}
EPOCHS=${EPOCHS:-200}
SAVE_EVERY=${SAVE_EVERY:-200}              # 200 = only model-final.ckpt; each one is 1.0 GB
BATCH=${BATCH:-64}
HIDDEN=${HIDDEN:-768}
BLOCKS=${BLOCKS:-8}
HEADS=${HEADS:-8}
N=${N:-256}                                # samples per cell
STEPS=${STEPS:-500}                        # 100 loses to 500 on both schedules
SAMPLE_SEEDS=${SAMPLE_SEEDS:-"0 1 2"}
BASE_ETAS=${BASE_ETAS:-"5 20"}             # single schedule: only base_eta does anything
TOPO_ETAS=${TOPO_ETAS:-"0 10 20"}          # two schedule
CHEM_ETAS=${CHEM_ETAS:-"0 10 20"}
GRAMMAR=${GRAMMAR:-"on off"}
OUT_ROOT=${OUT_ROOT:-output}

# Order is part of the name: without it a raw run would land on the dfs run and then skip
# training, because model-final.ckpt would already be there.
RUN=$OUT_ROOT/${TOKENIZER}_${ORDER}_${SCHEDULE}_seed${TSEED}

cd "${SLURM_SUBMIT_DIR:-.}"

module load cuda/13.0.2
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1 AMP_ATTN_BACKEND=$ATTN

echo "== $TOKENIZER / $ORDER labels / $SCHEDULE / train seed $TSEED -> $RUN"
echo "== node $(hostname)  job ${SLURM_JOB_ID:-local}  $(date -Is)"
python -c "import torch; (torch.zeros(1, device='cuda') + 1).item(); print('GPU ok:', torch.cuda.get_device_name(0))"

# ---- train (skipped when this run already finished) ----
if [ ! -f "$RUN/model-final.ckpt" ]; then
    python -m training.train \
        --tokenizer "$TOKENIZER" --order "$ORDER" --schedule "$SCHEDULE" \
        --epochs "$EPOCHS" --batch_size "$BATCH" \
        --hidden_size "$HIDDEN" --n_blocks "$BLOCKS" --n_heads "$HEADS" \
        --attn_backend "$ATTN" --save_every "$SAVE_EVERY" --num_samples 0 \
        --seed "$TSEED" --no_wandb \
        --out_dir "$RUN" ${TRAIN_EXTRA:-}
else
    echo "== training already done, skipping"
fi

# ---- sample and score every cell, once per sampling seed ----
# One folder per sampling seed, so `sweep*/steps*.csv` still finds every cell and the seed is
# recoverable from the folder name.
sample_cell() {            # $1 cell path (no extension), rest: flags
    local cell="$1"; shift
    [ -f "$cell.txt" ] && return 0
    python -m training.sample \
        --checkpoint_path "$RUN/model-final.ckpt" --num_samples "$N" --batch_size 64 \
        --steps "$STEPS" --seed "$SSEED" --attn_backend "$ATTN" \
        --output_file "$cell.csv" "$@"
    python evaluation/run.py "$cell.csv" > "$cell.txt"
}

for SSEED in $SAMPLE_SEEDS; do
    SWEEP="$RUN/sweep_seed${SSEED}"
    mkdir -p "$SWEEP"
    for g in $GRAMMAR; do
        if [ "$g" = off ]; then g_flags="--no_grammar_check"; suffix="_nogrammar"
        else g_flags=""; suffix=""; fi
        if [ "$SCHEDULE" = single ]; then
            # no token classes, so topo and chem etas are inert; pin them to 0 and vary base
            for b in $BASE_ETAS; do
                sample_cell "$SWEEP/steps${STEPS}_b${b}${suffix}" \
                    --topo_eta 0 --chem_eta 0 --base_eta "$b" $g_flags
            done
        else
            # base eta tracks chem, as the existing sweep_eta cells do, so the grids are comparable
            for t in $TOPO_ETAS; do
                for c in $CHEM_ETAS; do
                    sample_cell "$SWEEP/steps${STEPS}_t${t}_c${c}_b${c}${suffix}" \
                        --topo_eta "$t" --chem_eta "$c" --base_eta "$c" $g_flags
                done
            done
        fi
    done
done

echo "== results $(date -Is)"
for f in "$RUN"/sweep_seed*/steps*.txt; do echo; echo "-- ${f#$RUN/}"; tail -n +2 "$f"; done
