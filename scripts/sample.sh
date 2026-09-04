#!/bin/bash
#SBATCH --partition=nvgpu
#SBATCH --constraint=GPU_SKU:RTX6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-23:59:59
#SBATCH --mail-user=jjung2@uvm.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#
# Sample one arm across the eta x steps grid. Two modes:
#
#   submit :  ./scripts/sample.sh <arm> <ckpt>              -> sbatch 18 jobs
#   one cell: ./scripts/sample.sh <arm> <ckpt> <eta> <steps>  (the sbatch entry)
#
# train.sh calls the submit mode when training finishes, so each arm fans out
# into 18 independent jobs that queue in parallel instead of running back to
# back in one allocation.
#
# On what eta means (DFM.py:399-412):
#     dt = 1 / steps ;  remask_rate = min(dt * eta, 1.0) = min(eta / steps, 1.0)
# the sampler only sees the RATIO eta/steps, so once eta >= steps the rate
# clamps to 1.0 and further eta does nothing. Those cells still run -- they are
# part of the requested grid -- but they are flagged at submit time.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$ROOT"

ETAS=(0 10 50 100 200 500)
STEPS=(100 500 1000)
N=500
BATCH=250

ARM="${1:?usage: $0 <arm> <checkpoint> [eta steps]}"
CKPT="${2:?usage: $0 <arm> <checkpoint> [eta steps]}"

# Paths come from dataset.ARMS -- arm names have a variable number of "_" parts
# (brics_safe, brics_reorder_safe) so they cannot be split by hand. The safe
# package prints a banner to STDOUT on import, hence the tagged line.
read -r TOK CSV < <(.venv/bin/python -c "
from dataset import ARMS
import sys
a = '$ARM'
if a not in ARMS: sys.exit(1)
print('ARMPATHS', *ARMS[a])" 2>/dev/null | sed -n 's/^ARMPATHS //p')
[[ -n "${TOK:-}" && -n "${CSV:-}" ]] || { echo "unknown arm '$ARM'" >&2; exit 2; }

# ---------------------------------------------------------------- submit mode
if [[ $# -lt 4 ]]; then
    for f in "$CKPT" "$TOK" "$CSV"; do
        [[ -s "$f" ]] || { echo "missing $f" >&2; exit 1; }
    done
    # sample.py passes use_charge_filter= only if it was never fixed; catch it
    # here rather than after 18 jobs each load a checkpoint and die.
    if grep -q "use_charge_filter=args" sample.py && ! grep -q "use_charge_filter" DFM.py; then
        echo "sample.py passes use_charge_filter= but DFM.py dropped it -> TypeError" >&2
        exit 1
    fi

    mkdir -p logs/slurm "samples/${ARM}"
    echo "submitting ${#ETAS[@]} x ${#STEPS[@]} = $(( ${#ETAS[@]} * ${#STEPS[@]} )) jobs for ${ARM}"
    for st in "${STEPS[@]}"; do
        for eta in "${ETAS[@]}"; do
            tag="eta${eta}_steps${st}"
            if [[ -s "samples/${ARM}/${tag}.txt" ]]; then
                echo "  ${tag} -- already sampled, skipping"; continue
            fi
            note=""
            [[ "$eta" -ge "$st" && "$eta" -ne 0 ]] && note="   [saturated: remask_rate clamps to 1.0]"
            jid=$(sbatch --parsable --job-name="${ARM}_${tag}" \
                         "$0" "$ARM" "$CKPT" "$eta" "$st")
            echo "  ${tag} -> job ${jid}${note}"
        done
    done
    exit 0
fi

# -------------------------------------------------------------- one-cell mode
ETA="$3"
STEPS_N="$4"
TAG="eta${ETA}_steps${STEPS_N}"
OUT="samples/${ARM}/${TAG}.txt"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module load cuda/12.6.2 || true
fi
source .venv/bin/activate

mkdir -p "samples/${ARM}"
echo "[sample] ${ARM}  eta=${ETA}  steps=${STEPS_N}  n=${N}"
echo "[job   ] ${SLURM_JOB_ID:-local} on $(hostname)"

python sample.py \
    --checkpoint_path "$CKPT" \
    --tokenizer_path  "$TOK" \
    --safe_csv        "$CSV" \
    --num_samples "$N" \
    --batch_size  "$BATCH" \
    --eta   "$ETA" \
    --steps "$STEPS_N" \
    --output_file "$OUT"

if [[ -f eval/metrics.py ]]; then
    python eval/metrics.py "$OUT" --label "$TAG" \
        --append "samples/${ARM}/summary.csv" || true
fi
