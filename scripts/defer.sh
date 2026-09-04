#!/bin/bash
#SBATCH --partition=nvgpu
#SBATCH --constraint=GPU_SKU:RTX6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH --mail-user=jjung2@uvm.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=logs/slurm/defer-%x-%j.out
#SBATCH --error=logs/slurm/defer-%x-%j.err
#
# Does holding the ring labels back until the end raise validity?
#
#   sbatch --job-name=defer150 scripts/defer.sh 150
#   sbatch --job-name=defer150 scripts/defer.sh 150 safe_bpe_159
#   N=200 MODES="base defer" sbatch scripts/defer.sh 150     # quick look
#
# Four arms, all from the SAME checkpoint -- this changes nothing but the
# sampler, so any difference is the sampler:
#
#   base          confidence order, ring labels committed whenever they come up
#   defer         confidence order + ring labels held back, resolved sequentially
#   random        random order, no veto        <- the "adversarially bad sampler"
#   random_defer  random order + veto          <- does the veto rescue it?
#
# The last pair is the one worth waiting for. If the veto only helps the random
# arm, then confidence ordering was already doing this implicitly and there is
# nothing here to adopt. If it helps both, it is doing something MaskGIT is not.
#
# Only safe_bpe_159 is supported and that is deliberate: its tokenizer keeps
# ring labels as standalone tokens, so the veto can hold back a bond without
# holding back the atoms around it. The "free" BPE arms merge labels into tokens
# like `N1.C1(=O)[C@H](CC`, where deferring a label means deferring a whole
# residue -- sample.py refuses those rather than quietly running a different
# experiment. `python ring_tokens.py --arm <arm>` prints the verdict.
set -euo pipefail

EPOCH="${1:?usage: sbatch scripts/defer.sh <epoch> [arm]}"
ARM="${2:-safe_bpe_159}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
mkdir -p logs/slurm samples results

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module load cuda/12.6.2
fi
source .venv/bin/activate

N="${N:-1000}"                       # +-1.5pp; below ~500 the arms do not separate
BATCH="${BATCH:-32}"                 # safe_bpe_159 is long; the deferred phase adds no memory
COMMIT="${COMMIT:-1}"                # 1 = fully sequential. This is the part that works.
MODES="${MODES:-base defer random random_defer}"

dir=$(ls -dt output/*-"${ARM}"/ 2>/dev/null | head -1 || true)
if [[ -z "$dir" ]]; then
    echo "no output dir matching output/*-${ARM}/"
    echo "train the arm first:   ./scripts/sweep.sh ${ARM} --epochs 150"
    exit 1
fi
ckpt="${dir%/}/model-epoch_${EPOCH}.ckpt"
[[ -f "$ckpt" ]] || { echo "missing ${ckpt}"; ls "${dir%/}"/model-epoch_*.ckpt; exit 1; }

# Same denoising budget as the run itself, so the only thing that differs
# between the arms below is which positions get committed when.
STEPS=$(python -c "import json;print(json.load(open('${dir%/}/model_config.json')).get('num_steps',100))")

echo "[defer] arm=${ARM} epoch=${EPOCH} steps=${STEPS} n=${N} commit=${COMMIT}"
python -c "import torch;assert torch.cuda.is_available() and torch.zeros(1,device='cuda').is_cuda,'no usable GPU -- submit with sbatch'"

# Build (or reuse) the ring-label id set once, before the loop, so a problem
# with it fails here rather than four times over.
python ring_tokens.py --arm "${ARM}"

for mode in $MODES; do
    case "$mode" in
        base)         flags=(--unmask_order confidence) ;;
        defer)        flags=(--unmask_order confidence --defer_rings --defer_commit "$COMMIT") ;;
        random)       flags=(--unmask_order random) ;;
        random_defer) flags=(--unmask_order random --defer_rings --defer_commit "$COMMIT") ;;
        *) echo "unknown mode ${mode}"; exit 1 ;;
    esac

    out="samples/${ARM}_e${EPOCH}_${mode}.tsv"
    echo
    echo "=== ${mode} -> ${out}"
    python sample.py --checkpoint_path "$ckpt" --num_samples "$N" \
        --batch_size "$BATCH" --steps "$STEPS" --arm "$ARM" \
        --output_file "$out" "${flags[@]}"

    # One summary row per mode, in its own file. Four jobs appending to a shared
    # CSV would race on the header; eval_all.sh --collect gathers them.
    python metrics.py "$out" --label "${ARM}_e${EPOCH}_${mode}" \
        --append "samples/${ARM}_e${EPOCH}_${mode}.summary.csv"
    python peptide_metrics.py "$out" | tee "samples/${ARM}_e${EPOCH}_${mode}.funnel.txt"
done

echo
echo "=== validity by sampler ==="
for mode in $MODES; do
    f="samples/${ARM}_e${EPOCH}_${mode}.tsv"
    [[ -f "$f" ]] || continue
    tot=$(wc -l < "$f")
    bad=$(grep -c 'INVALID' "$f" || true)
    awk -v m="$mode" -v t="$tot" -v b="$bad" \
        'BEGIN{printf "  %-14s %6.2f%%  (%d/%d parsed)\n", m, 100*(t-b)/t, t-b, t}'
done
echo
echo "compare the funnels too -- validity can rise while the molecules get worse:"
echo "  head -30 samples/${ARM}_e${EPOCH}_*.funnel.txt"
