#!/bin/bash
#SBATCH --partition=nvgpu
#SBATCH --constraint=GPU_SKU:H200
#SBATCH --exclude=h2node09
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-06:00:00
#SBATCH --mail-user=jjung2@uvm.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=logs/slurm/eval-%x-%j.out
#SBATCH --error=logs/slurm/eval-%x-%j.err
#
# Score ONE arm at ONE epoch. This NEEDS A GPU, so submit it:
#
#   sbatch --job-name=e150-path_bpe_1024 eval.sh 150 path_bpe_1024
#
# Do NOT run it on a login node. sample.py falls back to CPU when CUDA is
# unavailable rather than failing, so a login-node run does not error -- it
# just takes days, and on most clusters gets killed for hogging the head node.
# Use eval_all.sh to submit every arm at once.
set -euo pipefail

EPOCH="${1:?usage: sbatch eval.sh <epoch> <arm>}"
ARM="${2:?usage: sbatch eval.sh <epoch> <arm>}"
shift 2 || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
mkdir -p logs/slurm samples

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module load cuda/12.6.2
fi
source .venv/bin/activate

N=1000            # +-1.5pp; n=256 gives +-3pp and cannot separate the arms
BATCH=64          # long arms (safe_bpe_159 at 1301 tokens) need the headroom

dir=$(ls -dt output/*-"${ARM}"/ 2>/dev/null | head -1 || true)
[[ -z "$dir" ]] && { echo "no output dir for arm ${ARM}"; exit 1; }
ckpt="${dir%/}/model-epoch_${EPOCH}.ckpt"
[[ -f "$ckpt" ]] || { echo "missing ${ckpt}"; exit 1; }

# Match the denoising steps the run itself used, so sampling work is comparable
# across arms with very different sequence lengths.
STEPS=$(python -c "import json;print(json.load(open('${dir%/}/model_config.json')).get('num_steps',100))")

echo "[eval] arm=${ARM} epoch=${EPOCH} steps=${STEPS} n=${N}"
python -c "import torch;assert torch.cuda.is_available() and torch.zeros(1,device='cuda').is_cuda, 'no usable GPU -- submit this with sbatch'"

out="samples/${ARM}_e${EPOCH}.tsv"
python sample.py --checkpoint_path "$ckpt" --num_samples "$N" \
    --batch_size "$BATCH" --steps "$STEPS" --output_file "$out" "$@"

# Each job writes its OWN summary row. Eight eval jobs appending to one
# sweep_summary.csv concurrently would race: several would each see the file as
# missing and write their own header row. Collect afterwards with:
#   ./eval_all.sh --collect
python metrics.py "$out" --label "${ARM}_e${EPOCH}" --append "samples/${ARM}_e${EPOCH}.summary.csv"
python peptide_metrics.py "$out" | tee "samples/${ARM}_e${EPOCH}.funnel.txt"
