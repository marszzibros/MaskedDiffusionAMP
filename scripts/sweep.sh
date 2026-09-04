#!/bin/bash
#
# Submit the tokenizer sweep -- one SLURM job per arm.
#
#   ./scripts/sweep.sh                       # all eight arms, full 501 epochs
#   ./scripts/sweep.sh --epochs 150          # short ranking pass first (recommended)
#   ./scripts/sweep.sh safe_bpe_159 smiles_ape_1030    # just these arms
#   DRY=1 ./scripts/sweep.sh                 # print the sbatch lines, submit nothing
#
# Recommended order of operations: run everything to 150 epochs (~1/3 the cost),
# rank the arms, then re-submit only the top few to 501. Validity ranks stably
# well before convergence.
set -euo pipefail

# Launched as ./scripts/<name>.sh; every path below is relative to the repo root.
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

ALL=(
  safe_bpe_159        # SAFE, the BPE that ships with the data
  safe_ape_518        # SAFE, APE
  safe_ape_1030       # SAFE, APE  -- predicted ~= safe_ape_518, see note below
  smiles_ape_70       # SMILES, APE
  smiles_ape_134      # SMILES, APE
  smiles_ape_262      # SMILES, APE
  smiles_ape_518      # SMILES, APE
  smiles_ape_1030     # SMILES, APE
)

ARMS=(); EXTRA=()
for a in "$@"; do
  case "$a" in
    -*) EXTRA+=("$a") ;;
    *)  if [[ ${#EXTRA[@]} -gt 0 ]]; then EXTRA+=("$a"); else ARMS+=("$a"); fi ;;
  esac
done
[[ ${#ARMS[@]} -eq 0 ]] && ARMS=("${ALL[@]}")

mkdir -p logs/slurm
echo "submitting ${#ARMS[@]} arm(s)${EXTRA[*]:+ with extra flags: ${EXTRA[*]}}"
for arm in "${ARMS[@]}"; do
  cmd=(sbatch --job-name="$arm" scripts/train.sh "$arm" ${EXTRA[@]+"${EXTRA[@]}"})
  if [[ -n "${DRY:-}" ]]; then
    echo "  ${cmd[*]}"
  else
    out=$("${cmd[@]}")
    echo "  ${arm}: ${out}"
  fi
done

cat <<'NOTE'

Pre-registered predictions (from the cross-token constraint budget):
  1. safe_ape_518 ~= safe_ape_1030 -- a bigger SAFE vocabulary absorbs branches
     but almost no ring bonds (34.7 -> 34.6), so the two should land together.
  2. The WORST SMILES arm beats the BEST SAFE arm (23.5 vs 37.6 constraints).
  3. SMILES improves monotonically with vocabulary size.
If any of these fails, the constraint-budget model is wrong -- which is a more
useful result than the ranking.

Score every arm the same way once they finish:
  python sample.py --checkpoint_path output/<dir>/model-epoch_150.ckpt \
      --num_samples 1000 --output_file samples/<arm>.tsv
  python metrics.py samples/<arm>.tsv --label <arm> --append results/sweep_summary.csv
  python peptide_metrics.py samples/<arm>.tsv
NOTE
