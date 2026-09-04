#!/bin/bash
#
# Submit one evaluation job per arm.
#
#   ./eval_all.sh 150                 # every arm that has an epoch-150 ckpt
#   ./eval_all.sh 150 path_bpe_1024   # just this one
#   DRY=1 ./eval_all.sh 150           # print, submit nothing
#
# Score every arm at the SAME epoch or the comparison means nothing. Find the
# highest epoch they all reached:
#   ls output/*/model-epoch_*.ckpt | sed 's/.*epoch_//;s/\.ckpt//' | sort -n | uniq -c
set -euo pipefail

# Launched as ./scripts/<name>.sh; every path below is relative to the repo root.
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"
if [[ "${1:-}" == "--collect" ]]; then
  out=sweep_summary.csv
  first=$(ls samples/*.summary.csv 2>/dev/null | head -1)
  [[ -z "$first" ]] && { echo "no per-job summaries in samples/ yet"; exit 1; }
  head -1 "$first" > "$out"
  for f in samples/*.summary.csv; do tail -n +2 "$f" >> "$out"; done
  echo "merged $(ls samples/*.summary.csv | wc -l) rows -> $out"
  column -s, -t < "$out" 2>/dev/null || cat "$out"
  exit 0
fi

EPOCH="${1:?usage: ./eval_all.sh <epoch> [arm ...]   |   ./eval_all.sh --collect}"
shift || true

ARMS=("$@")
if [[ ${#ARMS[@]} -eq 0 ]]; then
  # only directories that actually carry an arm suffix; the older
  # timestamp-only runs predate --arm and have no comparable configuration
  mapfile -t ARMS < <(ls -d output/*/ 2>/dev/null \
    | grep -oP '(?<=/)[0-9]{8}-[0-9]{6}-\K[^/]+' | sort -u)
fi

mkdir -p logs/slurm
for arm in "${ARMS[@]}"; do
  dir=$(ls -dt output/*-"${arm}"/ 2>/dev/null | head -1 || true)
  [[ -z "$dir" || ! -f "${dir%/}/model-epoch_${EPOCH}.ckpt" ]] && \
    { echo "  skip ${arm}: no epoch-${EPOCH} checkpoint"; continue; }
  cmd=(sbatch --job-name="e${EPOCH}-${arm}" eval.sh "$EPOCH" "$arm")
  if [[ -n "${DRY:-}" ]]; then echo "  ${cmd[*]}"; else echo "  $("${cmd[@]}")  <- ${arm}"; fi
done
