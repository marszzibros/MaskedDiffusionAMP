#!/bin/bash
#
# Sample one trained arm across the eta x steps grid.
#
#   ./scripts/sweep_sample.sh <arm> <checkpoint.ckpt>
#   ./scripts/sweep_sample.sh brics_none output/2026.../model-epoch_500.ckpt
#   ./scripts/sweep_sample.sh --dry-run brics_none path/to.ckpt
#
# 6 etas x 3 step counts = 18 runs of 1000 sequences each per arm.
#
# On what eta actually means (DFM.py:399-412):
#     dt          = 1 / steps
#     remask_rate = min(dt * eta, 1.0)   = min(eta / steps, 1.0)
# so the sampler only ever sees the RATIO eta/steps. Once eta >= steps the rate
# clamps to 1.0 and further eta does nothing -- which is why high eta only helps
# when paired with high steps. Saturated cells are flagged but still run, since
# they are what the grid asked for.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$ROOT"

DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi

ARM="${1:?usage: $0 [--dry-run] <arm> <checkpoint>}"
CKPT="${2:?usage: $0 [--dry-run] <arm> <checkpoint>}"

ETAS=(0 10 50 100 200 500)
STEPS=(100 500 1000)
N=1000
BATCH=250

SLICER="${ARM%_*}"; SPLIT="${ARM#*_}"
TOK="tokenizers/tok_${SLICER}_${SPLIT}.json"
CSV="tokenizers/safe_${SLICER}.csv"
OUT="samples/${ARM}"

echo "=== preflight ==="
fail=0
[[ -s "$CKPT" ]] || { echo "  BLOCKED: no checkpoint at $CKPT" >&2; fail=1; }
[[ -s "$TOK"  ]] || { echo "  BLOCKED: no tokenizer at $TOK" >&2; fail=1; }
[[ -s "$CSV"  ]] || { echo "  BLOCKED: no corpus at $CSV" >&2; fail=1; }

# sample.py passes use_charge_filter=, which Xiaohan's DFM.py removed. That is a
# TypeError on EVERY run -- and it fires only after the checkpoint has loaded,
# so on a cluster you lose the queue wait before finding out.
if grep -q "use_charge_filter=args.use_charge_filter" sample.py \
   && ! grep -q "use_charge_filter" DFM.py; then
  echo "  BLOCKED: sample.py:100 passes use_charge_filter= but DFM.py's" >&2
  echo "           generate_sample no longer accepts it -> TypeError every run." >&2
  echo "           Fix: delete that line from sample.py." >&2
  fail=1
fi
[[ "$fail" -eq 0 ]] || { echo; echo "Refusing to sample." >&2; exit 1; }
echo "  ok: $ARM  ($TOK)"

mkdir -p "$OUT"
echo
echo "=== ${#ETAS[@]} etas x ${#STEPS[@]} step counts = $((${#ETAS[@]}*${#STEPS[@]})) runs x $N sequences ==="
for st in "${STEPS[@]}"; do
  for eta in "${ETAS[@]}"; do
    tag="eta${eta}_steps${st}"
    out="${OUT}/${tag}.txt"
    note=""
    if [[ "$eta" -ge "$st" && "$eta" -ne 0 ]]; then
      note="   [saturated: eta/steps >= 1, remask_rate clamps to 1.0]"
    fi
    if [[ -s "$out" ]]; then echo "  $tag -- cached$note"; continue; fi
    echo "  $tag$note"
    [[ "$DRY" -eq 1 ]] && continue
    .venv/bin/python sample.py \
        --checkpoint_path "$CKPT" \
        --tokenizer_path  "$TOK" \
        --safe_csv        "$CSV" \
        --num_samples "$N" \
        --batch_size  "$BATCH" \
        --eta   "$eta" \
        --steps "$st" \
        --output_file "$out" \
      > "logs/sample_${ARM}_${tag}.out" 2>&1 \
      || { echo "    FAILED -- see logs/sample_${ARM}_${tag}.out"; continue; }
  done
done

# Score whatever landed. eval/metrics.py is the moved metrics.py; peptide_metrics
# adds the funnel. Both are optional -- sampling is the deliverable here.
if [[ "$DRY" -eq 0 && -f eval/metrics.py ]]; then
  echo
  echo "=== scoring ==="
  for f in "$OUT"/*.txt; do
    [[ -s "$f" ]] || continue
    .venv/bin/python eval/metrics.py "$f" \
        --label "$(basename "${f%.txt}")" \
        --append "${OUT}/summary.csv" || true
  done
  echo "wrote ${OUT}/summary.csv"
fi
