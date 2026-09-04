#!/bin/bash
#
# Build the tokenizer arms.
#
#   ./run.sh                  # build everything missing, then score
#   ./run.sh --score-only     # re-score what is already built
#   ./run.sh --force          # rebuild even if outputs exist
#
# Three stages. The slicer changes the corpus, the reorder pass rewrites that
# corpus, and only then is a tokenizer trained -- so each stage runs once per
# thing it produces rather than once per arm:
#
#   1  amp.csv    --slicer S-->  safe_S.csv           (one encode per slicer)
#   2  safe_S.csv --reorder-->   safe_S_reorder.csv   (one pass per slicer)
#   3  safe_*.csv --train--->    tok_<arm>_safe.json  (one training per arm)
#
# SLICERS is brics and recap only. hr, rotatable, mmpa and attach overflow the
# SMILES %99 ring-label ceiling -- their corpora decode at 2-62%, and the
# failures concentrate in long peptides, so they cannot be salvaged by
# filtering. Their numbers are already in tokenizers/report.csv.
#
# SPLITTER is "safe" only. With splitter=none BPE learns merges with absolute
# ring numbers baked in (a token valid at residue 27 is useless at residue 12),
# which is the tokenizer-level version of the memorization that collapses
# novelty.
#
# CPU only -- no GPU, no SLURM. Encoding is the slow part (~2-3 min per slicer),
# reordering ~1 min, each tokenizer ~35 s. Both stages skip existing outputs.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$ROOT"

OUT="$ROOT/tokenizers"
AMP_SAFE="$ROOT/molecular_dataset/AMP_SAFE"
AMP_CSV="$ROOT/molecular_dataset/dataset/data/dbaasp/amp.csv"

SLICERS=(brics recap)
SPLITTER=safe
VOCAB=4000
MIN_FREQ=2
SAMPLE=1500

SCORE_ONLY=0
FORCE=0
for a in "$@"; do
  case "$a" in
    --score-only) SCORE_ONLY=1 ;;
    --force)      FORCE=1 ;;
    *) echo "usage: $0 [--score-only] [--force]" >&2; exit 2 ;;
  esac
done

[[ -f "$ROOT/.venv/bin/python" ]] || { echo "no .venv at $ROOT/.venv" >&2; exit 1; }
PY="$ROOT/.venv/bin/python"
[[ -f "$AMP_CSV" ]] || { echo "missing $AMP_CSV (dataset stage 2 output)" >&2; exit 1; }

mkdir -p "$OUT" "$ROOT/logs/tokenizer"
MANIFEST="$OUT/manifest.tsv"

train_tokenizer() {   # <variant> <corpus>
  local variant="$1" corpus="$2"
  local arm="${variant}_${SPLITTER}"
  local tok="$OUT/tok_${arm}.json"
  if [[ -s "$tok" && "$FORCE" -eq 0 ]]; then
    echo "[train ] $arm -- cached"
  else
    echo "[train ] $arm ..."
    if ! "$PY" "$AMP_SAFE/train_safe_tokenizer.py" \
          --in-path "$corpus" --out-path "$tok" \
          --splitter "$SPLITTER" --vocab-size "$VOCAB" --min-frequency "$MIN_FREQ" \
          > "$ROOT/logs/tokenizer/train_${arm}.out" 2>&1; then
      echo "[train ] $arm FAILED -- see logs/tokenizer/train_${arm}.out"; return
    fi
    grep -E "^vocab size" "$ROOT/logs/tokenizer/train_${arm}.out" | sed 's/^/           /'
  fi
  printf '%s\t%s\t%s\t%s\n' "$variant" "$SPLITTER" "$corpus" "$tok" >> "$MANIFEST"
}

if [[ "$SCORE_ONLY" -eq 0 ]]; then
  : > "$MANIFEST"
  for S in "${SLICERS[@]}"; do
    CORPUS="$OUT/safe_${S}.csv"

    # --- stage 1: SMILES -> SAFE -------------------------------------------
    if [[ -s "$CORPUS" && "$FORCE" -eq 0 ]]; then
      echo "[encode] $S -- cached ($(wc -l < "$CORPUS") rows)"
    else
      echo "[encode] $S ..."
      if ! "$PY" "$AMP_SAFE/smiles_to_safe.py" \
            --in-path "$AMP_CSV" --out-path "$CORPUS" \
            --log-path "$ROOT/logs/tokenizer/encode_${S}.log" --slicer "$S" \
            > "$ROOT/logs/tokenizer/encode_${S}.out" 2>&1; then
        echo "[encode] $S FAILED -- see logs/tokenizer/encode_${S}.out"; continue
      fi
      tail -2 "$ROOT/logs/tokenizer/encode_${S}.out" | sed 's/^/           /'
    fi

    # --- stage 2: depth-first fragment reorder ------------------------------
    REORDER="$OUT/safe_${S}_reorder.csv"
    if [[ -s "$REORDER" && "$FORCE" -eq 0 ]]; then
      echo "[reorder] ${S} -- cached"
    else
      echo "[reorder] ${S} ..."
      if ! "$PY" "$ROOT/scripts/reorder_safe.py" \
            --in-path "$CORPUS" --out-path "$REORDER" \
            > "$ROOT/logs/tokenizer/reorder_${S}.out" 2>&1; then
        echo "[reorder] $S FAILED -- see logs/tokenizer/reorder_${S}.out"; continue
      fi
      head -2 "$ROOT/logs/tokenizer/reorder_${S}.out" | sed 's/^/           /'
    fi

    # --- stage 3: one tokenizer per corpus ----------------------------------
    train_tokenizer "$S"            "$CORPUS"
    train_tokenizer "${S}_reorder"  "$REORDER"
  done
fi

echo
echo "=== scoring ==="
"$PY" "$ROOT/scripts/tokenizer_report.py" --dir "$OUT" --sample "$SAMPLE"
