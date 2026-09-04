#!/bin/bash
#
# Tokenizer grid: 6 slicers x 2 splitters = 12 arms.
#
#   ./run.sh                  # run every arm, then score
#   ./run.sh --score-only     # re-score what is already built
#   ./run.sh --force          # rebuild even if outputs exist
#
# Two stages, because the slicer changes the corpus and the splitter only
# changes the tokenizer:
#
#   stage 1  amp.csv --slicer S--> tokenizers/safe_S.csv        (6 encodes)
#   stage 2  safe_S.csv --splitter P--> tokenizers/tok_S_P.json (12 trainings)
#
# CPU only -- no GPU, no SLURM. Encoding is the slow part (~2-3 min per slicer);
# each tokenizer training is ~35 s. Whole grid lands around 25-40 min cold, and
# both stages skip work that already exists so reruns are cheap.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$ROOT"

OUT="$ROOT/tokenizers"
AMP_SAFE="$ROOT/molecular_dataset/AMP_SAFE"
AMP_CSV="$ROOT/molecular_dataset/dataset/data/dbaasp/amp.csv"

SLICERS=(hr rotatable recap mmpa attach brics)
SPLITTERS=(safe none)

# Upper bound only. Under splitter=safe the SAFE pre-tokenizer splits to single
# atoms and BPE saturates ~159 regardless; under splitter=none the merges are
# real and this bound actually binds.
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

if [[ ! -f "$ROOT/.venv/bin/python" ]]; then
  echo "no .venv at $ROOT/.venv -- this grid needs the project venv" >&2; exit 1
fi
PY="$ROOT/.venv/bin/python"

if [[ ! -f "$AMP_CSV" ]]; then
  echo "missing $AMP_CSV (dataset stage 2 output)" >&2; exit 1
fi

mkdir -p "$OUT" "$ROOT/logs/tokenizer"
MANIFEST="$OUT/manifest.tsv"

if [[ "$SCORE_ONLY" -eq 0 ]]; then
  : > "$MANIFEST"

  # ---- stage 1: SMILES -> SAFE, once per slicer -------------------------
  for S in "${SLICERS[@]}"; do
    CORPUS="$OUT/safe_${S}.csv"
    if [[ -s "$CORPUS" && "$FORCE" -eq 0 ]]; then
      echo "[encode] $S -- cached ($(wc -l < "$CORPUS") rows)"
    else
      echo "[encode] $S ..."
      # smiles_to_safe.py chdir's to molecular_dataset/, so pass absolute paths.
      # A slicer that fails on this corpus should not kill the whole grid.
      if ! "$PY" "$AMP_SAFE/smiles_to_safe.py" \
            --in-path  "$AMP_CSV" \
            --out-path "$CORPUS" \
            --log-path "$ROOT/logs/tokenizer/encode_${S}.log" \
            --slicer   "$S" > "$ROOT/logs/tokenizer/encode_${S}.out" 2>&1; then
        echo "[encode] $S FAILED -- see logs/tokenizer/encode_${S}.out; skipping arm"
        continue
      fi
      tail -2 "$ROOT/logs/tokenizer/encode_${S}.out" | sed 's/^/           /'
    fi

    # ---- stage 2: train a tokenizer per splitter ------------------------
    for P in "${SPLITTERS[@]}"; do
      TOK="$OUT/tok_${S}_${P}.json"
      if [[ -s "$TOK" && "$FORCE" -eq 0 ]]; then
        echo "[train ] $S/$P -- cached"
      else
        echo "[train ] $S/$P ..."
        if ! "$PY" "$AMP_SAFE/train_safe_tokenizer.py" \
              --in-path  "$CORPUS" \
              --out-path "$TOK" \
              --splitter "$P" \
              --vocab-size "$VOCAB" \
              --min-frequency "$MIN_FREQ" \
              > "$ROOT/logs/tokenizer/train_${S}_${P}.out" 2>&1; then
          echo "[train ] $S/$P FAILED -- see logs/tokenizer/train_${S}_${P}.out"
          continue
        fi
        grep -E "^vocab size" "$ROOT/logs/tokenizer/train_${S}_${P}.out" | sed 's/^/           /'
      fi
      printf '%s\t%s\t%s\t%s\n' "$S" "$P" "$CORPUS" "$TOK" >> "$MANIFEST"
    done
  done
fi

# ---- stage 3: score every arm that got built ----------------------------
echo
echo "=== scoring ==="
"$PY" "$ROOT/scripts/tokenizer_report.py" --dir "$OUT" --sample "$SAMPLE"
