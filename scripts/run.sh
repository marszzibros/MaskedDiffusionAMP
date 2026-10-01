#!/bin/bash
cd "$(dirname "$0")/.."
mkdir -p logs

TOKENIZERS="xh amide"        # xh = BRICS, amide = amide_brics
ORDERS="dfs raw"             # attachment-label numbering
SCHEDULES="two single"       # two = topology/chemistry mask schedules, single = one linear schedule

EPOCHS=200
BATCH=64
HIDDEN=768
BLOCKS=8
HEADS=8

for t in $TOKENIZERS; do
  for o in $ORDERS; do
    for s in $SCHEDULES; do
      sbatch --job-name="AMP_${t}_${o}_${s}" --output="logs/%x_%j.out" \
        scripts/train.sh "$t" "$o" "$s" "$EPOCHS" "$BATCH" "$HIDDEN" "$BLOCKS" "$HEADS"
    done
  done
done
