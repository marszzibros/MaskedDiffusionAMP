#!/bin/bash
cd "$(dirname "$0")/.."
mkdir -p logs

# --- training arms: one sbatch job each -------------------------------------
TOKENIZERS=${TOKENIZERS:-"xh amide"}      # xh = BRICS, amide = amide_brics
ORDERS=${ORDERS:-"dfs raw"}               # attachment-label numbering
SCHEDULES=${SCHEDULES:-"two single"}      # two = topology/chemistry mask schedules, single = one linear schedule

EPOCHS=${EPOCHS:-200}
BATCH=${BATCH:-64}
HIDDEN=${HIDDEN:-768}
BLOCKS=${BLOCKS:-8}
HEADS=${HEADS:-8}

# --- per-arm sampling sweep: cells inside each job, read by scripts/train.sh -
# Exported so sbatch (--export=ALL by default) carries them to the job; a plain
# shell variable would not reach train.sh.
export STEPS=${STEPS:-"100 500"}
export ETAS=${ETAS:-"0 1 2 5 10 50"}
export GRAMMAR=${GRAMMAR:-"on off"}       # the t=0.5 SAFE grammar repair; "on off" measures its effect
export N=${N:-256}                        # samples per cell
export ATTN=${ATTN:-auto}

cells=0
for _ in $STEPS; do for _ in $ETAS; do for _ in $GRAMMAR; do cells=$((cells+1)); done; done; done
arms=0
for _ in $TOKENIZERS; do for _ in $ORDERS; do for _ in $SCHEDULES; do arms=$((arms+1)); done; done; done
echo "submitting $arms arms x $cells sweep cells x $N samples  (grammar: $GRAMMAR)"

for t in $TOKENIZERS; do
  for o in $ORDERS; do
    for s in $SCHEDULES; do
      sbatch --job-name="AMP_${t}_${o}_${s}" --output="logs/%x_%j.out" \
        scripts/train.sh "$t" "$o" "$s" "$EPOCHS" "$BATCH" "$HIDDEN" "$BLOCKS" "$HEADS"
    done
  done
done
