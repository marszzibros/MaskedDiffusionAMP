#!/bin/bash
# Submit the 3-seed sweep: tokenizer x label order x schedule = 8 arms, each trained at 3
# training seeds = 24 jobs, each training then sampling every cell at 3 sampling seeds.
#
#   bash scripts/seed_sweep/run_all.sh                 # all 24
#   bash scripts/seed_sweep/run_all.sh --dry           # print the matrix and the cost only
#   SCHEDULES=single bash scripts/seed_sweep/run_all.sh        # the cheap half first
#   ORDERS=dfs       bash scripts/seed_sweep/run_all.sh        # ordered labels only
#   TSEEDS=0         bash scripts/seed_sweep/run_all.sh        # one seed of everything
#
# Everything below is overridable from the environment; the exports reach the job because
# sbatch passes the environment on by default.
cd "$(dirname "$0")/../.."
mkdir -p logs

TOKENIZERS=${TOKENIZERS:-"amide xh"}
ORDERS=${ORDERS:-"dfs raw"}
SCHEDULES=${SCHEDULES:-"single two"}
TSEEDS=${TSEEDS:-"0 1 2"}

export EPOCHS=${EPOCHS:-200}
export SAVE_EVERY=${SAVE_EVERY:-200}       # 200 = keep only model-final.ckpt (1.0 GB per run)
export BATCH=${BATCH:-64}
export HIDDEN=${HIDDEN:-768}
export BLOCKS=${BLOCKS:-8}
export HEADS=${HEADS:-8}
export N=${N:-256}
export STEPS=${STEPS:-500}
export SAMPLE_SEEDS=${SAMPLE_SEEDS:-"0 1 2"}
export BASE_ETAS=${BASE_ETAS:-"5 20"}
export TOPO_ETAS=${TOPO_ETAS:-"0 10 20"}
export CHEM_ETAS=${CHEM_ETAS:-"0 10 20"}
export GRAMMAR=${GRAMMAR:-"on off"}
export ATTN=${ATTN:-auto}
export OUT_ROOT=${OUT_ROOT:-output}

DRY=""
[ "$1" = "--dry" ] && DRY=1

# ---- cost, from this repo's own logs: 10.6 GPU-h per training, 16 min per cell ----
n_ss=0; for _ in $SAMPLE_SEEDS; do n_ss=$((n_ss+1)); done
n_g=0;  for _ in $GRAMMAR; do n_g=$((n_g+1)); done
n_b=0;  for _ in $BASE_ETAS; do n_b=$((n_b+1)); done
n_t=0;  for _ in $TOPO_ETAS; do n_t=$((n_t+1)); done
n_c=0;  for _ in $CHEM_ETAS; do n_c=$((n_c+1)); done
cells_single=$(( n_b * n_g * n_ss ))
cells_two=$(( n_t * n_c * n_g * n_ss ))

jobs=0; cells=0
for t in $TOKENIZERS; do for o in $ORDERS; do for s in $SCHEDULES; do for d in $TSEEDS; do
    jobs=$((jobs+1))
    if [ "$s" = single ]; then cells=$((cells+cells_single)); else cells=$((cells+cells_two)); fi
done; done; done; done

train_h=$(( jobs * 106 / 10 ))
sample_h=$(( cells * 16 / 60 ))
echo "jobs: $jobs   cells: $cells   (single arm $cells_single cells, two arm $cells_two cells)"
echo "cost: ~${train_h} GPU-h training + ~${sample_h} GPU-h sampling = ~$(( train_h + sample_h )) GPU-h"
echo "per job: single ~$(( 106 / 10 + cells_single * 16 / 60 ))h, two ~$(( 106 / 10 + cells_two * 16 / 60 ))h  (wall limit 48h)"
# A checkpoint is 1.0 GB. SAVE_EVERY >= EPOCHS keeps only model-final.ckpt.
n_ck=1; [ "$SAVE_EVERY" -lt "$EPOCHS" ] && n_ck=$(( EPOCHS / SAVE_EVERY ))
echo "disk: ~$(( jobs * n_ck )) GB of checkpoints ($n_ck per run at SAVE_EVERY=$SAVE_EVERY), into $OUT_ROOT/"
echo "      free now: $(df -BG --output=avail . | tail -1 | tr -d ' G') GB"
echo

for t in $TOKENIZERS; do
  for o in $ORDERS; do
    for s in $SCHEDULES; do
      for d in $TSEEDS; do
        name="AMPss_${t}_${o}_${s}_s${d}"
        if [ -n "$DRY" ]; then
          echo "would submit: $name   -> ${OUT_ROOT:-output}/${t}_${o}_${s}_seed${d}"
        else
          sbatch --job-name="$name" --output="logs/%x_%j.out" \
            scripts/seed_sweep/arm_job.sh "$t" "$o" "$s" "$d"
        fi
      done
    done
  done
done
