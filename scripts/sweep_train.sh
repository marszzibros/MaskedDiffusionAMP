#!/bin/bash
#
# Launch one training per tokenizer arm.
#
#   ./scripts/sweep_train.sh            # preflight, then sbatch every arm
#   ./scripts/sweep_train.sh --dry-run  # preflight + print the commands only
#   ./scripts/sweep_train.sh --local    # run arms sequentially here, no SLURM
#
# Arms are 2 slicers x {plain, depth-first reordered}, all splitter=safe.
# hr/rotatable/mmpa/attach are excluded (they overflow the %99 ring-label
# ceiling); splitter=none is excluded (its merges memorize absolute ring
# numbers). The reorder axis is the one that actually moves pair span: 144 -> 9.
#
# Everything except the arm is pinned in train.sh so the comparison is controlled.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$ROOT"

ARMS=(brics_safe brics_reorder_safe recap_safe recap_reorder_safe)
# Built by ./run.sh; dataset.ARMS is the authority on paths.

MODE=sbatch
for a in "$@"; do
  case "$a" in
    --dry-run) MODE=dry ;;
    --local)   MODE=local ;;
    *) echo "usage: $0 [--dry-run|--local]" >&2; exit 2 ;;
  esac
done

fail=0
say_fail() { echo "  BLOCKED: $*" >&2; fail=1; }

echo "=== preflight ==="

# 1+2. Ask the code itself rather than grepping for names: trainer.py must
#      accept --arm, and dataset.py must resolve every arm to a loadable
#      tokenizer. With no argparse python silently DISCARDS unknown flags, so
#      all four arms would train the same hardcoded config.
if ! .venv/bin/python - "${ARMS[@]}" <<'PY' 2>/dev/null
import sys, warnings; warnings.filterwarnings("ignore")
src = open("trainer.py").read()
assert "argparse" in src, "trainer.py has no argparse"
from dataset import ARMS, load_tokenizer
for arm in sys.argv[1:]:
    assert arm in ARMS, f"{arm} not in dataset.ARMS"
    tok = load_tokenizer(ARMS[arm][0])
    assert tok.token2id, f"{arm} tokenizer is empty"
PY
then
  say_fail "trainer.py does not accept --arm, or dataset.py cannot resolve an
           arm to a loadable tokenizer. Run the block above by hand to see why."
fi

# 3. every arm needs its tokenizer and corpus on disk. Paths come from
#    dataset.ARMS -- arm names have a variable number of "_" parts
#    (brics_safe vs brics_reorder_safe), so they cannot be split by hand.
while read -r arm tok csv; do
  [[ -s "$tok" ]] || say_fail "$arm: missing $tok  (run ./run.sh)"
  [[ -s "$csv" ]] || say_fail "$arm: missing $csv  (run ./run.sh)"
done < <(.venv/bin/python -c "
from dataset import ARMS
for a in ${ARMS[@]@Q}.split():
    print('ARMPATHS', a, *ARMS[a])" 2>/dev/null | sed -n 's/^ARMPATHS //p')

if [[ "$fail" -ne 0 ]]; then
  cat >&2 <<'MSG'

Refusing to launch. Fix the blockers above first -- launching now would burn
GPU hours on arms that are silently identical.
MSG
  exit 1
fi
echo "  all arms wired up"

echo
echo "=== launching (${MODE}) ==="
mkdir -p logs/slurm
: > sweep_arms.tsv
for arm in "${ARMS[@]}"; do
  case "$MODE" in
    dry)   echo "  sbatch --job-name=$arm train.sh $arm" ;;
    local) echo "  [local] $arm"; bash train.sh "$arm" ;;
    sbatch)
      jid=$(sbatch --parsable --job-name="$arm" train.sh "$arm")
      echo "  $arm -> job $jid"
      printf '%s\t%s\n' "$arm" "$jid" >> sweep_arms.tsv ;;
  esac
done
[[ "$MODE" == "sbatch" ]] && echo && echo "wrote sweep_arms.tsv"
exit 0
