# seed_sweep

The 3-seed sweep. Every arm is trained three times and sampled at three sampling seeds, so an
arm-vs-arm difference can be read against the spread between identical recipes instead of
against nothing.

Why it exists: in the sweep already on disk, `<arm>` and `<arm>_control` turned out to be the
same recipe differing only in whether a seed was set, and they land up to **0.333 apart** on
strict. That is larger than the tokenizer gap, the schedule gap and every variant effect except
the balanced loss, so none of those claims is currently separable from noise.

## The matrix

8 arms x 3 training seeds = **24 jobs**, one sbatch job each, training then sampling in the same
allocation.

| | |
|---|---|
| tokenizer | `amide`, `xh` (BRICS) |
| label order | `dfs` (ordered), `raw` (unordered) |
| schedule | `single`, `two` |
| training seed | 0, 1, 2 |

Run directories are `<tokenizer>_<order>_<schedule>_seed<N>`, e.g. `amide_dfs_two_seed0`. The
order is **in the name** — it has to be, or a `raw` run would land on the `dfs` run's folder and
then skip training because `model-final.ckpt` is already there.

These names do not collide with the 12 older `<tokenizer>_<schedule>_plain_seed<N>` runs, which
are left where they are. `collect.py` reads only the new four-part names, so the old runs are
reported as skipped rather than silently mixed in.

Self-conditioning used to be an axis here (`plain` / `sc`). It was removed from the main code
before the GitHub push, so these scripts no longer build it. To re-run that axis, re-apply the
patch kept in the `experiments/` archive outside the repo
(`MaskedDiffusionAMP_experiments_2026-10-19.tar.gz`, at `experiments/_attic/`).

Training: 200 epochs, batch 64, 768 wide, 8 blocks, 8 heads, `--num_samples 0`, matching the
older arms so the cells stay comparable.

## The sampling grid

500 steps only (100 loses to 500 on both schedules), repair on **and** off, every cell repeated
at sampling seeds 0, 1, 2.

| schedule | cells per sampling seed | why |
|---|---|---|
| `single` | `base_eta` {5, 20} x repair {on, off} = 4 | no token classes exist, so `topo_eta` and `chem_eta` do nothing; they are pinned to 0 |
| `two` | `topo_eta` {0, 10, 20} x `chem_eta` {0, 10, 20} x repair {on, off} = 18 | `base_eta` tracks `chem_eta`, as the existing `sweep_eta` cells do, so the two grids line up |

Cells land in `output/<arm>_seed<N>/sweep_seed<M>/`, one folder per sampling seed, which keeps
`sweep*/steps*.csv` working for `evaluation/sweep.py` and keeps the seed readable from the path.

## Cost

From this repo's own logs: **10.6 GPU-h per training**, **16 min per 256-sample cell at 500 steps**.

| | jobs | cells | GPU-h |
|---|---|---|---|
| training | 24 | — | 254 |
| sampling, `single` arms | 12 | 12 each = 144 | 38 |
| sampling, `two` arms | 12 | 54 each = 648 | 173 |
| **total** | **24** | **792** | **~465** |

Per job: a `single` arm is about 14 h, a `two` arm about 25 h, both inside the 48 h wall limit.

Disk: a checkpoint is **1.0 GB**, so `SAVE_EVERY` matters. It defaults to 200 here, which equals
`EPOCHS` and therefore keeps only `model-final.ckpt` — 24 GB for the whole sweep. The older runs
used `--save_every 50`, i.e. 4 GB per run, which would be 96 GB across 24 jobs.

Half the budget goes to the `raw` arms, which previously came out 70-97% disconnected. That
effect is far larger than the 0.333 strict seed spread, so if the only question is whether
ordering matters, `TSEEDS=0 ORDERS=raw` settles it for about a third of the cost.

**The 3x3 chemistry grid is two thirds of the sampling budget.** On the single seed measured so
far, `chem_eta` moved strict by 0.012 across its whole range while `topo_eta` moved it 0.129.
Keeping all nine combinations re-tests that properly at 3 seeds, which is the point; if the
budget needs cutting later, `CHEM_ETAS=0` drops 36 cells per `two` job and saves about
115 GPU-h without touching any other conclusion.

## Run it

```bash
bash scripts/seed_sweep/run_all.sh --dry      # print the matrix, cost and disk; submit nothing
bash scripts/seed_sweep/run_all.sh            # all 24

SCHEDULES=single bash scripts/seed_sweep/run_all.sh    # the cheap 12 first (~14 h each)
ORDERS=dfs       bash scripts/seed_sweep/run_all.sh    # ordered labels only (the old 12-job matrix)
TSEEDS=0         bash scripts/seed_sweep/run_all.sh    # one seed of everything, as a pilot
OUT_ROOT=output/fresh bash scripts/seed_sweep/run_all.sh   # a separate tree
```

Submit from the VACC login node — `sbatch` is not on the workstation, and the workstation cannot
train at all (its driver is 12.6; the pinned torch wants CUDA 13). These scripts live under
`scripts/` precisely so they are tracked: a `git pull` on VACC brings them, with no rsync step.

Both scripts are resumable: training is skipped when `model-final.ckpt` exists and a cell is
skipped when its `.txt` exists, so a job that hits the wall clock is just resubmitted.

Every knob is an environment variable — `TOKENIZERS ORDERS SCHEDULES TSEEDS SAMPLE_SEEDS
BASE_ETAS TOPO_ETAS CHEM_ETAS GRAMMAR STEPS N EPOCHS SAVE_EVERY BATCH HIDDEN BLOCKS HEADS
OUT_ROOT ATTN`.

## Collect the results

```bash
python scripts/seed_sweep/collect.py --summary      # -> results/seed_sweep.csv
```

One row per cell, with `tseed` and `sseed` as columns, so the two noise sources separate: vary
`tseed` at a matched cell for the training spread, vary `sseed` for the sampling spread. The
`--summary` table prints the mean max-min across training seeds per arm, which is the floor any
arm-vs-arm claim has to clear.

