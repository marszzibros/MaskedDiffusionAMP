# AMIDE BRICS

The AMIDE BRICS fragmentation for SAFE peptide strings, plus plain train, sample and evaluation scripts for masked
discrete flow matching with either tokenizer. 

amide_brics cuts one bond per peptide link, the carbonyl-carbon to nitrogen bond, where BRICS also cuts the
N-C(alpha) bond and leaves each backbone nitrogen as a one-atom fragment. Ring-shaped peptides, which have no
cuttable amide, are cut with BRICS. The result is about half as many fragments and attachment labels per molecule.
Because the fragments differ, the vocabulary is refitted; a corpus is always used with its own vocabulary.

```
amp_diffusion/    the model: DiscreteFlowMatching, DiT, data module, SAFE tokenizer, callbacks
dataset_build/    makes the corpora and vocabularies
  slicer.py         encode(smiles, order, rule): cut, number the labels, verify the round trip
  safe_utils.py     the label fix, DFS relabelling and decode check
  build.py          re-cuts the shipped corpus and fits a vocabulary
training/         train.py, sample.py, variants.py (which corpus and vocabulary to use)
evaluation/       run.py scores a sample file; one file per metric
scripts/          run.sh submits all arms, train.sh trains one arm and sweeps it
data/             everything the code reads (below)
pyproject.toml, uv.lock
```

## Environment

```bash
uv sync
```

`uv.lock` pins torch 2.14.1 from the CUDA 13.0 index (`pytorch-cu130` in `pyproject.toml`). That build carries kernels for
Blackwell GPUs such as the RTX PRO 6000, and needs a driver that supports CUDA 13. For an older GPU or driver, change the
index to cu126 and run `uv lock`. flash-attn is an optional extra (`uv sync --extra flash`) and needs a wheel built for the
GPU; without it, use `--attn_backend sdpa` (`auto`, the default in `scripts/train.sh`, falls back to sdpa on its own).

## Data

| Folder | What it is |
|---|---|
| `data/dbaasp/` | the peptide and activity tables (`amp.csv`, `amp_activity.csv`) every corpus is paired with |
| `data/xh/` | the shipped BRICS corpus and its 226-token vocabulary: `corpus/modified_amp_safe.csv`, `vocab/safe_vocab.csv` |
| `data/amide_brics/` | amide_brics, DFS labels (206 tokens) |
| `data/amide_brics_raw/` | amide_brics, the encoder's own label numbering (206 tokens) |
| `data/brics_raw/` | plain BRICS, the encoder's own label numbering (179 tokens) |

Each build folder holds `corpus/`, `vocab/` and `report.json`. They are made from the SMILES in `data/xh/corpus/` by
`dataset_build/build.py`:

```bash
python -m dataset_build.build                              # amide_brics, dfs   -> data/amide_brics/        (about 20 s on 19 cores)
python -m dataset_build.build --order raw                  # amide_brics, raw   -> data/amide_brics_raw/
python -m dataset_build.build --rule brics --order raw     # plain BRICS, raw   -> data/brics_raw/
```

### Label order: dfs and raw

| `--order` | Labels |
|---|---|
| `dfs` (default) | renumbered 1, 2, 3, ... in DFS order over the fragment graph, `%100` written `%(100)`: the shipped corpus's convention |
| `raw` | the encoder's own numbering, scattered, `%100` left ambiguous |

| Pair | Corpus and vocabulary |
|---|---|
| `--tokenizer xh --order dfs` | `data/xh/` |
| `--tokenizer xh --order raw` | `data/brics_raw/` |
| `--tokenizer amide --order dfs` | `data/amide_brics/` |
| `--tokenizer amide --order raw` | `data/amide_brics_raw/` |


## Train and sample

```bash
python -m training.train --smoke --tokenizer amide --attn_backend sdpa       # 2 epochs, 4 batches: proves the wiring
python -m training.train --tokenizer amide --epochs 100 --batch_size 64
python -m training.train --tokenizer xh --epochs 100 --batch_size 64         # the BRICS comparison
python -m training.train --tokenizer amide --order raw --schedule single
python -m training.sample --checkpoint_path output/<run>/model-final.ckpt --num_samples 256
python evaluation/run.py results/samples.csv
```

`--schedule two` (default) masks topology tokens on a cosine over t in [0, 0.5] and chemistry tokens on [0.3, 1.0].
`--schedule single` gives the model no token classes, so every token is masked on 1 - t. A run directory holds
`model_config.json` (it records the tokenizer and order, which `sample.py` reads back), `metrics.csv`,
`model-epoch_N.ckpt` every `--save_every` epochs and `model-final.ckpt`. `sample.py` writes a `smiles,safe` CSV.
`--seed` (default 0) seeds torch, numpy and the data order and is recorded in `model_config.json`, so a run can be repeated
(GPU kernels may still differ in the last digits).
