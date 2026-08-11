# MaskedDiffusionAMP

Conditional **discrete flow matching** over antimicrobial peptide (AMP) sequences.
A DiT backbone is trained to denoise masked sequences, conditioned on target
species, target groups, and MIC bin, with per-condition classifier-free guidance
at sampling time.

| File | Role |
|------|------|
| [`DFM.py`](DFM.py) | `DiscreteFlowMatching` — the Lightning module: masking corruption, masked-position loss, and the guided sampler. |
| [`models/DiTwithCondition.py`](models/DiTwithCondition.py) | `DIT` — rotary DiT with adaLN-zero conditioning; one `VectorEmbedder` per condition, each carrying a learned null embedding for CFG. |
| [`models/ema.py`](models/ema.py) | Exponential moving average of the model weights. |
| [`trainer.py`](trainer.py) | Training entry point (config, datamodule, W&B logger, checkpoint callback). |
| [`sample.py`](sample.py) | Generation from a trained checkpoint. |
| [`molecular_dataset/`](molecular_dataset/) | Separate repo (cloned, not a submodule): builds the AMP dataset from DBAASP and trains a SAFE tokenizer. |

---

## Environment

One uv environment covers both the dataset pipeline and training. Requires
**Python ≥ 3.12** and [uv](https://docs.astral.sh/uv/).

```bash
# from the repo root
uv sync
```

That creates `.venv/` and installs everything in [`pyproject.toml`](pyproject.toml).
Run things with `uv run python trainer.py`, or activate with
`source .venv/bin/activate`.

### flash-attn

[`models/DiTwithCondition.py`](models/DiTwithCondition.py) imports `flash_attn`,
which compiles against an already-installed torch and therefore cannot be
resolved in the same pass as torch. Install it **after** `uv sync`, from a
prebuilt wheel matching this stack (cu126 / torch 2.13 / cp312):

```bash
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu126torch2.13-cp312-cp312-linux_x86_64.whl
```

Building from source instead (`uv pip install flash-attn --no-build-isolation`)
works but takes 10-40 minutes and needs `nvcc` on `PATH`.

Note that `uv sync` prunes anything not in the lockfile, so **re-run the wheel
install after any later `uv sync`**. If the Python, torch, or CUDA version
changes, the wheel URL must change to match — a mismatch fails at `import
flash_attn` with an `undefined symbol` error, not at install time.

### CUDA version

`pyproject.toml` pins the **cu126** PyTorch wheel index. 

### transformers is pinned below 5.x

`safe-mol` 0.1.x imports symbols that transformers 5.x removed, so the pin is
`transformers>=4.42,<5`. The training code only uses
`get_linear_schedule_with_warmup` / `get_cosine_schedule_with_warmup`, which are
stable across the 4.x line, so this costs nothing on the training side.

---

## Data

The dataset lives in the nested [`molecular_dataset/`](molecular_dataset/) repo,
which stores its CSVs in **Git LFS**. A plain `git clone` leaves ~130-byte
pointer stubs in place of the real files, so fetch them explicitly:

```bash
cd molecular_dataset
git lfs install
git lfs pull                                  # ~232 MB total
git lfs pull --include="dataset/data/**"      # ~56 MB, peptide pipeline only
```

The pipeline itself (DBAASP download → peptide/activity tables → SAFE encoding →
tokenizer) is documented in
[`molecular_dataset/README.md`](molecular_dataset/README.md). Its scripts run
under this same environment, so `uv run python molecular_dataset/dataset/build_amp_dataset.py`
works from the repo root — no second venv needed.

`molecular_dataset/` is a separate git repository and is git-ignored here. To
track it against a fixed revision instead, replace the clone with a submodule:

```bash
git submodule add git@github.com:marszzibros/molecular_dataset.git
```

---

## Training

```bash
uv run python trainer.py
```

Hyperparameters are the `model_config` dict at the top of
[`trainer.py`](trainer.py); it is written to `output/<timestamp>/model_config.json`
alongside the checkpoints. The values that must agree with the tokenizer are
`num_tokens`, `mask_token_id`, `pad_token_id`, and `max_length` — a mismatch here
fails silently rather than loudly, because the mask/pad ids are only used to
index into the logits.

### Data module contract

`DiscreteFlowMatching.training_step` expects each batch to provide:

| key | shape | notes |
|-----|-------|-------|
| `sequence` | `(B, L)` int64, or `(B, V, L)` one-hot | one-hot is `argmax`-ed over dim 1 |
| `condition` | `(B, 26)` float | slice layout below |

The condition vector layout is fixed by
[`DFM.py:85-88`](DFM.py#L85-L88) and must be reproduced by any new dataset:

```
[0:6]    species    one-hot   -> species_vec
[6:11]   groups     multi-hot -> groups_vec
[11:16]  objects              -> ignored by the model
[16:26]  MIC decile one-hot   -> mic_vec
```

### Sequence length

SAFE strings are long: median **296** tokens, p95 **681**, max **1374** — versus
68 for the old character-level peptides. `max_length=None` (the default) sets the
cap to the corpus maximum, so **no molecule is dropped and none is truncated**.

### Decoding

`AMPSafeDataModule.decode(ids)` gives the SAFE string; `decode_to_smiles(ids)`
gives `(safe_string, smiles_or_None)`. `smiles` is `None` when the generated
tokens do not form a decodable molecule — unbalanced ring closures, a fragment
cut off mid-attachment. That None-rate is the validity metric, logged each epoch
as `sample_validity` and written next to each sample in `generated_samples.txt`.

`SafeDecoder` in [`dataset.py`](dataset.py) is the same thing standalone, needing
only `tokenizer.json` — used by `sample.py` so generation does not load the
corpus.

## Sampling

```bash
uv run python sample.py \
    --checkpoint_path output/<timestamp>/model-epoch_800.ckpt \
    --num_samples 256
```

