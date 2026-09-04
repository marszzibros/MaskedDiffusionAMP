# AMP_SAFE — SAFE encoding & tokenizer

Converts the peptide SMILES built by [`../dataset`](../dataset) into
[SAFE](https://github.com/datamol-io/safe) strings and trains a SAFE tokenizer.
Two stages:

3. **[`smiles_to_safe.py`](smiles_to_safe.py)** — `amp.csv` → `amp_safe.csv`
   (SMILES → SAFE).
4. **[`train_safe_tokenizer.py`](train_safe_tokenizer.py)** — `amp_safe.csv` →
   `tokenizer.json`.

Consumes `dataset/data/dbaasp/amp.csv` produced by the dataset stage; see the
[top-level README](../README.md) for the full pipeline.

## Files

| File | Role |
|------|------|
| [`smiles_to_safe.py`](smiles_to_safe.py) | Stage 3: encode each SMILES to a SAFE string with a reused `SAFEConverter` (BRICS fragmentation). Failures are logged, not fatal. |
| [`train_safe_tokenizer.py`](train_safe_tokenizer.py) | Stage 4: train `safe.SAFETokenizer` (BPE with SAFE's SMILES-aware splitter) and save a reloadable `tokenizer.json`. |

## Usage

Run from the **project root** (each script `chdir`s there; paths are
project-root-relative):

```bash
# Stage 3 — SMILES -> SAFE  (~2-3 min over ~21k molecules)
uv run python AMP_SAFE/smiles_to_safe.py

# Stage 4 — train the SAFE tokenizer
uv run python AMP_SAFE/train_safe_tokenizer.py --vocab-size 1000
```

## Outputs

- **`dataset/data/safe/amp_safe.csv`** — `amp_id, smiles, safe`.
- **`dataset/data/safe/tokenizer.json`** — trained tokenizer.
- **`log/safe/encode_failures.log`** — SMILES that could not be SAFE-encoded.

## Load the tokenizer

```python
from safe import SAFETokenizer
tok = SAFETokenizer.load("dataset/data/safe/tokenizer.json")
ids = tok.encode(safe_string)      # SAFE string -> token ids
```

## Notes

- **transformers < 5 required.** `safe-mol` 0.1.x breaks on transformers 5.x;
  the version is pinned in [`../pyproject.toml`](../pyproject.toml) so
  `import safe` works directly.
- **Stereochemistry is preserved** through SAFE (verified round-trip), so the
  D/L amino-acid distinction survives into the SAFE strings and tokenizer.
- **`allow_empty=True`** keeps molecules that cannot be fragmented (small or
  undividable ones) whole — encoded as a single fragment — instead of dropping
  them, so encoding does not fail on those.
- **Vocab size is an upper bound.** The chemical alphabet is small, so BPE
  saturates well below `--vocab-size` (≈159 tokens on this dataset). Lower
  `--min-frequency` for more merges.
