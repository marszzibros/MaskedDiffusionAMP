# molecular_dataset

Build a machine-learning-ready **antimicrobial peptide (AMP)** dataset from the
[DBAASP](https://dbaasp.org) database, then convert every peptide to a
[SAFE](https://github.com/datamol-io/safe) string and train a SAFE tokenizer.

The pipeline downloads peptides from DBAASP, converts activity values to a common
unit, attaches the canonical SMILES from DBAASP (peptides without one are
skipped), splits the result into linked peptide / activity tables, and finally
produces SAFE strings and a trained tokenizer.

Peptides are keyed by **canonical SMILES**, not by sequence, so that D-amino
acids (encoded as lowercase letters) and terminal modifications — which two
different molecules can share as the same FASTA string — are kept distinct.


| Stage | Script | Reads | Writes |
|-------|--------|-------|--------|
| 1. Download | [`dataset/run_dbaasp.py`](dataset/run_dbaasp.py) → [`dataset/dbaasp_download.py`](dataset/dbaasp_download.py) | DBAASP API | `dataset/data/dbaasp/dbaasp_id.txt`, `dataset/data/dbaasp/dbaasp_new_info.csv` |
| 2. Restructure | [`dataset/build_amp_dataset.py`](dataset/build_amp_dataset.py) | `dbaasp_new_info.csv` | `dataset/data/dbaasp/amp.csv`, `dataset/data/dbaasp/amp_activity.csv` |
| 3. SAFE encode | [`AMP_SAFE/smiles_to_safe.py`](AMP_SAFE/smiles_to_safe.py) | `amp.csv` | `dataset/data/safe/amp_safe.csv` |
| 4. Train tokenizer | [`AMP_SAFE/train_safe_tokenizer.py`](AMP_SAFE/train_safe_tokenizer.py) | `amp_safe.csv` | `dataset/data/safe/tokenizer.json` |

Supporting module (in `dataset/`): [`data_utils.py`](dataset/data_utils.py)
(molecular weight and µM ↔ µg/ml conversion).

## Repository layout

```
dataset/                    DBAASP download + dataset building  (see dataset/README.md)
    run_dbaasp.py           stage 1 entry point
    dbaasp_download.py      DBAASP API client (imports data_utils)
    data_utils.py
    build_amp_dataset.py    stage 2
    data/                   all generated data lives here
        dbaasp/             dbaasp_new_info.csv, amp.csv, amp_activity.csv, ...
        safe/               amp_safe.csv, tokenizer.json
AMP_SAFE/                   SAFE encoding + tokenizer  (see AMP_SAFE/README.md)
    smiles_to_safe.py       stage 3
    train_safe_tokenizer.py stage 4
log/                        run logs (dbaasp/, safe/)
pyproject.toml              uv environment definition
```

Per-folder docs: [`dataset/README.md`](dataset/README.md) and
[`AMP_SAFE/README.md`](AMP_SAFE/README.md).

Each entry script `chdir`s to the project root on startup, so the default
(project-root-relative) paths resolve no matter which directory you launch from.

---

## Requirements

- **Python ≥ 3.12**
- **[uv](https://docs.astral.sh/uv/)** for environment management
- Core libraries (installed by uv from [`pyproject.toml`](pyproject.toml)):
  `rdkit`, `datamol`, `safe-mol`, `transformers`, `tokenizers`, `pandas`,
  `numpy`, `requests`.

### Set up the environment with uv

```bash
# 1. install uv (once, if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. create the virtual environment and install all dependencies
uv sync
```

`uv sync` reads `pyproject.toml`, creates a `.venv/`, and installs the pinned
dependencies. Run scripts either through uv or by activating the venv:

```bash
uv run python AMP_SAFE/smiles_to_safe.py        # via uv
# or
source .venv/bin/activate && python AMP_SAFE/smiles_to_safe.py
```

---

## Usage

Run the stages in order from the project root (each reads the previous stage's
output). Data lands under `dataset/data/` and logs under `log/`, created
automatically.

```bash
# 1. Download DBAASP  (long: crawls the API; skip if dbaasp_new_info.csv exists)
uv run python dataset/run_dbaasp.py
#    reuse an existing ID list to skip ID discovery:
uv run python dataset/run_dbaasp.py --id-path dataset/data/dbaasp/dbaasp_id.txt

# 2. Build the linked peptide / activity tables
uv run python dataset/build_amp_dataset.py

# 3. Convert SMILES to SAFE  (~2-3 min over ~21k molecules)
uv run python AMP_SAFE/smiles_to_safe.py

# 4. Train the SAFE tokenizer
uv run python AMP_SAFE/train_safe_tokenizer.py --vocab-size 1000
```

Load the trained tokenizer:

```python
from safe import SAFETokenizer
tok = SAFETokenizer.load("dataset/data/safe/tokenizer.json")
ids = tok.encode(safe_string)      # SAFE string -> token ids
```

---

## Outputs

All artifacts are written under `dataset/data/`; peptides join across tables on `amp_id`.

**`amp.csv`** — one row per peptide (structure + metadata)

| column | description |
|--------|-------------|
| `amp_id` | stable integer key |
| `name`, `sequence` | peptide name and FASTA (case preserved: lowercase = D-amino acid) |
| `smiles` | canonical SMILES (from DBAASP) |
| `nTerminus`, `cTerminus` | terminal modifications (e.g. `AMD`, `C16`) |
| `targetGroups` | canonical set ⊆ {`GRAM-`, `GRAM+`, `MAMMALIAN CELL`, `FUNGUS`, `OTHER`} |
| `targetObjects` | canonical set ⊆ {`LIPID BILAYER`, `DNA / RNA`, `CYTOPLASMIC PROTEIN`, `MEMBRANE PROTEIN`, `OTHER`} |
| `unusuals` | unusual amino acids present |

**`amp_activity.csv`** — one row per (peptide, species)

| column | description |
|--------|-------------|
| `amp_id` | join key into `amp.csv` |
| `species` | strain-normalized species (`Genus species`, lowercased) |
| `mic_ug_per_ml` | MIC averaged over all strains/replicates, in µg/ml |
| `n_measurements` | number of raw measurements averaged |

Restricted to MIC against six species (configurable at the top of
`dataset/build_amp_dataset.py`): *E. coli, P. aeruginosa, K. pneumoniae, S. aureus,
B. subtilis, S. epidermidis*.

**`amp_safe.csv`** — `amp_id`, `smiles`, `safe` (SAFE encoding of each SMILES).

**`dataset/data/safe/tokenizer.json`** — trained SAFE tokenizer.

Failures are logged (not fatal): `log/dbaasp/dbaasp.log` (peptides not saved
during download) and `log/safe/encode_failures.log` (SMILES that could not be
SAFE-encoded).

---

## Notes & caveats

- **SMILES is the identity key.** DBAASP encodes D-amino acids as lowercase
  letters and keeps terminal modifications out of the sequence string, so two
  chemically different peptides can share a FASTA sequence. Deduplication is done
  on canonical SMILES; expect `sequence` to repeat across rows with distinct
  `smiles` (e.g. lipidated variants). Join on `amp_id`, not `sequence`.

- **SAFE encoding preserves stereochemistry** (verified round-trip), so the D/L
  distinction survives into the SAFE strings and tokenizer. Molecules that cannot
  be fragmented are kept whole (`allow_empty=True`) rather than dropped.

- **MIC units.** Activity values are converted to **µg/ml** during download and
  averaged as-is (arithmetic mean). Peptides containing non-standard residues
  (an `X` / unusual amino acid) use an approximate molecular weight, so their
  unit conversions are less reliable.

- **SAFE + transformers.** `safe-mol` (0.1.x) was written for transformers 4.x
  and imports symbols that transformers 5.x removed, but it declares no upper
  bound. `pyproject.toml` therefore pins `transformers>=4.42,<5` so `import safe`
  works directly. If your environment still has transformers 5.x installed, apply
  the pin with:

  ```bash
  uv pip install 'transformers<5'
  ```

- **Generated artifacts.** Everything under `dataset/data/` and `log/` is
  reproducible from the scripts; consider git-ignoring them (`amp_safe.csv` is
  ~18 MB).
