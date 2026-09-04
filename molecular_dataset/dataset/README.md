# dataset — DBAASP download & AMP dataset building

Builds the antimicrobial-peptide (AMP) dataset from [DBAASP](https://dbaasp.org)
in two stages:

1. **[`run_dbaasp.py`](run_dbaasp.py)** — crawl the DBAASP API into a raw
   per-peptide table.
2. **[`build_amp_dataset.py`](build_amp_dataset.py)** — restructure that into
   linked peptide / activity tables.

The SAFE encoding + tokenizer stages live in [`../AMP_SAFE`](../AMP_SAFE); see the
[top-level README](../README.md) for the full pipeline.

## Files

| File | Role |
|------|------|
| [`run_dbaasp.py`](run_dbaasp.py) | Stage 1 entry point (CLI). Creates output dirs, runs the download, reports not-saved peptides. |
| [`dbaasp_download.py`](dbaasp_download.py) | DBAASP REST client: the `DBAASP` class (ID discovery + detail download over multiprocessing), `convert_units_filter` (activity → µg/ml), `choose_smiles` (canonical SMILES selection). |
| [`build_amp_dataset.py`](build_amp_dataset.py) | Stage 2: normalize species/strains, average MIC, canonicalize target categories, split into tables. |
| [`data_utils.py`](data_utils.py) | Molecular weight + µM ↔ µg/ml conversion helpers. |
| `data/` | All generated data (see Outputs). |

## Usage

Run from the **project root** (each script `chdir`s there; paths are
project-root-relative):

```bash
# Stage 1 — download DBAASP (long crawl over the whole ID range)
uv run python dataset/run_dbaasp.py
#   reuse an existing ID list to skip ID discovery:
uv run python dataset/run_dbaasp.py --id-path dataset/data/dbaasp/dbaasp_id.txt

# Stage 2 — build the linked peptide / activity tables
uv run python dataset/build_amp_dataset.py
```

## Outputs (under `dataset/data/`)

- **`dbaasp/dbaasp_id.txt`** — DBAASP monomer IDs found during discovery.
- **`dbaasp/dbaasp_new_info.csv`** — raw per-peptide details: name, sequence,
  smiles, nTerminus, cTerminus, targetGroups, targetObjects, per-species
  `targetActivities`, `toxicities`, `unusuals`.
- **`dbaasp/amp.csv`** — one row per peptide:
  `amp_id, name, sequence, smiles, nTerminus, cTerminus, targetGroups,
  targetObjects, unusuals`.
- **`dbaasp/amp_activity.csv`** — one row per (peptide, species):
  `amp_id, species, mic_ug_per_ml, n_measurements`.

Not-saved peptides are logged to `log/dbaasp/dbaasp.log`
(`NOT SAVED <id> <sequence> <reason>`).

## Key behavior & configuration

- **SMILES source.** Uses DBAASP's canonical structure; peptides without a
  DBAASP SMILES are skipped and logged.
- **Identity key.** Deduplication is on canonical SMILES, not sequence — DBAASP
  writes D-amino acids as lowercase and keeps terminal modifications out of the
  sequence string, so distinct molecules can share a FASTA. Join on `amp_id`.
- **MIC units.** `convert_units_filter` converts µM / nM / mM / etc. to µg/ml.
- **Stage-2 config** (top of [`build_amp_dataset.py`](build_amp_dataset.py)):
  `MIC_MEASURES` (default `{"MIC"}`), the six `TARGET_SPECIES`, and the
  `KEEP_GROUPS` / `KEEP_OBJECTS` canonical category sets. Species are normalized
  to `Genus species` (strain dropped) and MIC is averaged across strains and
  replicates.
