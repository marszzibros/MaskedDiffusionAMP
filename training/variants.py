"""The corpus + vocabulary pairs: xh (BRICS, the shipped one) and amide (amide_brics), each with DFS or raw labels.

Each vocabulary was fitted to its own corpus and its `type` column drives the two-schedule masking, so a
corpus is never used with another's vocabulary. All of it lives in data/:

    --tokenizer xh    --order dfs    data/xh/             the shipped BRICS corpus and its 226-token vocabulary
    --tokenizer xh    --order raw    data/brics_raw/      plain BRICS, the encoder's own label numbering
    --tokenizer amide --order dfs    data/amide_brics/
    --tokenizer amide --order raw    data/amide_brics_raw/

data/dbaasp/ holds the peptide and activity tables every variant is paired with. The build folders are made by
`python -m dataset_build.build` (see the README).
"""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VARIANTS = ("xh", "amide")
ORDERS = ("dfs", "raw")

_FOLDER = {("xh", "dfs"): "xh", ("xh", "raw"): "brics_raw",
           ("amide", "dfs"): "amide_brics", ("amide", "raw"): "amide_brics_raw"}
_BUILD = {"brics_raw": "--rule brics --order raw", "amide_brics": "", "amide_brics_raw": "--order raw"}


@dataclass(frozen=True)
class Variant:
    name: str
    order: str
    vocab_path: str
    corpus_csv: str     # also the pool that sampling draws generation lengths from


def add_args(ap):
    ap.add_argument("--tokenizer", choices=VARIANTS, default="xh",
                    help="xh = shipped BRICS corpus and vocabulary; amide = the amide_brics pair")
    ap.add_argument("--order", choices=ORDERS, default="dfs",
                    help="dfs = attachment labels renumbered in DFS order (the shipped convention); "
                         "raw = the encoder's own numbering. Each has its own refit vocabulary.")


def resolve(name, order="dfs"):
    if (name, order) not in _FOLDER:
        raise ValueError(f"unknown tokenizer {name!r} or order {order!r}; choose from {VARIANTS} and {ORDERS}")
    folder = _FOLDER[(name, order)]
    vocab = DATA / folder / "vocab" / "safe_vocab.csv"
    corpus = DATA / folder / "corpus" / "modified_amp_safe.csv"
    missing = [str(p) for p in (vocab, corpus) if not p.exists()]
    if missing:
        raise SystemExit(f"--tokenizer {name} --order {order} needs {missing}; "
                         f"build it with: python -m dataset_build.build {_BUILD.get(folder, '')}".rstrip())
    return Variant(name, order, str(vocab), str(corpus))


def activate(variant):
    """Point amp_diffusion at this vocabulary. Call before the datamodule's setup(), which reads the path."""
    from amp_diffusion import data
    data.TOKENIZER_PATH = variant.vocab_path


def data_dir(variant, run_dir):
    """The `file_path` for AMPSafeDataModule.

    The dataset reads <dir>/safe/modified_amp_safe.csv and <dir>/dbaasp/, so the chosen corpus is shown to it
    through a directory of symlinks inside the run directory.
    """
    root = Path(run_dir).resolve() / "data"
    (root / "safe").mkdir(parents=True, exist_ok=True)
    for target, name in ((DATA / "dbaasp", root / "dbaasp"),
                         (variant.corpus_csv, root / "safe" / "modified_amp_safe.csv")):
        if os.path.lexists(name):
            os.unlink(name)
        os.symlink(os.path.abspath(target), name)
    return str(root) + os.sep
