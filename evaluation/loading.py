"""Turning files into `Sample` records. Not a metric: the shared input every metric takes.

Each SMILES is parsed once here, so the metrics do not each re-parse the same molecule.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# the corpus the models train on; its `smiles` column is the same for the xh and amide corpora
CORPUS = str(Path(__file__).resolve().parents[1] / "data" / "xh" / "corpus" / "modified_amp_safe.csv")


@dataclass(frozen=True)
class Sample:
    """One generated molecule. smiles and mol are None when the SAFE string did not decode."""
    safe: str
    smiles: str | None
    mol: object | None

    @property
    def valid(self):
        return self.mol is not None


def parse(smiles):
    """The Mol for a SMILES string, or None. 'INVALID' (what sample.py writes) and junk both give None."""
    if not isinstance(smiles, str) or smiles == "INVALID":
        return None
    return Chem.MolFromSmiles(smiles)


def load_samples(path):
    """[Sample] from the CSV sample.py writes (`smiles,safe` header)."""
    df = pd.read_csv(path)
    missing = {"smiles", "safe"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing column(s) {sorted(missing)}; expected the `smiles,safe` CSV from sample.py")
    out = []
    for safe, smiles in zip(df["safe"], df["smiles"]):
        if not isinstance(safe, str):
            continue
        mol = parse(smiles)
        out.append(Sample(safe, Chem.MolToSmiles(mol) if mol is not None else None, mol))
    return out


def load_reference(path=CORPUS, ignore_stereo=False):
    """Set of canonical SMILES of the training corpus: what novelty is measured against."""
    out = set()
    for smiles in pd.read_csv(path)["smiles"].dropna():
        mol = parse(smiles)
        if mol is not None:
            out.add(Chem.MolToSmiles(mol, isomericSmiles=not ignore_stereo))
    return out
