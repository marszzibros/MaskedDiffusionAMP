"""The amide_brics slicer: one cut per non-ring amide C-N bond, plain BRICS where the molecule has none.

BRICS cuts both sides of every backbone nitrogen, which leaves the nitrogen as a one-atom fragment and
doubles the number of attachment labels. This rule cuts only the carbonyl-carbon to nitrogen bond:

    [C;$(C=O)]!@[#7]      a carbonyl carbon, joined by a non-ring bond, to a nitrogen

Ring-shaped peptides have no such bond; they are cut with BRICS, as in the shipped corpus.

Two label numberings, chosen with `order`:

    dfs   labels renumbered 1, 2, 3, ... in DFS order, %100 written as %(100): how the shipped corpus
          (data/xh/corpus/modified_amp_safe.csv) was finished
    raw   the encoder's own numbering, untouched: how the corpus was first written (labels scattered, %100 ambiguous)

`rule` picks the cuts: amide_brics (above) or plain brics. Only a string that decodes back to its source molecule
is returned.
"""
import sys
from dataclasses import dataclass
from typing import Optional

from rdkit import Chem, RDLogger

from .safe_utils import reorder_and_reindex_safe, safe_to_smiles

AMIDE_SMARTS = "[C;$(C=O)]!@[#7]"

_converters = {}


@dataclass(frozen=True)
class Encoded:
    safe: Optional[str]         # None when the molecule could not be cut and verified
    error: Optional[str]        # why, when safe is None
    used_brics: bool            # True when the amide rule found nothing and BRICS was used


def quiet():
    """Silence RDKit and the safe library's per-molecule warnings. Call once per process."""
    RDLogger.DisableLog("rdApp.*")
    from loguru import logger
    logger.remove()
    # reorder_and_reindex_safe is a recursive walk over the fragment graph
    sys.setrecursionlimit(20000)


def _get(name):
    if not _converters:
        from safe import SAFEConverter
        # ignore_stereo stays False: every residue is a stereocentre
        _converters["amide"] = SAFEConverter(slicer=[AMIDE_SMARTS], ignore_stereo=False)
        _converters["brics"] = SAFEConverter(slicer="brics", ignore_stereo=False)
    return _converters[name]


ORDERS = ("dfs", "raw")
RULES = ("amide_brics", "brics")


def encode(smiles, order="dfs", rule="amide_brics"):
    """SMILES -> Encoded. The SAFE string has been cut, numbered as `order` says and checked against the source molecule."""
    if order not in ORDERS or rule not in RULES:
        raise ValueError(f"order must be one of {ORDERS} and rule one of {RULES}")
    source = Chem.MolFromSmiles(smiles)
    if source is None:
        return Encoded(None, "bad_smiles", False)
    used_brics = rule == "brics"
    try:
        raw = _get("brics" if used_brics else "amide").encoder(smiles, allow_empty=False)
    except Exception as exc:
        if used_brics:
            return Encoded(None, f"not_encoded_{type(exc).__name__}", True)
        used_brics = True
        try:
            raw = _get("brics").encoder(smiles, allow_empty=False)
        except Exception as exc:
            return Encoded(None, f"not_encoded_{type(exc).__name__}", True)
    try:
        safe = reorder_and_reindex_safe(raw) if order == "dfs" else raw
    except Exception as exc:
        return Encoded(None, f"relabel_{type(exc).__name__}", used_brics)
    if safe_to_smiles(safe) != Chem.MolToSmiles(source):
        return Encoded(None, "roundtrip_mismatch", used_brics)
    return Encoded(safe, None, used_brics)
