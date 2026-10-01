"""
Peptide backbone rate: how many samples carry an intact run of amino-acid residues.
"""
from functools import lru_cache

from rdkit import Chem

# N-C(alpha)-C(=O): an sp3 carbon between a three-connected nitrogen and a carbonyl carbon (proline's ring N matches too)
RESIDUE = Chem.MolFromSmarts("[NX3:1][CX4:2][CX3:3]=O")


def backbone_length(mol):
    """Residues in the longest peptide-bond-linked run of `mol` (0 if it has none)."""
    residues = {(n, c) for n, _, c, _ in mol.GetSubstructMatches(RESIDUE)}
    by_n = {n: (n, c) for n, c in residues}
    # the residue after (n, c) is the one whose nitrogen is bonded to this carbonyl carbon
    nxt = {}
    for n, c in residues:
        for nb in mol.GetAtomWithIdx(c).GetNeighbors():
            if nb.GetIdx() in by_n and nb.GetIdx() != n:
                nxt[(n, c)] = by_n[nb.GetIdx()]

    @lru_cache(maxsize=None)
    def run(res):
        # a run cannot loop: a ring of residues would make the recursion infinite
        return 1 + (run(nxt[res]) if res in nxt and nxt[res] != res else 0)

    try:
        return max((run(r) for r in residues), default=0)
    except RecursionError:      # a head-to-tail cyclic peptide has no start; count its residues
        return len(residues)


def backbone(samples, min_residues=5):
    lengths = [backbone_length(s.mol) for s in samples if s.valid]
    if not lengths:
        return float("nan")
    return sum(n >= min_residues for n in lengths) / len(lengths)


def mean_backbone_length(samples):
    lengths = [backbone_length(s.mol) for s in samples if s.valid]
    return sum(lengths) / len(lengths) if lengths else float("nan")
