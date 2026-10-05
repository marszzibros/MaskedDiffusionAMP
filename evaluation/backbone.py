"""
Peptide backbone rate: how many samples carry an intact run of amino-acid residues.
"""
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

    # Follow the links from each residue until the run ends or comes back to a residue it has already seen. A head-to-tail
    # cyclic peptide has no start and counts as the residues of its ring; a stray two-residue ring beside a chain adds nothing
    # to the chain (the old shortcut returned the molecule's whole residue count whenever any ring existed).
    longest = 0
    for start in residues:
        seen, cur = set(), start
        while cur is not None and cur not in seen:
            seen.add(cur)
            cur = nxt.get(cur)
        longest = max(longest, len(seen))
    return longest


def backbone(samples, min_residues=5):
    lengths = [backbone_length(s.mol) for s in samples if s.valid]
    if not lengths:
        return float("nan")
    return sum(n >= min_residues for n in lengths) / len(lengths)


def mean_backbone_length(samples):
    lengths = [backbone_length(s.mol) for s in samples if s.valid]
    return sum(lengths) / len(lengths) if lengths else float("nan")
