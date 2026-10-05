"""
Strict peptide rate: the share of all samples that are one connected molecule, carry a run of at least `min_residues`
peptide-bonded residues, and contain no N-N or C(=O)-C(=O) bond.

The amide cut leaves every attachment point a carbonyl carbon or an amide nitrogen, so a label paired with the wrong
partner is still chemically legal and decodes. It shows up as one of those two bonds, which the training corpus almost
never has (about 0.1% of molecules have N-N, none have C(=O)-C(=O)).
"""
from rdkit import Chem

from backbone import backbone_length
from connectivity import is_connected

NON_PEPTIDE_LINKS = (
    Chem.MolFromSmarts("[NX3;!a]-[NX3;!a]"),     # N-N
    Chem.MolFromSmarts("[CX3](=O)-[CX3]=O"),     # C(=O)-C(=O)
)


def has_non_peptide_link(mol):
    return any(mol.HasSubstructMatch(pattern) for pattern in NON_PEPTIDE_LINKS)


def is_strict(mol, min_residues=5):
    return is_connected(mol) and backbone_length(mol) >= min_residues and not has_non_peptide_link(mol)


def strict(samples, min_residues=5):
    if not samples:
        return float("nan")
    return sum(s.valid and is_strict(s.mol, min_residues) for s in samples) / len(samples)
