"""
Connected rate: the share of all samples that decode to one connected molecule.

A SAFE string whose attachment labels do not pair up decodes to several pieces. RDKit still accepts that, so validity
counts it. Every molecule in the training corpus is one piece.
"""
from rdkit import Chem


def is_connected(mol):
    return len(Chem.GetMolFrags(mol)) == 1


def connected(samples):
    if not samples:
        return float("nan")
    return sum(s.valid and is_connected(s.mol) for s in samples) / len(samples)
