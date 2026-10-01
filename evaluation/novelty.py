"""Fraction of valid samples whose canonical SMILES is not in the training corpus.
"""
from rdkit import Chem


def novelty(samples, reference, ignore_stereo=False):
    smiles = [Chem.MolToSmiles(s.mol, isomericSmiles=not ignore_stereo) for s in samples if s.valid]
    if not smiles:
        return float("nan")
    return sum(c not in reference for c in smiles) / len(smiles)
