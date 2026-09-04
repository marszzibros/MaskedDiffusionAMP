import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import re

symbol2weight: Dict[str, float] = {
    "A": 89.094,
    "C": 121.154,
    "D": 133.104,
    "E": 147.131,
    "F": 165.192,
    "G": 75.067,
    "H": 155.156,
    "I": 131.175,
    "K": 146.189,
    "L": 131.175,
    "M": 149.208,
    "N": 132.119,
    "O": 255.313,
    "P": 115.132,
    "Q": 146.146,
    "R": 174.203,
    "S": 105.093,
    "T": 119.119,
    "U": 168.064,
    "V": 117.148,
    "W": 204.228,
    "Y": 181.191,
}

wildcard2members: Dict[str, Tuple[str]] = {
    "B": ("D", "N"),
    "J": ("I", "L"),
    "X": tuple(symbol2weight.keys()),
    "Z": ("E", "Q"),
}

non_wildcard_symbols = list(symbol2weight.keys())
wildcard_symbols = list(wildcard2members.keys())

for wildcard, members in wildcard2members.items():
    symbol2weight[wildcard] = float(np.mean([symbol2weight[x] for x in members]))

def camel_to_snake_case(in_str: str) -> str:
    in_str = in_str.replace("_", "").replace("/", "")
    return re.sub("(?!^)([A-Z]+)", r"_\1", in_str).lower()

def molecular_weight(seq: str) -> float:
    return sum(symbol2weight[symbol] for symbol in seq)


def uM_to_ug_per_ml(conc: float, seq: str) -> float:
    """
    Converts between micro-Moles per Liter (micro-Molar concentration) and
    micrograms per milliliter using the estimated molecular weight of an amino
    acid sequence.
    Estimated molecular weight is given in Daltons, or grams per mole.

    Dimensional Arithmetic:
        micro-moles per liter
            = 10**-6 mole / liter

        micro-grams per milliliter
            = 10**-6 gram / 10**-3 liter

        (micro-mole / liter) * (gram / mole) * 10**-3
            = (10**-6 mole / liter) * (gram / mole) * 10**-3
            = (10**-6 gram / liter) * 10**-3
            = (10**-6 gram / 10**3 * 1 liter)
            = micro-grams / milliliter

    Args:
        conc: float, Micro-molar concentration.
        seq: str, Amino acid sequence in FASTA format.

    Returns: float, concentration in micrograms per milliliter.
    """
    return conc * molecular_weight(seq) * 10 ** -3