import re
from pathlib import Path

import pandas as pd
import safe as sf
from rdkit import Chem

def parse_safe_globally(safe_str):
    pattern = re.compile(r'(\[[^\]]+\])|(%\(\d+\))|(%\d{3})|(%\d{2})|([0-9])|([^\[\]%0-9]+)')
    
    # First pass: find all anchors globally to count frequencies
    counts = {}
    for match in pattern.finditer(safe_str):
        if match.group(2): counts[match.group(2)] = counts.get(match.group(2), 0) + 1
        elif match.group(3): counts[match.group(3)] = counts.get(match.group(3), 0) + 1
        elif match.group(4): counts[match.group(4)] = counts.get(match.group(4), 0) + 1
        elif match.group(5): counts[match.group(5)] = counts.get(match.group(5), 0) + 1

    # Now parse fragment by fragment
    frags = safe_str.split('.')
    parsed_frags = []
    
    for frag in frags:
        parts = []
        for match in pattern.finditer(frag):
            if match.group(1):
                parts.append(('text', match.group(1)))
            elif match.group(2):
                parts.append(('anchor', match.group(2)))
            elif match.group(3):
                v = match.group(3)
                if counts[v] % 2 != 0:
                    parts.append(('anchor', v[:3]))
                    parts.append(('anchor', v[3]))
                else:
                    parts.append(('anchor', v))
            elif match.group(4):
                parts.append(('anchor', match.group(4)))
            elif match.group(5):
                parts.append(('anchor', match.group(5)))
            elif match.group(6):
                parts.append(('text', match.group(6)))
        parsed_frags.append(parts)
        
    return parsed_frags

def fix_safe_for_rdkit(safe_str):
    parsed_frags = parse_safe_globally(safe_str)
    new_frags = []
    for frag in parsed_frags:
        new_frag = ""
        for t, v in frag:
            if t == 'anchor' and v.startswith('%') and not v.startswith('%(') and len(v) > 3:
                new_frag += f"%({v[1:]})"
            else:
                new_frag += v
        new_frags.append(new_frag)
    return '.'.join(new_frags)

# --- copied from reorganize.py ---
def safe_to_smiles(safe_str):
    """Decode a SAFE string to canonical SMILES, or None if it is not valid."""
    if not safe_str:
        return None
    try:
        # fixed_safe_str = fix_safe_for_rdkit(safe_str)
        # print(f"Fixed SAFE string for RDKit: {fixed_safe_str}")
        fixed_safe_str = safe_str  # Use the original SAFE string without fixing
        smiles = sf.decode(fixed_safe_str, canonical=True, fix=True, ignore_errors=False)
    except Exception as e:
        print(f"Error occurred while decoding SAFE string: {e}")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"RDKit failed to parse SMILES: {smiles}")
        return None
    else:
        return Chem.MolToSmiles(mol)


if __name__ == "__main__":
    # Example usage
    safe_str = "[C@@H]12NC(=O)[C@H]3NC(=O)[C@H]4NC1=O.c15ccccc1.c16ccccc1.C76.C65.NC(N)=NCCC2.C3CCN=C(C)C.C4CCN=C(N)N"
    smiles = safe_to_smiles(safe_str)
    print(f"SMILES: {smiles}")