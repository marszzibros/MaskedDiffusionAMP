"""SAFE string helpers: the label fix for RDKit, the DFS relabelling, and the decode check.

The functions the shipped corpus was made with, so that this folder builds a corpus by the same rules.
"""
import re

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
        fixed_safe_str = fix_safe_for_rdkit(safe_str)
        smiles = sf.decode(fixed_safe_str, canonical=True, ignore_errors=False)
    except Exception:
        return None
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def reorder_and_reindex_safe(safe_str):
    parsed_frags = parse_safe_globally(safe_str)

    anchor_to_frags = {}
    for idx, parts in enumerate(parsed_frags):
        for ptype, pval in parts:
            if ptype == 'anchor':
                anchor_to_frags.setdefault(pval, []).append(idx)

    visited_frags = set()
    new_frags = []
    anchor_map = {}
    next_anchor_id = 1

    def format_anchor(num):
        if num < 10:
            return str(num)
        elif num < 100:
            return f"%{num}"
        else:
            return f"%({num})"

    def dfs(frag_idx):
        nonlocal next_anchor_id
        visited_frags.add(frag_idx)

        parts = parsed_frags[frag_idx]
        new_frag_str = ""
        outgoing_anchors = []

        for ptype, pval in parts:
            if ptype == 'anchor':
                if pval not in anchor_map:
                    anchor_map[pval] = format_anchor(next_anchor_id)
                    next_anchor_id += 1
                new_frag_str += anchor_map[pval]
                outgoing_anchors.append(pval)
            else:
                new_frag_str += pval

        new_frags.append(new_frag_str)

        for old_anchor in outgoing_anchors:
            for neighbor_idx in anchor_to_frags.get(old_anchor, []):
                if neighbor_idx not in visited_frags:
                    dfs(neighbor_idx)

    # Some fragments might be disconnected components, iterate to ensure all are visited
    for i in range(len(parsed_frags)):
        if i not in visited_frags:
            dfs(i)

    return '.'.join(new_frags)

