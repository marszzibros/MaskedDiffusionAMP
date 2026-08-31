import re
from pathlib import Path

import pandas as pd
import safe as sf
from rdkit import Chem


# --- copied from reorganize.py ---
def safe_to_smiles(safe_str):
    """Decode a SAFE string to canonical SMILES, or None if it is not valid."""
    if not safe_str:
        return None
    try:
        smiles = sf.decode(safe_str, canonical=True, ignore_errors=False)
    except Exception:
        return None
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def parse_fragment(frag_str):
    pattern = re.compile(r'(\[[^\]]+\])|(%\d{2})|([0-9])|([^\[\]%0-9]+)')
    parts = []
    for match in pattern.finditer(frag_str):
        if match.group(1):
            parts.append(('text', match.group(1)))
        elif match.group(2):
            parts.append(('anchor', match.group(2)))
        elif match.group(3):
            parts.append(('anchor', match.group(3)))
        elif match.group(4):
            parts.append(('text', match.group(4)))
    return parts


def reorder_and_reindex_safe(safe_str):
    frags = safe_str.split('.')
    parsed_frags = [parse_fragment(f) for f in frags]

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
        return str(num) if num < 10 else f"%{num}"

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

    for i in range(len(frags)):
        if i not in visited_frags:
            dfs(i)

    return '.'.join(new_frags)


# --- conversion + validation ---
def convert_and_validate(input_csv: str | Path, output_csv: str | Path | None = None):
    in_path = Path(input_csv)
    out_path = Path(output_csv) if output_csv is not None else in_path.with_name('modified_amp_safe.csv')

    df = pd.read_csv(in_path)
    filtered_rows = []
    mismatches = []
    checked = 0

    for idx, row in df.iterrows():
        original_safe = row['safe']
        if pd.isna(original_safe):
            continue

        before = safe_to_smiles(original_safe)
        reorganized = reorder_and_reindex_safe(original_safe)
        after = safe_to_smiles(reorganized)
        checked += 1

        if before != after:
            filtered_rows.append(int(idx))
            mismatches.append({
                'row': int(idx),
                'before_smiles': before,
                'after_smiles': after,
                'original_safe': original_safe,
                'reorganized_safe': reorganized,
            })
            continue

        df.at[idx, 'safe'] = reorganized

    cleaned_df = df.drop(index=filtered_rows)
    cleaned_df.to_csv(out_path, index=False)

    print(f'checked_rows={checked}')
    print(f'dropped_rows={len(filtered_rows)}')
    print(f'mismatch_count={len(mismatches)}')
    if mismatches:
        print('first_mismatch=', mismatches[0])
    else:
        print('all_smiles_equivalent_after_reorganization=True')

    print(f'output_csv={out_path}')
    return mismatches


if __name__ == '__main__':
    convert_and_validate('molecular_dataset/dataset/data/safe/amp_safe.csv')
