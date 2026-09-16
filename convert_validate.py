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
        original_smiles = row['smiles']
        if pd.isna(original_safe):
            continue
            
        original_mol = Chem.MolFromSmiles(original_smiles)
        original_canonical = Chem.MolToSmiles(original_mol) if original_mol else None

        before = safe_to_smiles(original_safe)
        
        if before != original_canonical:
            filtered_rows.append(int(idx))
            mismatches.append({
                'row': int(idx),
                'error': 'safe_to_smiles != original_smiles',
                'original_canonical': original_canonical,
                'before_smiles': before,
                'original_safe': original_safe
            })
            continue

        reorganized = reorder_and_reindex_safe(original_safe)
        after = safe_to_smiles(reorganized)
        checked += 1

        if before != after:
            filtered_rows.append(int(idx))
            mismatches.append({
                'row': int(idx),
                'error': 'reorganized_safe_smiles != original_safe_smiles',
                'before_smiles': before,
                'after_smiles': after,
                'original_safe': original_safe,
                'reorganized_safe': reorganized,
            })
            continue

        df.at[idx, 'safe'] = reorganized

    cleaned_df = df.drop(index=filtered_rows)
    cleaned_df.to_csv(out_path, index=False)
    
    if mismatches:
        pd.DataFrame(mismatches).to_csv(out_path.with_name('mismatches.csv'), index=False)

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
