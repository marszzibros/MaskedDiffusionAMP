"""
Reorder a SAFE corpus so bonded fragments sit next to each other.

Stage 3.5 of the pipeline: amp_safe.csv -> amp_safe_reorder.csv, between
smiles_to_safe.py and train_safe_tokenizer.py.

SAFE emits fragments in an essentially arbitrary order, so the two halves of a
ring-closure pair often land hundreds of tokens apart. A masked diffusion model
fills positions in parallel with no stack, so every open pair is a constraint it
has to satisfy across that gap. Walking the fragment graph depth-first and
re-emitting in traversal order puts each fragment beside the one it bonds to,
which collapses the typical gap without changing the molecule, the token count
or the vocabulary:

    brics, splitter=safe     median pair span 144 -> 9      (16x)
    token count                              324 -> 324     (unchanged)
    max ring label                            37 -> 37      (unchanged)

Note what it does NOT fix: anchors are renumbered strictly increasing from 1 and
never recycled, so the highest label still equals the total pair count. Corpora
that overflow the SMILES %99 ceiling (hr, mmpa, rotatable) stay broken.

The DFS reindexing is Xiaohan Zhang's, from reorganize.py on safe_xiaohan.

Usage:
    python scripts/reorder_safe.py --in-path  tokenizers/safe_brics.csv \
                                   --out-path tokenizers/safe_brics_reorder.csv
"""

import argparse
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from rdkit import Chem, RDLogger

import safe as sf

RDLogger.DisableLog("rdApp.*")

# Fragments nest a few levels at most, but a 300-fragment peptide chains the DFS
# that deep, and the default limit of 1000 frames is not generous enough.
sys.setrecursionlimit(50000)

# [bracket atom] | %nn anchor | single-digit anchor | anything else.
# Bracket atoms are matched first so digits inside [NH3+] are never read as
# ring-closure labels.
_TOKEN_RE = re.compile(r"(\[[^\]]+\])|(%\d{2})|([0-9])|([^\[\]%0-9]+)")


def parse_fragment(frag_str):
    """Split one fragment into ('anchor'|'text', value) parts."""
    parts = []
    for m in _TOKEN_RE.finditer(frag_str):
        if m.group(1):
            parts.append(("text", m.group(1)))
        elif m.group(2):
            parts.append(("anchor", m.group(2)))
        elif m.group(3):
            parts.append(("anchor", m.group(3)))
        elif m.group(4):
            parts.append(("text", m.group(4)))
    return parts


def reorder_and_reindex_safe(safe_str):
    """Re-emit fragments in DFS order over the bond graph, renumbering anchors."""
    frags = safe_str.split(".")
    parsed = [parse_fragment(f) for f in frags]

    anchor_to_frags = {}
    for idx, parts in enumerate(parsed):
        for ptype, pval in parts:
            if ptype == "anchor":
                anchor_to_frags.setdefault(pval, []).append(idx)

    visited, new_frags, anchor_map = set(), [], {}
    next_anchor_id = 1

    def format_anchor(num):
        return str(num) if num < 10 else f"%{num}"

    def dfs(frag_idx):
        nonlocal next_anchor_id
        visited.add(frag_idx)

        out, outgoing = "", []
        for ptype, pval in parsed[frag_idx]:
            if ptype == "anchor":
                if pval not in anchor_map:
                    anchor_map[pval] = format_anchor(next_anchor_id)
                    next_anchor_id += 1
                out += anchor_map[pval]
                outgoing.append(pval)
            else:
                out += pval
        new_frags.append(out)

        # follow each anchor to the fragment on its other end
        for old_anchor in outgoing:
            for neighbor in anchor_to_frags.get(old_anchor, []):
                if neighbor not in visited:
                    dfs(neighbor)

    for i in range(len(frags)):
        if i not in visited:
            dfs(i)

    return ".".join(new_frags)


def canonical(safe_str):
    """Canonical SMILES for a SAFE string, or None if it does not decode."""
    try:
        smi = sf.decode(safe_str, canonical=True, ignore_errors=False)
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def main():
    ap = argparse.ArgumentParser(description="Reorder SAFE fragments depth-first.")
    ap.add_argument("--in-path", required=True)
    ap.add_argument("--out-path", required=True)
    ap.add_argument("--safe-col", default="safe")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the decode check (about 3x faster)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    src = df[args.safe_col].astype(str)

    out, n_reordered, n_failed, n_changed_mol = [], 0, 0, 0
    for s in src:
        try:
            r = reorder_and_reindex_safe(s)
        except (RecursionError, Exception):
            out.append(s)
            n_failed += 1
            continue

        # A reordering that alters the molecule is worse than no reordering, so
        # fall back to the original rather than writing a corrupted row.
        if not args.no_verify:
            a, b = canonical(s), canonical(r)
            if a is not None and a != b:
                out.append(s)
                n_changed_mol += 1
                continue

        out.append(r)
        n_reordered += 1

    df[args.safe_col] = out
    df.to_csv(args.out_path, index=False)

    n = len(src)
    print(f"reordered {n_reordered}/{n} SAFE strings")
    if n_failed:
        print(f"  {n_failed} could not be reordered (kept original)")
    if n_changed_mol:
        print(f"  {n_changed_mol} changed the molecule (kept original)")
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
