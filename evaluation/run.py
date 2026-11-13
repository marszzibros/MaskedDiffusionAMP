"""Score a sample file.

    python evaluation/run.py results/samples.csv

validity    share of samples that decode to a molecule
novelty     share of valid samples that are not in the training corpus
backbone    share of valid samples with at least 5 peptide-bonded residues in a row
connected   share of all samples that decode to one connected molecule
strict      share of all samples that are connected, have that backbone and no N-N or C(=O)-C(=O) bond
fragments   SAFE fragments per sample, mean and median
"""
import os
import re
import sys
from collections import Counter

from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # the metric files sit beside this one

from backbone import backbone, backbone_length               # noqa: E402
from connectivity import connected, is_connected             # noqa: E402
from fragments import mean_fragments, median_fragments       # noqa: E402
from loading import load_reference, load_samples             # noqa: E402
from novelty import novelty                                  # noqa: E402
from strict import NON_PEPTIDE_LINKS, strict                 # noqa: E402
from validity import validity                                # noqa: E402

_ANCHOR_RE = re.compile(r"\[[^\]]+\]|(%\(\d+\)|%\d{2}|[1-9])")


def _analyze_safe_topology(safe_str):
    """Analyze intra-fragment rings vs inter-fragment attachment points in a SAFE string."""
    frags = safe_str.split(".")
    frag_anchors = [
        [m.group(1) for m in _ANCHOR_RE.finditer(frag) if m.group(1)]
        for frag in frags
    ]
    global_counts = Counter(a for fa in frag_anchors for a in fa)

    # An anchor that appears odd number of times inside a fragment is an inter-fragment attachment point
    inter_counts = Counter()
    for fa in frag_anchors:
        for a, cnt in Counter(fa).items():
            if cnt % 2 == 1:
                inter_counts[a] += 1

    odd_global = [a for a, cnt in global_counts.items() if cnt % 2 == 1]
    odd_inter = [a for a, cnt in inter_counts.items() if cnt % 2 == 1]
    overused_inter = [a for a, cnt in inter_counts.items() if cnt > 2]

    # Count fragments that have NO inter-fragment attachment points at all (isolated fragments)
    isolated_frags = sum(
        1 for fa in frag_anchors
        if not any(Counter(fa)[a] % 2 == 1 for a in fa)
    ) if len(frags) > 1 else 0

    return {
        "odd_global": odd_global,
        "odd_inter": odd_inter,
        "overused_inter": overused_inter,
        "isolated_frags": isolated_frags,
        "num_frags": len(frags),
    }


def _diagnose_invalid_safe(safe_str):
    """Categorize why a SAFE string failed to decode into a valid RDKit molecule."""
    reasons = []
    if "[MASK]" in safe_str or "?" in safe_str:
        reasons.append("unmasked_token")

    # Check parentheses balance outside of [...] atom blocks
    stripped = re.sub(r"\[[^\]]*\]", "", safe_str)
    depth = 0
    unbalanced_paren = False
    for ch in stripped:
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                unbalanced_paren = True
                break
            depth -= 1
    if depth != 0 or unbalanced_paren:
        reasons.append("unbalanced_parens")
    if "()" in stripped:
        reasons.append("empty_parens")

    topo = _analyze_safe_topology(safe_str)
    if topo["odd_global"]:
        reasons.append("odd_linkage_count")
    if topo["overused_inter"]:
        reasons.append("inter_linkage_used(>2_frags)")

    if not reasons:
        reasons.append("chem_valence_or_ring_error")
    return reasons


def analyze_failures(samples, min_residues=5):
    n_total = len(samples)
    invalids = [s for s in samples if not s.valid]
    valids = [s for s in samples if s.valid]

    print("\n--- Failure Mode Analysis ---")
    print(f"1. Invalid SAFE strings: {len(invalids)} / {n_total} ({len(invalids)/n_total:.1%})")
    if invalids:
        reason_counts = Counter()
        for s in invalids:
            for item in _diagnose_invalid_safe(s.safe):
                reason_counts[item] += 1
        for reason, cnt in reason_counts.most_common():
            print(f"     {reason:<28} {cnt:4d} ({cnt/len(invalids):5.1%} of invalid, {cnt/n_total:5.1%} of all)")

    disconnected = [s for s in valids if not is_connected(s.mol)]
    print(f"\n2. Valid but Disconnected: {len(disconnected)} / {n_total} ({len(disconnected)/n_total:.1%})")
    if disconnected:
        frag_counts = Counter(len(Chem.GetMolFrags(s.mol)) for s in disconnected)
        disc_reasons = Counter()
        for s in disconnected:
            topo = _analyze_safe_topology(s.safe)
            if topo["odd_global"]:
                disc_reasons["unpaired_linkage_(odd_global)"] += 1
            if topo["overused_inter"]:
                disc_reasons["duplicate_inter_linkage(>2_frags)"] += 1
            if topo["isolated_frags"] > 0:
                disc_reasons["isolated_frag_(0_inter_links)"] += 1
            if not (topo["odd_global"] or topo["overused_inter"] or topo["isolated_frags"] > 0):
                disc_reasons["disjoint_closed_components"] += 1
        print(f"     component count dist:        {dict(sorted(frag_counts.items()))}")
        for r, cnt in disc_reasons.most_common():
            print(f"     {r:<32} {cnt:4d} ({cnt/len(disconnected):5.1%} of disconnected, {cnt/n_total:5.1%} of all)")

    conn_valids = [s for s in valids if is_connected(s.mol)]
    nn_pat, cc_pat = NON_PEPTIDE_LINKS
    n_nn = sum(s.mol.HasSubstructMatch(nn_pat) for s in conn_valids)
    n_cc = sum(s.mol.HasSubstructMatch(cc_pat) for s in conn_valids)
    n_short_bb = sum(backbone_length(s.mol) < min_residues for s in conn_valids)
    non_strict_conn = [
        s for s in conn_valids
        if backbone_length(s.mol) < min_residues
        or s.mol.HasSubstructMatch(nn_pat)
        or s.mol.HasSubstructMatch(cc_pat)
    ]
    print(f"\n3. Connected but Failed Strict: {len(non_strict_conn)} / {n_total} ({len(non_strict_conn)/n_total:.1%})")
    if conn_valids:
        print(f"     has N-N bond (wrong N-N pair):         {n_nn:4d} ({n_nn/n_total:5.1%} of all)")
        print(f"     has C(=O)-C(=O) bond (wrong C-C pair): {n_cc:4d} ({n_cc/n_total:5.1%} of all)")
        print(f"     short backbone (<{min_residues} residues):          {n_short_bb:4d} ({n_short_bb/n_total:5.1%} of all)")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python evaluation/run.py results/samples.csv")
    samples = load_samples(sys.argv[1])
    if not samples:
        raise SystemExit(f"{sys.argv[1]}: no samples")
    print(f"{sys.argv[1]}  ({len(samples)} samples)")
    print(f"  validity   {validity(samples):.3f}")
    print(f"  novelty    {novelty(samples, load_reference()):.3f}")
    print(f"  backbone   {backbone(samples):.3f}")
    print(f"  connected  {connected(samples):.3f}")
    print(f"  strict     {strict(samples):.3f}")
    print(f"  fragments  mean {mean_fragments(samples):.1f}, median {median_fragments(samples):.1f}")
    analyze_failures(samples)


if __name__ == "__main__":
    main()
