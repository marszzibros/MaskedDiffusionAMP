"""Score a sample dump: how many decode, how many are one peptide, how many are new.

    python peptide_metrics.py samples.tsv
    python peptide_metrics.py samples.tsv --out per_sample.csv
    python peptide_metrics.py a.tsv b.tsv          # compare two runs
"""
import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

AAS = "ACDEFGHIKLMNPQRSTVWY"
# N-CA-C=O. CA is X4 with one or two hydrogens, which admits glycine and
# excludes the alpha carbon of anything already substituted twice over.
BACKBONE = Chem.MolFromSmarts("[NX3][CX4;H1,H2][CX3]=[OX1]")
SSBOND = Chem.MolFromSmarts("[SX2][SX2]")
CATIONIC, ANIONIC = "KR", "DE"

_taut = rdMolStandardize.TautomerEnumerator()
_taut_cache = {}


def _canonical_tautomer(smiles):
    """Arginine alone appears as both CCCN=C(N)N and CCCNC(N)=N in this corpus,
    which are the same residue and different strings. Canonicalize, and memoize:
    a few dozen distinct residues cover an entire sample file."""
    if smiles not in _taut_cache:
        m = Chem.MolFromSmiles(smiles)
        _taut_cache[smiles] = Chem.MolToSmiles(_taut.Canonicalize(m)) if m else None
    return _taut_cache[smiles]


def cut_disulfides(mol):
    """A disulfide joins two cysteine side chains into one connected system, so a
    naive side-chain walk swallows the partner residue and both read as X. Break
    S-S first; each sulfur picks up a hydrogen and reads as a free thiol again."""
    hits = mol.GetSubstructMatches(SSBOND)
    if not hits:
        return mol, 0
    em = Chem.RWMol(mol)
    for a, b in hits:
        em.RemoveBond(a, b)
    out = em.GetMol()
    Chem.SanitizeMol(out)
    return out, len(hits)


def find_residues(mol):
    """{alpha carbon: (N, CA, C, O)}. Keyed on CA so a residue matched twice
    (the pattern can hit the same alpha carbon through two nitrogens) counts once."""
    out = {}
    for n, ca, c, o in mol.GetSubstructMatches(BACKBONE):
        out.setdefault(ca, (n, ca, c, o))
    return out


def _residue_atoms(mol, n, ca, c, o, blocked):
    """N, CA, C=O plus the side chain: everything reachable from CA without
    passing through the backbone N or C, and without entering another residue."""
    keep = {n, ca, c, o}
    stack = [a.GetIdx() for a in mol.GetAtomWithIdx(ca).GetNeighbors()
             if a.GetIdx() not in (n, c)]
    seen = set(stack)
    while stack:
        i = stack.pop()
        if i in blocked:
            continue
        keep.add(i)
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nb.GetIdx()
            if j not in seen and j not in (n, c):
                seen.add(j)
                stack.append(j)
    return keep


def residue_key(mol, n, ca, c, o, blocked=frozenset()):
    """Canonical SMILES of one residue with CA's chirality erased.

    The chirality is dropped on purpose: the key identifies *which* amino acid,
    and L/D is read separately off CA's CIP code. Keying on the whole residue
    rather than the side chain alone matters -- side chains alone collapse
    Pro onto Val and Ile onto Leu.
    """
    keep = _residue_atoms(mol, n, ca, c, o, blocked)
    em = Chem.RWMol(mol)
    for i in sorted(set(range(mol.GetNumAtoms())) - keep, reverse=True):
        em.RemoveAtom(i)
    sub = em.GetMol()
    remap = {a: k for k, a in enumerate(sorted(keep))}
    sub.GetAtomWithIdx(remap[ca]).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    try:
        Chem.SanitizeMol(sub)
    except Exception:
        return None
    return _canonical_tautomer(Chem.MolToSmiles(sub))


def build_reference():
    """{residue key: (letter, CIP code of the L form)}.

    The L/D rule is read off the reference rather than hardcoded as "L is S":
    cysteine's L form is R, because the sulfur outranks the carboxyl.
    """
    ref = {}
    for a in AAS:
        m = Chem.MolFromSequence(a)
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)
        n, ca, c, o = next(iter(find_residues(m).values()))
        ref[residue_key(m, n, ca, c, o)] = (
            a, m.GetAtomWithIdx(ca).GetPropsAsDict().get("_CIPCode"))
    return ref


def canonical_rotation(seq):
    """Smallest rotation. A head-to-tail cyclic peptide has no first residue, so
    two dumps of the same ring can start anywhere; compare rings by this."""
    return min((seq[i:] + seq[:i] for i in range(len(seq))), default=seq)


def _acyl_carbons(mol, carbonyl, came_from):
    """Carbon count of an acyl cap, counting the carbonyl carbon itself, so
    acetyl is 2 and palmitoyl is 16 -- DBAASP's own convention. Returns None if
    the cap is not a plain hydrocarbon."""
    n, seen, stack, plain = 0, {came_from}, [carbonyl], True
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        a = mol.GetAtomWithIdx(i)
        if a.GetSymbol() == "O" and a.GetDegree() == 1:
            continue                      # the carbonyl oxygen
        if a.GetSymbol() != "C":
            plain = False
            continue
        n += 1
        stack.extend(nb.GetIdx() for nb in a.GetNeighbors())
    return n if plain else None


def terminus_caps(mol, res, chain):
    """What sits on each end of the chain, in DBAASP's vocabulary.

    A cap is invisible in the sequence string -- the side-chain walk starts at
    CA and never crosses the backbone N -- so a lipopeptide and its bare analogue
    read identically. That matches how DBAASP stores them (its `sequence` column
    omits caps, they live in nTerminus/cTerminus), but it means the cap has to be
    reported separately or it is lost.
    """
    n_atom = res[chain[0]][0]
    c_atom = res[chain[-1]][2]

    n_term = "free"
    for nb in mol.GetAtomWithIdx(n_atom).GetNeighbors():
        if nb.GetIdx() == chain[0] or nb.GetSymbol() != "C":
            continue
        if any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(nb).GetSymbol() == "O"
               for b in nb.GetBonds()):
            k = _acyl_carbons(mol, nb.GetIdx(), n_atom)
            n_term = "other" if k is None else ("ACT" if k == 2 else f"C{k}")
        else:
            n_term = "N-alkyl"

    c_term = "free acid"
    for nb in mol.GetAtomWithIdx(c_atom).GetNeighbors():
        if nb.GetIdx() == chain[-1] or nb.GetSymbol() == "O":
            continue
        if nb.GetSymbol() == "N":
            c_term = "AMD" if nb.GetDegree() == 1 else "other amide"
        else:
            c_term = "other"
    if mol.GetAtomWithIdx(c_atom).GetDegree() == 2:
        c_term = "aldehyde"               # C(=O)H: the chain was cut, not capped
    return n_term, c_term


def sequence_from_mol(mol, ref):
    """Read a molecule as a peptide. None when it has no backbone at all."""
    mol, n_ss = cut_disulfides(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    res = find_residues(mol)
    if not res:
        return None
    backbone_atoms = {i for v in res.values() for i in v}

    # residue i precedes j when i's carbonyl carbon bonds to j's amide nitrogen
    nxt, prev = {}, {}
    by_n = {v[0]: ca for ca, v in res.items()}
    for ca, (_, _, cc, _) in res.items():
        for nb in mol.GetAtomWithIdx(cc).GetNeighbors():
            j = by_n.get(nb.GetIdx())
            if j is not None and j != ca:
                nxt[ca], prev[j] = j, ca

    starts = [ca for ca in res if ca not in prev]
    cyclic = not starts              # no residue without a predecessor = a ring
    if cyclic:
        starts = [min(res)]
    chains = []
    for s in starts:
        chain, seen, cur = [], set(), s
        while cur is not None and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = nxt.get(cur)
        chains.append(chain)
    chains.sort(key=len, reverse=True)

    letters = []
    for ca in chains[0]:
        n, _, c, o = res[ca]
        hit = ref.get(residue_key(mol, n, ca, c, o, backbone_atoms - {n, ca, c, o}))
        if hit is None:
            letters.append("X")
            continue
        letter, cip_L = hit
        cip = mol.GetAtomWithIdx(ca).GetPropsAsDict().get("_CIPCode")
        # cip_L is None for glycine, which has no alpha stereocentre.
        letters.append(letter if (cip_L is None or cip == cip_L) else letter.lower())

    seq = "".join(letters)
    n_std = sum(c != "X" for c in seq)
    n_term, c_term = ("cyclic", "cyclic") if cyclic else terminus_caps(mol, res, chains[0])
    return dict(
        sequence=seq,
        n_term=n_term,
        c_term=c_term,
        n_residues=len(res),
        # A backbone that breaks into several chains inside one component means
        # the longest chain is not the whole story; the sequence understates it.
        chain_covers_all=len(chains[0]) == len(res),
        n_chains=len(chains),
        cyclic=cyclic,
        n_disulfide=n_ss,
        pct_standard=n_std / len(seq) if seq else 0.0,
        n_D=sum(c.islower() for c in seq),
        net_charge=sum(c.upper() in CATIONIC for c in seq) - sum(c.upper() in ANIONIC for c in seq),
    )


def read_samples(path, decode_safe=True):
    """`safe<TAB>smiles` as sample.py writes it. A single-column file is taken as
    SAFE strings and decoded here."""
    rows = []
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if not parts[0]:
            continue
        safe_str, smi = parts[0], (parts[1] if len(parts) > 1 else None)
        if smi is None and decode_safe:
            try:
                import safe as sf
                smi = sf.decode(safe_str, as_mol=False, fix=True, remove_dummies=True)
            except Exception:
                smi = None
        rows.append((safe_str, None if smi in (None, "INVALID") else smi))
    return rows


def load_training(train_csv, cache=".train_canonical.csv"):
    """Canonical SMILES and sequences of the training molecules.

    Canonicalizing 20k SMILES takes about a minute, so keep the result next to
    the data; delete the cache if the corpus changes.
    """
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(train_csv):
        t = pd.read_csv(cache)
    else:
        d = pd.read_csv(train_csv)
        smis = []
        for s in d.smiles:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            smis.append(Chem.MolToSmiles(m) if m else None)
        t = pd.DataFrame({"canonical": smis, "sequence": d.get("sequence")})
        t.to_csv(cache, index=False)
    seqs = set(t.sequence.dropna()) if "sequence" in t else set()
    return (set(t.canonical.dropna()),
            seqs,
            {canonical_rotation(s) for s in seqs})


def score(path, ref, train, min_residues=3):
    train_smiles, train_seqs, train_rings = train
    rows = []
    for safe_str, smi in read_samples(path):
        mol = Chem.MolFromSmiles(smi) if smi else None
        row = dict(safe=safe_str, smiles=smi, valid=mol is not None)
        if mol is not None:
            canon = Chem.MolToSmiles(mol)
            frags = Chem.GetMolFrags(mol)
            row.update(canonical=canon, n_frag=len(frags),
                       heavy=mol.GetNumHeavyAtoms(),
                       known_molecule=canon in train_smiles)
            if len(frags) == 1:
                info = sequence_from_mol(mol, ref)
                if info:
                    row.update(info)
                    seq = info["sequence"]
                    ring = canonical_rotation(seq)
                    row["known_sequence"] = (seq in train_seqs
                                             or (info["cyclic"] and ring in train_rings))
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ("n_frag", "n_residues", "pct_standard", "known_sequence",
                "chain_covers_all", "net_charge", "cyclic", "n_term", "c_term"):
        if col not in df:
            df[col] = np.nan

    # These columns are object dtype -- only the single-component rows got a
    # value, the rest are NaN. astype(bool) after fillna is not optional: `~` on
    # an object column inverts the Python bools bitwise (~True is -2, which is
    # truthy) and silently marks everything as novel.
    def flag(col):
        return df[col].fillna(False).astype(bool)

    df["single"] = df.valid & (df.n_frag == 1)
    # A peptide, not just a molecule containing a peptide bond: one unbroken
    # chain of at least min_residues residues accounting for the whole backbone.
    df["is_peptide"] = df.single & (df.n_residues >= min_residues) & flag("chain_covers_all")
    df["all_standard"] = df.is_peptide & (df.pct_standard == 1.0)
    # Nested under all_standard on purpose: a sequence containing an X is one
    # this script could not fully read, so calling it novel would be a claim
    # about a string it does not actually know.
    df["novel"] = df.all_standard & ~flag("known_sequence")
    return df


def funnel(df, name, min_residues=3):
    n = len(df)
    steps = [
        ("generated", n),
        ("decodes to a molecule", int(df.valid.sum())),
        ("one connected component", int(df.single.sum())),
        (f"peptide backbone (>={min_residues} res)", int(df.is_peptide.sum())),
        ("every residue identified", int(df.all_standard.sum())),
        ("sequence new to the training set", int(df.novel.sum())),
    ]
    print(f"\n{name}")
    print(f"  {'step':<34} {'n':>6} {'% of all':>9} {'% of prev':>10}")
    prev = n
    for label, k in steps:
        print(f"  {label:<34} {k:>6} {100*k/max(n,1):>8.1f}% {100*k/max(prev,1):>9.1f}%")
        prev = k
    pep = df[df.is_peptide]
    if len(pep):
        print(f"  ---- of the {len(pep)} peptides ----")
        print(f"  median length                      {pep.n_residues.median():.0f} residues")
        print(f"  cyclic                             {int(pep.cyclic.sum())}")
        print(f"  net charge >= +2                   {int((pep.net_charge >= 2).sum())}")
        print(f"  cationic and >= 8 residues         "
              f"{int(((pep.net_charge >= 2) & (pep.n_residues >= 8)).sum())}")
        print(f"  exact training copies              {int(pep.known_sequence.fillna(False).astype(bool).sum())}")
        caps = pep.n_term.value_counts().head(4).to_dict()
        print(f"  N-terminus                         {caps}")
        print(f"  C-terminus                         {pep.c_term.value_counts().head(4).to_dict()}")
    if "known_molecule" in df:
        seen = df.known_molecule.fillna(False).astype(bool)
        print(f"  ---- of the {int(df.valid.sum())} decoded molecules ----")
        print(f"  exact training molecules           {int(seen.sum())} "
              f"({100*seen.sum()/max(int(df.valid.sum()),1):.1f}%)")


def selftest(ref, train_csv, n=300, seed=1):
    """Round-trip the twenty residues, then re-read molecules whose sequence
    DBAASP already records. Run this after touching the extraction code."""
    got = sequence_from_mol(Chem.MolFromSequence(AAS), ref)["sequence"]
    print(f"[selftest] 20 residues round trip: {got} {'OK' if got == AAS else 'FAILED'}")
    assert got == AAS, got

    d = pd.read_csv(train_csv)
    d = d[d.smiles.notna() & d.sequence.notna()].sample(n, random_state=seed)
    tally = dict(exact=0, case=0, rotation=0, substring=0, other=0)
    examples = []
    for _, r in d.iterrows():
        mol = Chem.MolFromSmiles(r.smiles)
        info = sequence_from_mol(mol, ref) if mol else None
        seq = info["sequence"] if info else ""
        if seq == r.sequence:
            tally["exact"] += 1
        elif seq.upper() == r.sequence.upper():
            tally["case"] += 1
        elif len(seq) == len(r.sequence) and canonical_rotation(seq) == canonical_rotation(r.sequence):
            tally["rotation"] += 1
        elif seq and seq in r.sequence:
            tally["substring"] += 1
        else:
            tally["other"] += 1
            if len(examples) < 5:
                examples.append((r.sequence, seq))
    ok = tally["exact"] + tally["case"] + tally["rotation"]
    print(f"[selftest] {n} DBAASP molecules: {tally}")
    print(f"[selftest] recovered {100*ok/n:.1f}% (exact, case, or rotation of a cyclic peptide)")
    for want, have in examples:
        print(f"    DBAASP {want}\n    read   {have}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tsv", nargs="*", help="sample dump(s) from sample.py")
    ap.add_argument("--selftest", action="store_true",
                    help="check the sequence reader against DBAASP and exit")
    ap.add_argument("--train", default="molecular_dataset/dataset/data/dbaasp/amp.csv",
                    help="training corpus, needs `smiles` and `sequence` columns")
    ap.add_argument("--min_residues", type=int, default=3,
                    help="shortest chain still called a peptide")
    ap.add_argument("--out", default=None,
                    help="write the per-sample table (one file per input, suffixed)")
    args = ap.parse_args()

    ref = build_reference()
    if args.selftest:
        selftest(ref, args.train)
        return
    if not args.tsv:
        raise SystemExit("pass at least one sample .tsv, or --selftest")
    train = load_training(args.train)
    print(f"[Reference] {len(ref)} residues | [Training] {len(train[0])} molecules, "
          f"{len(train[1])} sequences")

    for path in args.tsv:
        df = score(path, ref, train, args.min_residues)
        funnel(df, os.path.basename(path), args.min_residues)
        if args.out:
            out = args.out if len(args.tsv) == 1 else \
                f"{os.path.splitext(args.out)[0]}_{os.path.splitext(os.path.basename(path))[0]}.csv"
            df.to_csv(out, index=False)
            print(f"  -> {out}")


if __name__ == "__main__":
    main()
