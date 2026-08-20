"""Structural metrics for generated SAFE samples.

Reads a sample file written by sample.py (`<safe_string>\t<smiles|INVALID>` per
line) and reports the quantities that actually move during training, rather than
the all-or-nothing validity rate:

  ring_pair   fraction of ring/attachment labels appearing exactly twice.
              Every label must appear exactly twice to form a bond, so this is
              the constraint that gates whether a molecule assembles at all.
  frag_paren  fraction of '.'-separated fragments with balanced parentheses
              (local, within-fragment correctness).
  components  disconnected pieces per decoded molecule -- a real peptide is 1.
  amides      C(=O)N bonds per molecule; the peptide backbone. Corpus is ~17.
  diketones   C(=O)C(=O) per molecule; the signature of a linker that failed to
              bond, turning an amide into a 1,2-diketone. Corpus is 0.

Usage:
    python metrics.py samples.tsv [--label steps100_eta1] [--append summary.csv]
"""
import argparse
import collections
import csv
import re
import sys

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

TOKEN = re.compile(r'(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])')
RING = re.compile(r'%\d\d|\d')
AMIDE = Chem.MolFromSmarts('[CX3](=O)[NX3]')
DIKETONE = Chem.MolFromSmarts('[CX3](=O)[CX3]=O')


def ring_pairing(safe_str):
    """Fraction of ring labels appearing an EVEN number of times.

    Not "exactly twice": a label is free to be reused once its bond closes, and
    73% of the corpus does exactly that. Scoring on `== 2` puts the ceiling at
    0.963 for real molecules and understates generated samples by ~0.15.
    """
    counts = collections.Counter(t for t in TOKEN.findall(safe_str) if RING.fullmatch(t))
    if not counts:
        return None
    return sum(1 for v in counts.values() if v % 2 == 0) / len(counts)


def frag_parens(safe_str):
    frags = safe_str.split('.')
    return sum(1 for f in frags if f.count('(') == f.count(')')) / len(frags), len(frags)


def read_samples(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            safe, _, smiles = line.partition('\t')
            rows.append((safe, None if smiles in ('', 'INVALID') else smiles))
    return rows


def summarize(path):
    rows = read_samples(path)
    if not rows:
        return None

    pair, paren, nfrag = [], [], []
    for safe, _ in rows:
        p = ring_pairing(safe)
        if p is not None:
            pair.append(p)
        fp, nf = frag_parens(safe)
        paren.append(fp)
        nfrag.append(nf)

    valid = [s for _, s in rows if s]
    comps, amides, diket, heavy = [], [], [], []
    for smiles in valid:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            continue
        comps.append(len(Chem.GetMolFrags(m)))
        amides.append(len(m.GetSubstructMatches(AMIDE)))
        diket.append(len(m.GetSubstructMatches(DIKETONE)))
        heavy.append(m.GetNumHeavyAtoms())

    mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')
    return {
        'n': len(rows),
        'valid': len(valid) / len(rows),
        'ring_pair': mean(pair),
        'frag_paren': mean(paren),
        'frags': mean(nfrag),
        'components': mean(comps),
        'amides': mean(amides),
        'diketones': mean(diket),
        'heavy': mean(heavy),
    }


FIELDS = ['label', 'n', 'valid', 'ring_pair', 'frag_paren', 'frags',
          'components', 'amides', 'diketones', 'heavy']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('samples')
    ap.add_argument('--label', default='')
    ap.add_argument('--append', default=None,
                    help='CSV to append one summary row to (header written if new)')
    args = ap.parse_args()

    s = summarize(args.samples)
    if s is None:
        print(f'{args.samples}: empty', file=sys.stderr)
        return 1
    s['label'] = args.label or args.samples

    print(f"{s['label']:>22}  valid {s['valid']:5.1%}  ring_pair {s['ring_pair']:.3f}  "
          f"frag_paren {s['frag_paren']:.1%}  comp {s['components']:5.1f}  "
          f"amide {s['amides']:5.1f}  diket {s['diketones']:4.1f}  heavy {s['heavy']:5.0f}")

    if args.append:
        import os
        new = not os.path.exists(args.append)
        with open(args.append, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: s[k] for k in FIELDS})
    return 0


if __name__ == '__main__':
    sys.exit(main())
