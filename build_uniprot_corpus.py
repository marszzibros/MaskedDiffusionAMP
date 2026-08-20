"""Build a SAFE pretraining corpus from UniRef50 sequences.

    sequence => Chem.MolFromSequence => MolToSmiles => safe.encode

    wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz

Usage:
    python build_uniprot_corpus.py uniref50.fasta.gz out.csv --target 1000000
"""
import argparse, csv, gzip, os, random, sys
from multiprocessing import Pool

STANDARD = set("ACDEFGHIKLMNPQRSTVWY")


def iter_fasta(path):
    """Stream (header, sequence) pairs from a plain or gzipped FASTA."""
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f:
        head, buf = None, []
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if head is not None:
                    yield head, ''.join(buf)
                head, buf = line[1:], []
            else:
                buf.append(line)
        if head is not None:
            yield head, ''.join(buf)


def convert(args):
    """One sequence -> (id, seq, smiles, safe). Worker function."""
    seq_id, seq = args
    # imports live in the worker so each process initialises RDKit once
    import logging; logging.disable(logging.WARNING)
    try:
        from loguru import logger; logger.remove()
    except Exception:
        pass
    import safe as sf
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    try:
        m = Chem.MolFromSequence(seq)
        if m is None:
            return None
        smi = Chem.MolToSmiles(m)
        return (seq_id, seq, smi, sf.encode(smi))
    except Exception:
        return None


def length_sampler(amp_csv):
    """Return a function that accepts/rejects a length so the output matches the
    AMP corpus length distribution -- otherwise fine-tuning has to unlearn a
    length prior that pretraining installed."""
    import pandas as pd
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    df = pd.read_csv(amp_csv).drop_duplicates('amp_id')
    lens = []
    for s in df['smiles'].dropna().head(4000):
        m = Chem.MolFromSmiles(s)
        if m:                                      # ~7.5 heavy atoms per residue
            lens.append(max(3, round(m.GetNumHeavyAtoms() / 7.5)))
    hist = {}
    for L in lens:
        hist[L] = hist.get(L, 0) + 1
    peak = max(hist.values())
    print(f"[lengths] AMP corpus approx. {min(lens)}-{max(lens)} residues, "
          f"median {sorted(lens)[len(lens)//2]}")
    return lambda L: random.random() < hist.get(L, 0) / peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fasta')
    ap.add_argument('out_csv')
    ap.add_argument('--target', type=int, default=1000000)
    ap.add_argument('--min-len', type=int, default=5)
    ap.add_argument('--max-len', type=int, default=64,
                    help="64 residues ~= 1150 SAFE tokens, just inside max_length 1301")
    ap.add_argument('--match-lengths', metavar='AMP_CSV', default=None)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    accept_len = length_sampler(a.match_lengths) if a.match_lengths else (lambda L: True)

    def candidates():
        seen = set()
        kept = 0
        for i, (head, seq) in enumerate(iter_fasta(a.fasta)):
            if kept >= a.target:
                return
            if not (a.min_len <= len(seq) <= a.max_len):
                continue
            # UniProt sequences carry X/B/Z/U/O -- MolFromSequence chokes on them
            if not set(seq) <= STANDARD:
                continue
            if seq in seen:
                continue
            if not accept_len(len(seq)):
                continue
            seen.add(seq)
            kept += 1
            yield (head.split()[0], seq)

    n_ok = n_fail = 0
    with open(a.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['uniref_id', 'sequence', 'smiles', 'safe'])
        with Pool(a.workers) as pool:
            for rec in pool.imap_unordered(convert, candidates(), chunksize=200):
                if rec is None:
                    n_fail += 1
                    continue
                w.writerow(rec)
                n_ok += 1
                if n_ok % 25_000 == 0:
                    print(f"  {n_ok:,} converted ({n_fail:,} failed)", flush=True)

    print(f"\nwrote {a.out_csv}")
    print(f"  converted {n_ok:,}   failed {n_fail:,}")
    print(f"\nnext: point AMPSafeDataset at this file, set cond_dropout=1.0, "
          f"and emit zero condition tensors for these rows.")


if __name__ == '__main__':
    main()
