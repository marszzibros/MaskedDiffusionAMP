"""
Score the tokenizers produced by run.sh.

For each (slicer, splitter) arm this reports the three things that actually
decide whether an arm is worth training on:

  length     -- tokens per molecule. Sets the diffusion sequence length, so it
                sets the cost of the arm.
  pair span  -- token distance between the two halves of a ring-closure pair.
                This is the cross-token constraint budget: every pair the model
                has to close across N positions is a constraint it can violate.
                Pairs whose two ends land INSIDE one token are free.
  validity   -- exact tokenizer round-trip, plus SAFE -> SMILES parse. An arm
                that compresses well but cannot decode is worthless.

Usage:  python scripts/tokenizer_report.py --dir tokenizers [--sample 1500]
"""

import argparse
import csv
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import safe as sf
from safe import SAFETokenizer

RDLogger.DisableLog("rdApp.*")


def ring_events(s):
    """(char_index, label) for every ring-closure marker outside a bracket atom.

    Digits inside [...] are charges/H-counts, not ring bonds, so bracket depth
    has to be tracked rather than just regexing for digits.
    """
    out, depth, i = [], 0, 0
    while i < len(s):
        c = s[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
        elif depth == 0:
            if c == "%":
                m = re.match(r"%\((\d+)\)", s[i:]) or re.match(r"%(\d{2})", s[i:])
                if m:
                    out.append((i, "R" + m.group(1)))
                    i += m.end()
                    continue
            elif c.isdigit():
                out.append((i, "R" + c))
        i += 1
    return out


def char_to_token(tokens):
    """Map each character offset of the joined token stream to its token index."""
    m = []
    for ti, t in enumerate(tokens):
        m.extend([ti] * len(t))
    return m


def score(tok_path, safe_strings, sample):
    tok = SAFETokenizer.load(tok_path)
    hf = tok.get_pretrained()
    specials = {t for t in (hf.bos_token, hf.eos_token, hf.cls_token,
                            hf.sep_token, hf.pad_token, hf.unk_token) if t}

    lens, spans = [], []
    intra = roundtrip_ok = smiles_ok = n = 0

    for s in safe_strings[:sample]:
        ids = tok.encode(s)
        lens.append(len(ids))
        n += 1

        if hf.decode(ids, skip_special_tokens=True) == s:
            roundtrip_ok += 1

        try:
            smi = sf.decode(s, canonical=True, ignore_errors=False)
            smiles_ok += Chem.MolFromSmiles(smi) is not None
        except Exception:
            pass

        toks = [t for t in hf.convert_ids_to_tokens(ids) if t not in specials]
        if "".join(toks) != s:
            continue                       # not char-exact; span would be wrong
        c2t = char_to_token(toks)
        open_at = {}
        for ci, lab in ring_events(s):
            if ci >= len(c2t):
                continue
            ti = c2t[ci]
            if lab in open_at:
                d = ti - open_at.pop(lab)
                spans.append(d)
                intra += d == 0
            else:
                open_at[lab] = ti

    pct = lambda a, b: 100.0 * a / b if b else 0.0
    return dict(
        vocab=len(tok),
        n=n,
        med_len=float(np.median(lens)),
        p95_len=float(np.percentile(lens, 95)),
        max_len=int(np.max(lens)),
        med_span=float(np.median(spans)) if spans else float("nan"),
        p95_span=float(np.percentile(spans, 95)) if spans else float("nan"),
        n_pairs=len(spans),
        intra_pct=pct(intra, len(spans)),
        roundtrip_pct=pct(roundtrip_ok, n),
        smiles_pct=pct(smiles_ok, n),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tokenizers")
    ap.add_argument("--sample", type=int, default=1500,
                    help="molecules scored per arm (span math is the slow part)")
    ap.add_argument("--out", default=None, help="CSV to write (default <dir>/report.csv)")
    args = ap.parse_args()

    manifest = os.path.join(args.dir, "manifest.tsv")
    if not os.path.exists(manifest):
        sys.exit(f"no manifest at {manifest} -- run ./run.sh first")

    arms = []
    with open(manifest) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                arms.append(line.split("\t"))

    corpus_cache = {}
    rows = []
    hdr = (f"{'slicer':>9} {'splitter':>8} {'vocab':>6} | {'medlen':>6} {'p95len':>6} | "
           f"{'medspan':>7} {'p95span':>7} {'intra%':>6} | {'rt%':>5} {'smi%':>5}")
    print(hdr)
    print("-" * len(hdr))

    for slicer, splitter, corpus, tok_path in arms:
        if not os.path.exists(tok_path):
            print(f"{slicer:>9} {splitter:>8}   MISSING tokenizer ({tok_path})")
            continue
        if corpus not in corpus_cache:
            df = pd.read_csv(corpus)
            corpus_cache[corpus] = df["safe"].dropna().astype(str).tolist()
        r = score(tok_path, corpus_cache[corpus], args.sample)
        r.update(slicer=slicer, splitter=splitter, corpus=corpus, tokenizer=tok_path)
        rows.append(r)
        print(f"{slicer:>9} {splitter:>8} {r['vocab']:>6} | {r['med_len']:>6.0f} {r['p95_len']:>6.0f} | "
              f"{r['med_span']:>7.0f} {r['p95_span']:>7.0f} {r['intra_pct']:>5.1f}% | "
              f"{r['roundtrip_pct']:>4.0f}% {r['smiles_pct']:>4.0f}%")
        sys.stdout.flush()

    if not rows:
        sys.exit("no arms scored")

    out = args.out or os.path.join(args.dir, "report.csv")
    cols = ["slicer", "splitter", "vocab", "n", "med_len", "p95_len", "max_len",
            "med_span", "p95_span", "n_pairs", "intra_pct", "roundtrip_pct",
            "smiles_pct", "corpus", "tokenizer"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(f"\nwrote {out}")

    best = min(rows, key=lambda r: (r["med_span"] if r["med_span"] == r["med_span"] else 1e9))
    print(f"lowest median pair span: {best['slicer']}/{best['splitter']} "
          f"(span {best['med_span']:.0f}, len {best['med_len']:.0f}, "
          f"round-trip {best['roundtrip_pct']:.0f}%)")


if __name__ == "__main__":
    main()
