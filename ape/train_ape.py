"""Train an APE (Atom Pair Encoding) tokenizer on the AMP SAFE corpus.

APE is BPE with a chemistry-aware pre-tokenizer: the corpus is first split into
chemical units -- a bracket atom `[C@@H]` is one unit, a two-digit ring label
`%14` is one unit -- and merges only ever join whole units. The BPE tokenizer in
molecular_dataset/ works the same way at the character level, which is why its
159-token vocabulary contains `%` on its own and `%1` through `%9`: strings that
are not legal SMILES. APE cannot produce those.

    python ape/train_ape.py                          # vocab 1024, full corpus
    python ape/train_ape.py --vocab 512 --limit 2000 # quick run
    python ape/train_ape.py --compare only           # re-report an existing vocab

Writes ape/vocab_<size>.json and prints a length comparison against the current
BPE tokenizer, which is the number that matters here: the diffusion model has to
coordinate every position independently within a denoising step, so halving the
sequence length halves the number of decisions that must agree.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ape_tokenizer import APETokenizer

SAFE_CSV = "molecular_dataset/dataset/data/safe/amp_safe.csv"
BPE_JSON = "molecular_dataset/dataset/data/safe/tokenizer.json"


def load_corpus(path, limit=None, column="safe"):
    """`safe` or `smiles`. APE's pre-tokenizer is a SMILES regex either way --
    SAFE is SMILES syntax plus fragment separators -- so only the column changes."""
    df = pd.read_csv(path)
    if column not in df.columns:
        raise SystemExit(f"{path} has no column '{column}' (has {list(df.columns)})")
    seqs = df[column].dropna().tolist()
    return seqs[:limit] if limit else seqs


def roundtrip_check(tok, corpus, n=200):
    """Encode then decode; the string has to come back byte for byte.

    Not a formality: APETokenizer.encode does greedy longest-match over raw
    characters and does NOT re-run the pre-tokenizer, so a vocabulary entry
    spanning a boundary can in principle be matched where training never put
    one. If this fails the tokenizer cannot be used for generation.
    """
    bad = []
    for s in corpus[:n]:
        ids = tok.encode(s, add_special_tokens=False)
        back = "".join(tok.convert_ids_to_tokens(ids))
        if back != s:
            bad.append((s, back))
    return len(corpus[:n]) - len(bad), bad


def lengths(tok, corpus, kind):
    if kind == "ape":
        return np.array([len(tok.encode(s, add_special_tokens=False)) for s in corpus])
    return np.array([len(tok.tokenize(s)) for s in corpus])


def compare(ape, corpus, sample=1500):
    from safe.tokenizer import SAFETokenizer
    bpe = SAFETokenizer.load(BPE_JSON).get_pretrained()
    sub = corpus[:sample]
    a, b = lengths(ape, sub, "ape"), lengths(bpe, sub, "bpe")
    print(f"\n{'':<22}{'BPE (current)':>16}{'APE':>12}{'change':>10}")
    rows = [("vocabulary size", len(bpe.get_vocab()), len(ape.vocabulary)),
            ("median tokens/mol", int(np.median(b)), int(np.median(a))),
            ("mean tokens/mol", round(float(b.mean())), round(float(a.mean()))),
            ("longest molecule", int(b.max()), int(a.max()))]
    for label, x, y in rows:
        print(f"{label:<22}{x:>16,}{y:>12,}{y/x:>9.2f}x")

    pct_ape = sum(1 for t in ape.vocabulary if "%" in t)
    pct_bpe = sum(1 for t in bpe.get_vocab() if "%" in t)
    print(f"\n{'tokens containing %':<22}{pct_bpe:>16,}{pct_ape:>12,}")
    print(f"{'  as fraction of vocab':<22}{pct_bpe/len(bpe.get_vocab()):>15.0%}"
          f"{pct_ape/len(ape.vocabulary):>12.0%}")
    illegal = sorted(t for t in bpe.get_vocab()
                     if t.startswith("%") and not (len(t) == 3 and t[1:].isdigit()))
    print(f"  BPE tokens that are not legal SMILES: {len(illegal)} "
          f"({', '.join(illegal[:6])}{' ...' if len(illegal) > 6 else ''})")
    # Merges that swallow a ring label are worth seeing: the label numbers are
    # assigned by traversal order, so a token like `%17.N` memorizes an artifact.
    label_merges = [t for t in ape.vocabulary if "%" in t and len(t) > 3]
    print(f"  APE merges that bind a specific label: {len(label_merges)}"
          f"{' e.g. ' + ', '.join(sorted(label_merges)[:6]) if label_merges else ''}")
    cross = [t for t in ape.vocabulary if "." in t and len(t) > 1]
    print(f"  APE merges spanning a '.' fragment break: {len(cross)}"
          f"{' e.g. ' + ', '.join(sorted(cross)[:5]) if cross else ''}")
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=1024, help="max vocabulary size")
    ap.add_argument("--min_freq", type=int, default=50,
                    help="stop merging below this pair count. APETokenizer never "
                         "clears its pair_counts between iterations, so these "
                         "counts accumulate -- keep this low and let --vocab stop it")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N molecules")
    ap.add_argument("--corpus", default=SAFE_CSV)
    ap.add_argument("--column", default="safe", choices=["safe", "smiles"],
                    help="which column to train on")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None,
                    help="path to an existing vocab json; skips training")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    corpus = load_corpus(args.corpus, args.limit, args.column)
    print(f"[Corpus] {len(corpus):,} {args.column} strings from {args.corpus}")

    tok = APETokenizer()
    if args.compare:
        tok.load_vocabulary(args.compare)
        print(f"[Loaded] {args.compare} -> {len(tok.vocabulary)} tokens")
    else:
        t0 = time.time()
        tok.train(corpus, max_vocab_size=args.vocab, min_freq_for_merge=args.min_freq)
        # APETokenizer.train fills self.vocabulary but never refreshes the id ->
        # token map, so a freshly trained tokenizer decodes everything to <unk>.
        # load_vocabulary rebuilds it; training does not.
        tok.update_reverse_vocabulary()
        print(f"[Trained] {len(tok.vocabulary)} tokens in {time.time()-t0:.0f}s")
        out = args.out or os.path.join(here, f"vocab_{args.column}_{len(tok.vocabulary)}.json")
        tok.save_vocabulary(out)
        print(f"[Saved] {out}")

    ok, bad = roundtrip_check(tok, corpus)
    print(f"[Round trip] {ok}/{ok + len(bad)} exact")
    for s, back in bad[:2]:
        print(f"    in  {s[:70]}\n    out {back[:70]}")

    compare(tok, corpus)


if __name__ == "__main__":
    main()
