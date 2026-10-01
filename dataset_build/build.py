"""Build the amide_brics corpus and the vocabulary fitted to it.

    python -m dataset_build.build                       # amide_brics, DFS labels -> data/amide_brics/
    python -m dataset_build.build --order raw           # the encoder's own numbering -> data/amide_brics_raw/
    python -m dataset_build.build --rule brics          # plain BRICS -> data/brics/ (add --order raw for data/brics_raw/)
    python -m dataset_build.build --n 300               # quick check
    python -m dataset_build.build --out_dir DIR

Re-cuts every molecule of the shipped corpus from its `smiles` column, keeps the ones that round-trip, and fits
a new vocabulary on the result. Writes <out_dir>/corpus/modified_amp_safe.csv, <out_dir>/vocab/safe_vocab.csv
and <out_dir>/report.json.

The vocabulary is refitted because its `type` column (special / topology / chemistry) is what the two-schedule
masking reads, and the shipped vocabulary describes the BRICS fragmentation.
"""
import argparse
import json
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

import pandas as pd

from . import slicer

ROOT = Path(__file__).resolve().parent.parent
XH_CORPUS = ROOT / "data" / "xh" / "corpus" / "modified_amp_safe.csv"      # the shipped corpus: where the SMILES come from


def _work(job):
    amp_id, smiles, order, rule = job
    out = slicer.encode(smiles, order, rule)
    return amp_id, out.safe, out.error, out.used_brics


def build_corpus(df, jobs, order="dfs", rule="amide_brics"):
    rows = [(int(i), s, order, rule) for i, s in zip(df.amp_id, df.smiles)]
    if jobs <= 1:
        slicer.quiet()
        results = [_work(r) for r in rows]
    else:
        with mp.get_context("fork").Pool(jobs, initializer=slicer.quiet) as pool:
            results = list(pool.imap(_work, rows, chunksize=16))
    kept = [(i, s) for i, s, e, _ in results if s is not None]
    smiles_of = dict(zip(df.amp_id.astype(int), df.smiles))
    corpus = pd.DataFrame({"amp_id": [i for i, _ in kept],
                           "smiles": [smiles_of[i] for i, _ in kept],
                           "safe": [s for _, s in kept]})
    stats = {
        "n_in": len(df),
        "n_kept": len(corpus),
        "n_dropped": len(df) - len(corpus),
        "n_brics_fallback": sum(1 for _, s, _, b in results if s is not None and b),
        "drop_reasons": dict(Counter(e for _, s, e, _ in results if s is None)),
    }
    return corpus, stats


def refit_vocab(corpus, out_path, appearance_number=None):
    from amp_diffusion.tokenizer import OrthogonalSafeTokenizer

    tok = OrthogonalSafeTokenizer()
    tok.fit(list(corpus.safe), appearance_number=appearance_number)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save_csv(str(out_path))
    v = pd.read_csv(out_path)
    unk = sum("[UNK]" in tok.decode(tok.encode(s)) for s in corpus.safe.head(500))
    return {"vocab_size": len(v), "type_counts": v.type.value_counts().to_dict(), "unk_in_500": int(unk)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(XH_CORPUS), help="a CSV with amp_id and smiles columns")
    ap.add_argument("--rule", choices=slicer.RULES, default="amide_brics")
    ap.add_argument("--order", choices=slicer.ORDERS, default="dfs",
                    help="dfs: labels renumbered in DFS order (the shipped corpus's convention); raw: the encoder's own numbering")
    ap.add_argument("--appearance_number", type=int, default=None,
                    help="chemistry tokens seen fewer times than this are split into commoner ones. Default: 10 for "
                         "--rule brics (reproduces the shipped xh vocabulary), none for amide_brics (as in "
                         "the first amide_brics build)")
    ap.add_argument("--out_dir", default=None, help="default: data/<rule>[_raw]")
    ap.add_argument("--n", type=int, default=None, help="molecules (default: all)")
    ap.add_argument("--jobs", type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    args = ap.parse_args()

    df = pd.read_csv(args.corpus)
    if args.n:
        df = df.head(args.n)
    out_dir = Path(args.out_dir or ROOT / "data" / (args.rule + ("_raw" if args.order == "raw" else "")))
    print(f"{len(df)} molecules from {args.corpus}, {args.jobs} worker(s)", flush=True)

    corpus, cstats = build_corpus(df, args.jobs, args.order, args.rule)
    if corpus.empty:
        raise SystemExit("no molecule survived the round trip")
    corpus_path = out_dir / "corpus" / "modified_amp_safe.csv"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(corpus_path, index=False)
    print(f"corpus: {cstats['n_kept']}/{cstats['n_in']} kept ({cstats['n_brics_fallback']} cut with BRICS) -> {corpus_path}")
    if cstats["drop_reasons"]:
        print(f"  dropped: {cstats['drop_reasons']}")

    appearance = args.appearance_number if args.appearance_number is not None else (10 if args.rule == "brics" else None)
    vstats = refit_vocab(corpus, out_dir / "vocab" / "safe_vocab.csv", appearance)
    vstats["appearance_number"] = appearance
    print(f"vocab: {vstats['vocab_size']} tokens {vstats['type_counts']}, UNK in first 500 strings: {vstats['unk_in_500']}")

    report = {"slicer": args.rule, "order": args.order, "corpus": cstats, "vocab": vstats}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"report -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
