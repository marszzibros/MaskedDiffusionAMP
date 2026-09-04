"""
Train a SAFE tokenizer on the SAFE strings produced by smiles_to_safe.py.

Uses safe.tokenizer.SAFETokenizer (a BPE tokenizer whose pre-tokenizer is SAFE's
SMILES-aware splitter). Saves the trained tokenizer as a tokenizer.json that can
be reloaded with SAFETokenizer.load(path).

Usage:
    python train_safe_tokenizer.py
    python train_safe_tokenizer.py --vocab-size 1000 --min-frequency 2
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from safe import SAFETokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a SAFE tokenizer.")
    parser.add_argument("--in-path", default="dataset/data/safe/amp_safe.csv")
    parser.add_argument("--out-path", default="dataset/data/safe/tokenizer.json")
    parser.add_argument("--safe-col", default="safe")
    parser.add_argument("--tokenizer-type", default="bpe", help="'bpe' or 'word'")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--min-frequency", type=int, default=2)
    # With splitter="safe" the SAFE pre-tokenizer splits to single atoms, so BPE
    # has nothing left to merge and --vocab-size / --min-frequency are inert
    # (vocab saturates ~159, sequence length does not move at all). Pass
    # --splitter none to let BPE actually learn merges.
    parser.add_argument("--splitter", default="safe", choices=["safe", "none"])
    args = parser.parse_args()

    # run relative to the project root so default paths resolve consistently
    os.chdir(Path(__file__).resolve().parents[1])

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    df = pd.read_csv(args.in_path)
    safe_strings = df[args.safe_col].dropna().astype(str).tolist()
    if not safe_strings:
        raise SystemExit(f"No SAFE strings found in {args.in_path}:{args.safe_col}")

    # vocab_size / min_frequency are set on the trainer, via trainer_args
    tokenizer = SAFETokenizer(
        tokenizer_type=args.tokenizer_type,
        splitter=(None if args.splitter == "none" else "safe"),
        trainer_args={"vocab_size": args.vocab_size, "min_frequency": args.min_frequency},
    )
    tokenizer.train_from_iterator(safe_strings)
    tokenizer.save(args.out_path)

    print(f"trained on {len(safe_strings)} SAFE strings (splitter={args.splitter})")
    print(f"vocab size: {len(tokenizer)}")
    print(f"saved tokenizer -> {args.out_path}")

    # quick sanity check on the first molecule
    sample = safe_strings[0]
    ids = tokenizer.encode(sample)
    print(f"\nexample SAFE : {sample[:70]}")
    print(f"token ids    : {ids[:25]}{' ...' if len(ids) > 25 else ''}")


if __name__ == "__main__":
    main()
