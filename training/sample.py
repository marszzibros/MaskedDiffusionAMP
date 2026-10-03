"""Sample molecules from a trained checkpoint.

    python -m training.sample --checkpoint_path output/<run>/model-final.ckpt --num_samples 256

Writes results/samples.csv with columns smiles,safe (INVALID where the SAFE string does not decode), which
evaluation/run.py scores. The tokenizer and label order are read from model_config.json beside the checkpoint.
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from amp_diffusion import DiscreteFlowMatching, SafeDecoder
from . import variants


def load_model(path, device, attn_backend):
    # a checkpoint remembers the attention kernel it trained with; override it to sample on other hardware
    overrides = {"attn_backend": attn_backend} if attn_backend else {}
    model = DiscreteFlowMatching.load_from_checkpoint(path, map_location=device, **overrides)
    model.to(device)
    model.ema.move_shadow_params_to_device(device)
    return model.eval()


def saved_choice(checkpoint_path):
    """(tokenizer, order) that train.py recorded beside the checkpoint."""
    cfg = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "model_config.json")
    if not os.path.exists(cfg):
        raise SystemExit(f"{cfg} not found: pass --tokenizer and --order")
    saved = json.load(open(cfg))
    return saved["tokenizer"], saved.get("order", "dfs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--num_samples", type=int, required=True)
    ap.add_argument("--output_file", default="results/samples.csv")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--attn_backend", choices=("fa2", "sdpa", "auto"), default=None,
                    help="override the checkpoint's attention kernel (sdpa runs on any GPU)")
    ap.add_argument("--topo_eta", type=float, default=None, help="Langevin noise on topology tokens (default: the checkpoint's)")
    ap.add_argument("--chem_eta", type=float, default=None)
    ap.add_argument("--base_eta", type=float, default=None)
    ap.add_argument("--species", type=int, nargs="+", default=[0])
    ap.add_argument("--groups", type=int, nargs="+", default=[0])
    ap.add_argument("--mic", type=int, default=2, help="MIC bin; each sample uses this bin or a neighbour")
    ap.add_argument("--tokenizer", choices=variants.VARIANTS, default=None, help="default: from model_config.json")
    ap.add_argument("--order", choices=variants.ORDERS, default=None, help="default: from model_config.json")
    ap.add_argument("--no_grammar_check", action="store_true",
                    help="turn off the one-shot SAFE grammar repair at t=0.5")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint_path, device, args.attn_backend)
    for name in ("topo_eta", "chem_eta", "base_eta"):
        if getattr(args, name) is not None:
            setattr(model, name, getattr(args, name))
    model.hparams.num_steps = args.steps

    tokenizer, order = saved_choice(args.checkpoint_path) if args.tokenizer is None or args.order is None else (None, None)
    v = variants.resolve(args.tokenizer or tokenizer, args.order or order)
    decoder = SafeDecoder(v.vocab_path)
    if len(decoder.token_dict) != model.hparams.num_tokens:
        raise SystemExit(f"{v.name}/{v.order} has {len(decoder.token_dict)} tokens but the checkpoint was trained on "
                         f"{model.hparams.num_tokens}: wrong --tokenizer or --order")
    length_pool = decoder.length_pool(v.corpus_csv)       # generation lengths are drawn from the corpus
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    n_valid = 0
    with open(args.output_file, "w", newline="") as f, torch.no_grad():
        out = csv.writer(f)
        out.writerow(["smiles", "safe"])
        for start in tqdm(range(0, args.num_samples, args.batch_size), desc="Sampling"):
            n = min(args.batch_size, args.num_samples - start)
            mic = torch.zeros(n, model.dims["mic"], device=device)
            for b in range(n):
                mic[b, random.randint(max(0, args.mic - 1), min(model.dims["mic"] - 1, args.mic + 1))] = 1.0
            seqs = model.generate_sample(
                tokens_dict=decoder.token_dict,
                conditions={"species": args.species, "groups": args.groups, "mic": mic},
                scales={"species": 1.0, "groups": 1.0, "mic": 1.0},
                num_samples=n, max_length=model.hparams.max_length,
                length_pool=length_pool, decode_fn=decoder.decode,
                use_grammar_check=not args.no_grammar_check)
            for seq in seqs:
                smiles = decoder.smiles_from_safe(seq)
                n_valid += smiles is not None
                out.writerow([smiles or "INVALID", seq])
    print(f"{args.num_samples} samples -> {args.output_file}   decodable: {n_valid}/{args.num_samples}")


if __name__ == "__main__":
    main()
