"""Pretrain the DiT on generic UniRef50 peptides, then fine-tune with trainer.py.

Same model and objective as trainer.py; three things differ:

  cond_dropout = 1.0   every example uses the learned null_embedding for
                       species/groups/MIC, because UniProt peptides have none.
                       torch.rand() < 1.0 is always True, so this needs no new
                       code path -- see DFM.py:111.
  num_samples  = 0     generation is a validity check against AMP chemistry;
                       it costs 4 forward passes per step and tells you nothing
                       useful about generic peptides. Skip it while pretraining.
  smaller model        hidden 768 / 12 blocks / 12 heads = 100M params, vs
                       492M at 1536/16/12. head_dim 64 is the standard.

Build the corpus first:
    python build_uniprot_corpus.py uniref50.fasta.gz uniprot_safe.csv \
        --target 1000000 --match-lengths molecular_dataset/dataset/data/safe/amp_safe.csv

Then:
    python pretrainer.py                      # full run
    python pretrainer.py --limit 20000        # smoke test first

Fine-tune afterwards by pointing trainer.py at the resulting checkpoint via
--resume, with cond_dropout back to 0.1 and the AMP data module.
"""
import argparse
import datetime
import json
import os
import time

import lightning as L
import numpy as np
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from DFM import DiscreteFlowMatching
from dataset import UniProtSafeDataModule
from trainer import ForceSaveCallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="uniprot_safe.csv",
                    help="output of build_uniprot_corpus.py")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N peptides (smoke tests)")
    ap.add_argument("--epochs", type=int, default=20,
                    help="1M peptides is ~60x the AMP corpus, so far fewer epochs")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden_size", type=int, default=768)
    ap.add_argument("--n_blocks", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="higher than the 1e-4 fine-tuning rate: more data, fewer passes")
    ap.add_argument("--max_length", type=int, default=1301,
                    help="matching the AMP run is not required (no shape-dependent "
                         "parameter depends on it) but keeps the configs comparable")
    ap.add_argument("--save_every", type=int, default=2)
    args = ap.parse_args()

    if args.hidden_size % args.n_heads:
        raise SystemExit(f"n_heads must divide hidden_size "
                         f"({args.hidden_size}/{args.n_heads})")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("./", "output", f"{timestamp}-pretrain")
    os.makedirs(output_dir, exist_ok=True)

    model_config = {
        "model_name": "DiT",
        "stage": "pretrain",
        "corpus": args.corpus,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "warmup_ratio": 0.05,
        "num_samples": 0,          # no generation during pretraining
        "num_steps": 100,
        "learning_rate": args.lr,
        "scheduler_name": "cosine",
        "accumulate_grad_batches": 4,
        "max_length": args.max_length,
        "eta": 5,
        "output_dir": output_dir,
        "cond_dropout": 1.0,       # all conditions null -- UniProt has none
        "hidden_size": args.hidden_size,
        "n_blocks": args.n_blocks,
        "n_heads": args.n_heads,
    }

    dataset = UniProtSafeDataModule(
        csv_path=args.corpus,
        max_length=model_config["max_length"],
        batch_size=model_config["batch_size"],
        limit=args.limit)

    dataset.setup()
    model_config["num_tokens"] = dataset.num_tokens
    model_config["mask_token_id"] = dataset.mask_token_id
    model_config["pad_token_id"] = dataset.pad_token_id
    model_config["max_length"] = dataset.max_length
    print(f"[Data] {len(dataset.full_dataset):,} peptides, vocab {dataset.num_tokens}, "
          f"max_length {dataset.max_length}, median length "
          f"{int(np.median(dataset.length_pool))}")

    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=4)
    print(f"[Config] Saved to: {config_path}")

    model = DiscreteFlowMatching(
        model_name=model_config["model_name"],
        num_epochs=model_config["num_epochs"],
        warmup_ratio=model_config["warmup_ratio"],
        num_samples=model_config["num_samples"],
        num_steps=model_config["num_steps"],
        learning_rate=model_config["learning_rate"],
        scheduler_name=model_config["scheduler_name"],
        num_tokens=model_config["num_tokens"],
        accumulate_grad_batches=model_config["accumulate_grad_batches"],
        max_length=model_config["max_length"],
        mask_token_id=model_config["mask_token_id"],
        pad_token_id=model_config["pad_token_id"],
        eta=model_config["eta"],
        output_dir=model_config["output_dir"],
        cond_dropout=model_config["cond_dropout"],
        hidden_size=model_config["hidden_size"],
        n_blocks=model_config["n_blocks"],
        n_heads=model_config["n_heads"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_mol = len(dataset.full_dataset)
    print(f"[Model] {n_params/1e6:.1f}M parameters, {n_params/max(n_mol,1):,.0f} per peptide")

    force_saver = ForceSaveCallback(dirpath=output_dir, every_n_epochs=args.save_every)

    wandb_logger = WandbLogger(
        project="AMP_Mask_Diffusion",
        save_dir="logs/",
        name=f"pretrain_{time.strftime('%Y%m%d_%H%M%S')}",
        log_model=False,
        offline=False,
    )
    trainer = L.Trainer(
        max_epochs=model_config["num_epochs"],
        logger=wandb_logger,
        callbacks=[force_saver, LearningRateMonitor(logging_interval="step")],
    )

    trainer.fit(model, datamodule=dataset)
    print(f"\n[Done] checkpoints in {output_dir}")
    print(f"Fine-tune with:\n"
          f"  python trainer.py   # after pointing it at a checkpoint in {output_dir}\n"
          f"  and set hidden_size={args.hidden_size}, n_blocks={args.n_blocks}, "
          f"n_heads={args.n_heads}, cond_dropout=0.1")


if __name__ == "__main__":
    main()
