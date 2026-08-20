"""Fine-tune a pretrained checkpoint on the AMP data.

Pretraining (pretrainer.py) runs on generic UniRef50 peptides with all conditions
dropped. This script picks that checkpoint up and continues on DBAASP with real
species / target-group / MIC conditioning.

Usage:
    python finetune.py --checkpoint output/<pretrain_dir>/model-epoch_18.ckpt
    python finetune.py --from_scratch          # the control run, same settings

"""
import argparse
import datetime
import json
import os
import time

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from DFM import DiscreteFlowMatching
from dataset import AMPSafeDataModule
from trainer import ForceSaveCallback


def describe_checkpoint(path):
    """Read the pretrained hparams without building the model."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters", {})
    print(f"[Checkpoint] {path}")
    for k in ("hidden_size", "n_blocks", "n_heads", "num_tokens", "max_length",
              "cond_dropout", "num_samples", "learning_rate"):
        print(f"    {k:<16} {hp.get(k)}")
    print(f"    ema in ckpt      {'ema_state_dict' in ck}")
    return hp


def main():
    ap = argparse.ArgumentParser(description="Fine-tune a pretrained DFM on AMP data.")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="pretrained .ckpt from pretrainer.py")
    ap.add_argument("--from_scratch", action="store_true",
                    help="control run: same settings, random init, no checkpoint")
    ap.add_argument("--epochs", type=int, default=501)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cond_dropout", type=float, default=0.1)
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--eta", type=float, default=20.0,
                    help="best sampling eta from the sweep (steps=100)")
    ap.add_argument("--accumulate", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=25)
    # only used with --from_scratch; otherwise taken from the checkpoint
    ap.add_argument("--hidden_size", type=int, default=768)
    ap.add_argument("--n_blocks", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=12)
    args = ap.parse_args()

    if not args.checkpoint and not args.from_scratch:
        raise SystemExit("pass --checkpoint <ckpt>, or --from_scratch for the control run")
    if args.checkpoint and not os.path.isfile(args.checkpoint):
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    tag = "scratch" if args.from_scratch else "finetune"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("./", "output", f"{timestamp}-{tag}")
    os.makedirs(output_dir, exist_ok=True)

    dataset = AMPSafeDataModule(
        file_path="molecular_dataset/dataset/data/",
        max_length=None,
        batch_size=args.batch_size)
    dataset.setup()
    print(f"[Data] {len(dataset.full_dataset)} examples, vocab {dataset.num_tokens}, "
          f"max_length {dataset.max_length}, median length "
          f"{int(np.median(dataset.length_pool))}")

    if args.from_scratch:
        hidden, blocks, heads = args.hidden_size, args.n_blocks, args.n_heads
    else:
        hp = describe_checkpoint(args.checkpoint)
        hidden = hp.get("hidden_size", args.hidden_size)
        blocks = hp.get("n_blocks", args.n_blocks)
        heads = hp.get("n_heads", args.n_heads)
        # A vocab mismatch means the tokenizer changed between the two stages;
        # the embedding and output layers would not load.
        if hp.get("num_tokens") not in (None, dataset.num_tokens):
            raise SystemExit(
                f"vocab mismatch: checkpoint has {hp['num_tokens']} tokens, data has "
                f"{dataset.num_tokens}. The tokenizer must be identical across stages.")
        if hp.get("max_length") != dataset.max_length:
            print(f"[Note] max_length differs (pretrain {hp.get('max_length')}, "
                  f"AMP {dataset.max_length}). Harmless -- DDitFinalLayer stores "
                  f"seq_length but builds no parameter from it.")

    model_config = {
        "model_name": "DiT",
        "stage": tag,
        "init_from": args.checkpoint or "random",
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "warmup_ratio": 0.05,
        "num_samples": args.num_samples,
        "num_steps": 100,
        "learning_rate": args.lr,
        "scheduler_name": "cosine",
        "accumulate_grad_batches": args.accumulate,
        "max_length": dataset.max_length,
        "eta": args.eta,
        "output_dir": output_dir,
        "cond_dropout": args.cond_dropout,
        "hidden_size": hidden,
        "n_blocks": blocks,
        "n_heads": heads,
        "num_tokens": dataset.num_tokens,
        "mask_token_id": dataset.mask_token_id,
        "pad_token_id": dataset.pad_token_id,
    }
    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    common = dict(
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

    if args.from_scratch:
        print("[Init] random weights (control run)")
        model = DiscreteFlowMatching(**common)
    else:
        # Overriding hparams here rather than editing them: load_from_checkpoint
        # re-runs __init__ with these values, then loads the weights on top.
        # Optimizer state is deliberately NOT restored -- fine-tuning wants a
        # fresh optimizer and a fresh schedule.
        print(f"[Init] weights from {args.checkpoint}")
        model = DiscreteFlowMatching.load_from_checkpoint(
            args.checkpoint, map_location="cpu", strict=True, **common)

    n = sum(p.numel() for p in model.parameters())
    print(f"[Model] {n/1e6:.1f}M params ({hidden}/{blocks} blocks/{heads} heads), "
          f"{n/len(dataset.full_dataset):,.0f} per molecule")
    print(f"[Config] cond_dropout {args.cond_dropout}, lr {args.lr}, "
          f"{args.epochs} epochs -> {output_dir}")

    wandb_logger = WandbLogger(
        project="AMP_Mask_Diffusion",
        save_dir="logs/",
        name=f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}",
        log_model=False,
        offline=False,
    )
    trainer = L.Trainer(
        max_epochs=model_config["num_epochs"],
        logger=wandb_logger,
        callbacks=[ForceSaveCallback(dirpath=output_dir, every_n_epochs=args.save_every),
                   LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(model, datamodule=dataset)

    print(f"\n[Done] {output_dir}")
    print("Evaluate against the from-scratch control at the same epoch:")
    print(f"  sbatch quick.sh {output_dir}/model-epoch_150.ckpt {tag}_150")
    print("Reference (baseline, epoch 150, steps=100 eta=10): ring_pair 0.713, valid 0.730")


if __name__ == "__main__":
    main()
