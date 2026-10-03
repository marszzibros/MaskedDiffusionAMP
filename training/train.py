"""
Train the conditional discrete flow matching model on the SAFE peptide corpus, with either tokenizer.

    python -m training.train --tokenizer amide --epochs 100
    python -m training.train --tokenizer xh                    # the shipped BRICS corpus
    python -m training.train --order raw              # the encoder's own label numbering
    python -m training.train --schedule single                 # one linear mask schedule for every token
    python -m training.train --smoke                           # 2 epochs, 4 batches: proves the wiring

"""
import argparse
import datetime
import json
import os

import lightning as L
import numpy as np
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from amp_diffusion import AMPSafeDataModule, DiscreteFlowMatching, ForceSaveCallback

from . import variants


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    variants.add_args(ap)
    ap.add_argument("--schedule", choices=("two", "single"), default="two",
                    help="two: topology tokens masked on a cosine over t in [0, 0.5], chemistry on [0.3, 1.0]. "
                         "single: one linear 1 - t for every token.")
    ap.add_argument("--epochs", type=int, default=None, help="override num_epochs")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--hidden_size", type=int, default=None)
    ap.add_argument("--n_blocks", type=int, default=None)
    ap.add_argument("--n_heads", type=int, default=None)
    ap.add_argument("--num_samples", type=int, default=None,
                    help="molecules sampled at the end of every epoch (0 = none; default 10)")
    ap.add_argument("--save_every", type=int, default=25, help="must be < epochs, or no epoch checkpoints are written")
    ap.add_argument("--limit_train_batches", type=float, default=None, help="Lightning's cap per epoch, for smoke tests")
    ap.add_argument("--attn_backend", choices=("fa2", "sdpa", "auto"), default="fa2",
                    help="fa2 = flash-attn (needs a wheel built for this GPU); sdpa = pure torch; auto = probe and pick")
    ap.add_argument("--wandb", action="store_true", default=True)
    ap.add_argument("--no_wandb", dest="wandb", action="store_false", help="CSV logging only")
    ap.add_argument("--wandb_project", default="AMP_Mask_Diffusion")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, 4 batches, 2 samples")
    ap.add_argument("--out_root", default="output")
    ap.add_argument("--out_dir", default=None, help="the run directory itself, instead of <out_root>/<timestamp>-<tag>")
    return ap.parse_args()


def main():
    args = parse_args()

    tag = f"{args.tokenizer}-{args.schedule}" + ("-raw" if args.order == "raw" else "") + ("-smoke" if args.smoke else "")
    output_dir = args.out_dir or os.path.join(
        args.out_root, f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{tag}")
    os.makedirs(output_dir, exist_ok=True)

    model_config = {
        "model_name": "DiT",
        "batch_size": 8,
        "num_epochs": 51,
        "warmup_ratio": 0.05,
        "num_samples": 10,
        "num_steps": 500,
        "learning_rate": 1e-4,
        "scheduler_name": "cosine",
        "accumulate_grad_batches": 1,
        "max_length": None,         # None = fit the longest molecule in the corpus
        "topo_eta": 10.0,
        "chem_eta": 5.0,
        "base_eta": 5.0,
        "output_dir": output_dir,
        "cond_dropout": 0.1,
        "hidden_size": 768,
        "n_blocks": 12,
        "n_heads": 12,
        "attn_backend": args.attn_backend,
    }
    if args.smoke:
        # 2 epochs: ForceSaveCallback skips epoch 0, so a 1-epoch run would never write a checkpoint
        model_config.update(num_epochs=2, num_samples=2, num_steps=8)
        args.limit_train_batches = args.limit_train_batches or 4
        args.save_every = 1
    if args.epochs is not None:
        model_config["num_epochs"] = args.epochs
    for name in ("batch_size", "hidden_size", "n_blocks", "n_heads", "num_samples"):
        if getattr(args, name) is not None:
            model_config[name] = getattr(args, name)

    # Before the datamodule: the dataset reads the vocabulary path when it is built.
    variant = variants.resolve(args.tokenizer, args.order)
    variants.activate(variant)

    dataset = AMPSafeDataModule(file_path=variants.data_dir(variant, output_dir),
                                max_length=model_config["max_length"], batch_size=model_config["batch_size"])
    dataset.setup()
    model_config["num_tokens"] = dataset.num_tokens
    model_config["mask_token_id"] = dataset.mask_token_id
    model_config["pad_token_id"] = dataset.pad_token_id
    model_config["max_length"] = dataset.max_length
    # "single" hands the model no token classes, so every token is masked on 1 - t
    model_config["topology_token_ids"] = dataset.topology_token_ids if args.schedule == "two" else []
    model_config["chemical_token_ids"] = dataset.chemical_token_ids if args.schedule == "two" else []
    print(f"[Data] {len(dataset.full_dataset)} examples, vocab {dataset.num_tokens}, "
          f"max_length {dataset.max_length}, median length {int(np.median(dataset.length_pool))}")

    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump({**model_config, "tokenizer": args.tokenizer, "order": args.order, "schedule": args.schedule,
                   "vocab_path": variant.vocab_path, "corpus_csv": variant.corpus_csv}, f, indent=4)

    model = DiscreteFlowMatching(**{k: v for k, v in model_config.items() if k != "batch_size"})

    if args.save_every >= model_config["num_epochs"]:
        print(f"[Warn] save_every={args.save_every} >= num_epochs={model_config['num_epochs']}: "
              f"no epoch checkpoints will be saved.")
    trainer_kwargs = {}
    if args.limit_train_batches is not None:
        lim = args.limit_train_batches
        trainer_kwargs["limit_train_batches"] = int(lim) if lim >= 1 else lim

    loggers = [CSVLogger(save_dir=output_dir, name="", version="")]
    if args.wandb:
        loggers.append(WandbLogger(
            project=args.wandb_project,
            name=f"{tag}-{os.path.basename(output_dir)}",
            save_dir=output_dir,
            tags=[args.tokenizer, args.order, args.schedule],
            config={**{k: v for k, v in model_config.items() if k != "output_dir"},
                    "tokenizer": args.tokenizer, "order": args.order, "schedule": args.schedule,
                    "vocab_path": variant.vocab_path},
        ))

    trainer = L.Trainer(
        max_epochs=model_config["num_epochs"],
        logger=loggers,
        callbacks=[ForceSaveCallback(dirpath=output_dir, every_n_epochs=args.save_every),
                   LearningRateMonitor(logging_interval="step")],
        **trainer_kwargs,
    )
    trainer.fit(model, datamodule=dataset)

    # ForceSaveCallback only saves epochs divisible by save_every, so the last epoch needs its own save
    trainer.save_checkpoint(os.path.join(output_dir, "model-final.ckpt"))
    print(f"\n[Done] checkpoints in {output_dir}")


if __name__ == "__main__":
    main()
