
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback

from DFM import DiscreteFlowMatching
from dataset import AMPSafeDataModule, ARMS
import argparse
import datetime
import time
import sys
import os
import json

import numpy as np

class ForceSaveCallback(Callback):
    """
    Forces a checkpoint save every N epochs.
    - Saves 'model-epoch_XX.ckpt' only every N epochs.
    - Essential for manual_backward in train_step.
    """
    def __init__(self, dirpath, every_n_epochs=1):
        self.dirpath = dirpath
        self.every_n_epochs = every_n_epochs
        os.makedirs(dirpath, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        

        # last_path = os.path.join(self.dirpath, "last.ckpt")
        # trainer.save_checkpoint(last_path)
        
        if epoch % self.every_n_epochs == 0 and epoch != 0:
            filename = f"model-epoch_{epoch:02d}.ckpt"
            save_path = os.path.join(self.dirpath, filename)
            
            trainer.save_checkpoint(save_path)
            
            # if trainer.global_rank == 0:
            #     print(f"\n[Checkpointer] Saved interval checkpoint: {filename}")

def parse_args():
    p = argparse.ArgumentParser(description="Train the AMP masked-diffusion model.")
    # --arm fixes the tokenizer AND the SAFE corpus together (dataset.ARMS);
    # it is the only thing that should differ between sweep runs.
    p.add_argument("--arm", default=None, help=f"one of {sorted(ARMS)}, or omit for the default tokenizer")
    p.add_argument("--tag", default=None, help="output/ subdirectory name (default: timestamp)")
    p.add_argument("--epochs", type=int, default=501)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--accumulate", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden_size", type=int, default=1536)
    p.add_argument("--n_blocks", type=int, default=16)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--num_samples", type=int, default=5, help="per-epoch samples; 0 disables")
    p.add_argument("--seed", type=int, default=0)
    # Sequence length differs several-fold between arms, so denoising steps are
    # scaled to the corpus rather than fixed -- otherwise short arms idle and
    # long arms are under-resolved. Set --num_steps to override outright.
    p.add_argument("--steps_per_token", type=float, default=None)
    p.add_argument("--num_steps", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    run_name = args.tag or timestamp
    output_dir = os.path.join("./", "output", run_name)

    os.makedirs(output_dir, exist_ok=True)

    model_config = {
        "model_name": "DiT",
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "warmup_ratio": 0.05,   # ~25 epochs of warmup at 501 epochs
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "learning_rate": args.lr,
        "scheduler_name": "cosine",
        "accumulate_grad_batches": args.accumulate,   # effective batch 128
        "max_length": None, # None = fit the longest molecule in the corpus (1374 tokens)
        "eta": 5,
        "output_dir": output_dir, # Pass output_dir so model knows where to save generated samples
        "cond_dropout": 0.1,
        # 492M params, ~86 GB peak at batch 16 -- H200 (141 GB) only; this does
        # not fit a 16 GB card. n_heads must divide hidden_size: 1536/12 = 128,
        # which is one of flash-attn's tuned head dimensions (96 is not).
        "hidden_size": args.hidden_size,
        "n_blocks": args.n_blocks,
        "n_heads": args.n_heads,
    }

    model_config["arm"] = args.arm
    dataset = AMPSafeDataModule(
        file_path="molecular_dataset/dataset/data/",
        max_length=model_config['max_length'],
        batch_size=model_config['batch_size'],
        arm=args.arm)

    # Built here (rather than left to Lightning) so the vocab and the special
    # token ids come from the tokenizer instead of being hardcoded.
    dataset.setup()
    model_config['num_tokens'] = dataset.num_tokens
    model_config['mask_token_id'] = dataset.mask_token_id
    model_config['pad_token_id'] = dataset.pad_token_id
    model_config['max_length'] = dataset.max_length
    if args.steps_per_token is not None:
        model_config['num_steps'] = max(1, int(round(args.steps_per_token * dataset.max_length)))
    print(f"[Arm ] {args.arm or 'default'}  ->  steps {model_config['num_steps']}")
    print(f"[Data] {len(dataset.full_dataset)} examples, vocab {dataset.num_tokens}, "
          f"max_length {dataset.max_length}, median length {int(np.median(dataset.length_pool))}")

    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=4)
    print(f"[Config] Saved model hyperparameters to: {config_path}")

    model = DiscreteFlowMatching(
        model_name=model_config['model_name'], 
        num_epochs=model_config['num_epochs'],
        warmup_ratio=model_config['warmup_ratio'],
        num_samples=model_config['num_samples'],
        num_steps=model_config['num_steps'],
        learning_rate=model_config['learning_rate'],
        scheduler_name=model_config['scheduler_name'],
        num_tokens=model_config['num_tokens'],
        accumulate_grad_batches=model_config['accumulate_grad_batches'],
        max_length=model_config['max_length'],
        mask_token_id=model_config['mask_token_id'], 
        pad_token_id=model_config['pad_token_id'],
        eta=model_config['eta'],
        output_dir=model_config['output_dir'],
        cond_dropout=model_config['cond_dropout'],
        hidden_size=model_config['hidden_size'],
        n_blocks=model_config['n_blocks'],
        n_heads=model_config['n_heads'],
        )
    


    # Must be < num_epochs, or the callback never fires and the run produces no
    # checkpoints at all.
    force_saver = ForceSaveCallback(dirpath=output_dir, every_n_epochs=25)
    
    wandb_logger = WandbLogger(
        project="AMP_Mask_Diffusion",
        save_dir="logs/",
        name=f"DiT_{time.strftime('%Y%m%d_%H%M%S')}_DSF",
        log_model=False,
        offline=False
    )
    trainer = L.Trainer(
            max_epochs=model_config['num_epochs'],
            logger=wandb_logger,
            callbacks=[force_saver, LearningRateMonitor(logging_interval='step')], 
        )
        
    trainer.fit(model, datamodule=dataset)

if __name__ == "__main__":
    main()