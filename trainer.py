
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback

from DFM import DiscreteFlowMatching
from dataset import AMPSafeDataModule
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

def main():

    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    run_name = f"{timestamp}"
    output_dir = os.path.join("./", "output", run_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_config = {
        "model_name": "DiT",
        "batch_size": 16,
        "num_epochs": 201,
        "warmup_ratio": 0.05,   # ~25 epochs of warmup at 501 epochs
        "num_samples": 5,
        "num_steps": 1000,
        "learning_rate": 1e-4,
        "scheduler_name": "cosine",
        "accumulate_grad_batches": 8,   # effective batch 128
        "max_length": None, # None = fit the longest molecule in the corpus (1374 tokens)
        "eta": 500,
        "output_dir": output_dir, # Pass output_dir so model knows where to save generated samples
        "cond_dropout": 0.1,
        # 492M params, ~86 GB peak at batch 16 -- H200 (141 GB) only; this does
        # not fit a 16 GB card. n_heads must divide hidden_size: 1536/12 = 128,
        # which is one of flash-attn's tuned head dimensions (96 is not).
        "hidden_size": 1536,
        "n_blocks": 16,
        "n_heads": 12,
    }

    dataset = AMPSafeDataModule(
        file_path="molecular_dataset/dataset/data/",
        max_length=model_config['max_length'],
        batch_size=model_config['batch_size'])

    # Built here (rather than left to Lightning) so the vocab and the special
    # token ids come from the tokenizer instead of being hardcoded.
    dataset.setup()
    model_config['num_tokens'] = dataset.num_tokens
    model_config['mask_token_id'] = dataset.mask_token_id
    model_config['pad_token_id'] = dataset.pad_token_id
    model_config['max_length'] = dataset.max_length
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