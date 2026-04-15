
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback

from model import MaskedAMPDiffusion
from DFM import DiscreteFlowMatching
from dataset import AMPDatasetModule, SwissProtModule
import datetime
import time
import sys
import os
import json

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
    
    # model = MaskedAMPDiffusion(scheduler_name="cosine", learning_rate=lr, accumulate_grad_batches=1)
    
    model_config = {
        "model_name": "DiT",
        "batch_size":64,
        "num_epochs": 1601,
        "warmup_ratio": 0.05,
        "num_samples": 5,
        "num_steps": 100,
        "learning_rate": 1e-4,
        "scheduler_name": "cosine",
        "num_tokens": 49,
        "accumulate_grad_batches": 1,
        "max_length": 68,
        "mask_token_id": 48, # Crucial: Must pass the actual ID for <mask>
        "pad_token_id": 0,   # Crucial: Must pass the actual ID for <blank>
        "eta": 5,
        "output_dir": output_dir, # Pass output_dir so model knows where to save generated samples
        # Add conditioning params if using the CFG version:
        # "num_mechanisms": 10,
        # "cond_dropout": 0.1
    }

    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=4)
    print(f"[Config] Saved model hyperparameters to: {config_path}")
    
    dataset = AMPDatasetModule(
        file_path="data/", 
        max_length=model_config['max_length'], 
        batch_size=model_config['batch_size'], 
        pos_ratio=1.0)
    
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
        output_dir=model_config['output_dir']
        )
    


    force_saver = ForceSaveCallback(dirpath=output_dir, every_n_epochs=800)
    
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