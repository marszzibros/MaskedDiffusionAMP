
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
        
        if (epoch + 1) % self.every_n_epochs == 0:
            filename = f"model-epoch_{epoch:02d}.ckpt"
            save_path = os.path.join(self.dirpath, filename)
            
            trainer.save_checkpoint(save_path)
            
            if trainer.global_rank == 0:
                print(f"\n[Checkpointer] Saved interval checkpoint: {filename}")

def main():

    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    run_name = f"{timestamp}"
    output_dir = os.path.join("./", "output", run_name)
    
    os.makedirs(output_dir, exist_ok=True)
    dataset = AMPDatasetModule(file_path="data/", max_length=68, batch_size=64, pos_ratio=1.0)
    
    # model = MaskedAMPDiffusion(scheduler_name="cosine", learning_rate=lr, accumulate_grad_batches=1)
    
    model = DiscreteFlowMatching(
        model_name="DiT",
        num_epochs=601,
        warmup_ratio=0.05,
        num_samples=5,
        num_steps=20,
        learning_rate=1e-5,
        scheduler_name="cosine",
        num_tokens=49,
        accumulate_grad_batches=1,
        max_length=68,
        mask_token_id=48, # Crucial: Must pass the actual ID for <mask>
        pad_token_id=0, # Crucial: Must pass the actual ID for <blank>
        eta=0.1,
        output_dir=output_dir
    )
    


    force_saver = ForceSaveCallback(dirpath=output_dir, every_n_epochs=10)
    
    wandb_logger = WandbLogger(
        project="AMP_Mask_Diffusion",
        save_dir="logs/",
        name=f"DiT_{time.strftime('%Y%m%d_%H%M%S')}_DSF",
        log_model=False,
        offline=False
    )
    trainer = L.Trainer(
            max_epochs=601,
            logger=wandb_logger,
            callbacks=[force_saver, LearningRateMonitor(logging_interval='step')], 
        )
        
    trainer.fit(model, datamodule=dataset)

if __name__ == "__main__":
    main()
