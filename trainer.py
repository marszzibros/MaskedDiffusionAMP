
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from model import MaskedAMPDiffusion
from dataset import AMPDatasetModule, SwissProtModule

import time
import sys
import os

def main():
    lr = 1e-4

    os.system(f"rm -r logs")
    os.system(f"mkdir logs")
    # dataset = AMPDatasetModule(batch_size=256, pos_ratio=0.5)
    dataset = SwissProtModule(data_path="data/", max_length=66, batch_size=64)
    model = MaskedAMPDiffusion(scheduler_name="cosine", learning_rate=lr, accumulate_grad_batches=1)

    # Initialize the logger
    wandb_logger = WandbLogger(
        project="AMP_Mask_Diffusion",
        save_dir="logs/",
        name=f"DiT_{time.strftime('%Y%m%d_%H%M%S')}_Cosine_{lr}",
        log_model="all",
        offline=False
    )
    trainer = L.Trainer(
            max_epochs=301,
            logger=wandb_logger,
            callbacks=[LearningRateMonitor(logging_interval='step')], 
        )
    trainer.fit(model, datamodule=dataset)

if __name__ == "__main__":
    main()