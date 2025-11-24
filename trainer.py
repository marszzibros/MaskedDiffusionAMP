
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
    fusion = sys.argv[1]
    categorical_bin = int(sys.argv[2])
    os.system(f"rm -r logs/{fusion}_{categorical_bin}")
    os.system(f"mkdir logs/{fusion}_{categorical_bin}")
    # dataset = AMPDatasetModule(batch_size=256, pos_ratio=0.5)
    dataset = SwissProtModule(data_path="data/", max_length=66, batch_size=512, categorical_bin=categorical_bin)
    model = MaskedAMPDiffusion(scheduler_name="cosine", learning_rate=lr, structure=True, accumulate_grad_batches=2, fusion=fusion, num_categoricals=categorical_bin + 5)

    # Initialize the logger
    wandb_logger = WandbLogger(
        project="AMP_Mask_Diffusion",
        save_dir="logs/",
        name=f"DiT_{time.strftime('%Y%m%d_%H%M%S')}_Cosine_{lr}_{fusion}_{categorical_bin}",
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