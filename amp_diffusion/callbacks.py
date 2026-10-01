import os

from lightning.pytorch.callbacks import Callback


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

        if epoch % self.every_n_epochs == 0 and epoch != 0:
            filename = f"model-epoch_{epoch:02d}.ckpt"
            save_path = os.path.join(self.dirpath, filename)

            trainer.save_checkpoint(save_path)
