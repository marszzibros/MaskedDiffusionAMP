import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import lightning as L
import itertools
import transformers
import matplotlib.pyplot as plt
from utils import Diffusion, LogLinearNoise
from models import DIT, EMA
import os


class MaskedAMPDiffusion(L.LightningModule):
    def __init__(self, 
                 model_name="DiT", 
                 blank_weight=0.00,
                 num_epochs=301,
                 warmup_ratio=0.05,
                 num_samples=5,
                 num_steps=256,
                 learning_rate=2e-5,
                 scheduler_name="linear",
                 num_tokens=49,
                 accumulate_grad_batches=4,
                 max_length=68):

        super().__init__()
        self.blank_weight = blank_weight
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.scheduler_name = scheduler_name
        self.learning_rate = learning_rate
        self.num_tokens = num_tokens
        self.accumulate_grad_batches = accumulate_grad_batches
        self.diffusion = Diffusion(max_length=max_length)
        self.noiser = LogLinearNoise()

        os.system(f"rm -r results")
        os.system(f"mkdir results")

        if model_name == "DiT":
            # TODO: change 40 to real vocab size
            self.model = DIT(vocab_size=self.num_tokens, seq_length=max_length)

        self.ema = EMA(itertools.chain(self.model.parameters(), self.noiser.parameters()),
                               decay = 0.9999)
        self.samples_idx = None
        self.sigma = None
        self.dsigma = None
        self.xt = None

        self.automatic_optimization = False

        

    def forward(self, x, sigma, lengths):
        return self.model(x, sigma, lengths)
    def setup(self, stage=None):
        self.model.to(self.device)
        self.noiser.to(self.device)
    def on_fit_start(self):
        self.ema.move_shadow_params_to_device(self.device)
        # self.num_tokens = len(self.trainer.datamodule.full_dataset.tokens)
     
    def compute_nll_loss(self, output):
        
        output[:, :, self.diffusion.mask_index] = -float("inf") 

        loss = F.cross_entropy(
            output.view(-1, self.num_tokens),
            self.samples_idx.view(-1),
            reduction='none'
        )

        return loss.mean()

    
    def compute_loss(self, weight_tokens=1.5):
    
        with torch.amp.autocast('cuda', dtype=torch.float32):
            output = self.forward(self.xt, torch.zeros_like(self.sigma).to(self.device), self.pad_indices.sum(dim=1))

        nll_loss = self.compute_nll_loss(output)

        total_loss = weight_tokens * nll_loss

        return total_loss, nll_loss
    
    def process_noise(self, samples):
        self.samples_idx = torch.argmax(samples['sequence'], dim=1)
        self.pad_indices = (self.samples_idx != self.diffusion.blank_index)

        t = self.diffusion._sample_t(self.samples_idx.shape[0])
        self.sigma, self.dsigma = self.noiser(t)

        self.xt = self.diffusion.q_xt(self.samples_idx, 1 - torch.exp(-self.sigma[:, None].to(self.device)))

        self.masked = self.xt == self.diffusion.mask_index
        
        sigma_input = 1 - torch.exp(-self.sigma[:, None].to(self.device))



    def training_step(self, batch, batch_idx):

        self.process_noise(batch)
        loss, nll_loss = self.compute_loss()
        
        loss = loss / self.accumulate_grad_batches
        self.manual_backward(loss)

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:

            self.clip_gradients(self.optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            self.optimizer.step()
            
            self.optimizer.zero_grad()

            if self.scheduler_name != "":
                self.scheduler.step()

            self.ema.update(itertools.chain(self.model.parameters(), self.noiser.parameters()))
            
            self.log("train_loss", loss * self.accumulate_grad_batches)  # Rescale for logging
            self.log("nll_loss", nll_loss)

        else:
            self.log("train_loss", loss * self.accumulate_grad_batches)
            self.log("nll_loss", nll_loss)

        return loss

    def on_train_epoch_end(self):
        if self.current_epoch % 50 == 0 and self.current_epoch != 0:
            torch.save({
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "ema_state_dict": self.ema.state_dict(),
                "optimizer_state_dict": self.trainer.optimizers[0].state_dict(),
                "lr_scheduler_state_dict": self.trainer.lr_scheduler_configs[0].scheduler.state_dict(),
                "noiser_state_dict": self.noiser.state_dict()}, f"logs/epoch{self.current_epoch}.pt")
            
        # Apply EMA weights
        self.ema.store(itertools.chain(self.model.parameters(), self.noiser.parameters()))
        self.ema.copy_to(itertools.chain(self.model.parameters(), self.noiser.parameters()))

        sampled_output = self._sample()

        self.ema.restore(itertools.chain(self.model.parameters(), self.noiser.parameters()))

        key = f"train_samples_epoch_{self.current_epoch}"
        columns = ["Length", "Sample"]
        data = [[str(sample["length"]), sample["sequence"]] for sample in sampled_output]
        self.logger.log_text(key=key, columns=columns, data=data)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(itertools.chain(self.model.parameters(), self.noiser.parameters()),
                                lr=self.learning_rate,
                                betas=(0.9,0.999),
                                eps=1e-8,
                                weight_decay=1e-5)
        
        train_loader = self.trainer.datamodule.train_dataloader()
        total_training_steps = int(len(train_loader) / self.accumulate_grad_batches) * self.num_epochs
        num_warmup_steps = int(self.warmup_ratio * total_training_steps)

        if self.scheduler_name == "linear":
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps
            )
            scheduler_config = {"scheduler": self.scheduler, "interval": "step", "frequency": 1}
        elif self.scheduler_name == "cosine":
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps
            )
            scheduler_config = {"scheduler": self.scheduler, "interval": "step", "frequency": 1}
        else:
            scheduler_config = None

        if scheduler_config:
            return [self.optimizer], [scheduler_config]
        else:
            return [self.optimizer]
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()
        checkpoint["optimizer_state"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

    def _sample(self):
        self.model.eval()
        self.noiser.eval()

        eps = 1e-5

        with torch.no_grad():
            x = torch.full((self.num_samples, self.trainer.datamodule.max_length),
                fill_value=self.diffusion.mask_index,
                dtype=torch.int64,
                device=self.device)

            lengths = torch.randint(low=20, high=32, size=(x.shape[0],), device=self.device, dtype=torch.int32)

            timesteps = torch.linspace(1, eps, self.num_steps + 1, device=self.device)
            dt = (1 - eps) / self.num_steps
            p_x0_cache_x = None
            p_x0_cache_s = None
            for i in range(self.num_steps):
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
                sigma_t, _ = self.noiser(t)
                if t.ndim > 1:
                    t = t.squeeze(-1)

                move_chance_t_x = t[:, None, None]
                move_chance_s_x = (t - dt)[:, None, None]

                if p_x0_cache_x is None:
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        # output = self.forward(x, c.to(self.device), lengths, s)
                        output = self.forward(x, torch.zeros_like(sigma_t.squeeze()).to(self.device), lengths)
                    p_x0_cache_x = output.exp()

                q_xs_x = p_x0_cache_x * (move_chance_t_x - move_chance_s_x)


                q_xs_x[:, :, self.diffusion.mask_index] = move_chance_s_x[:, :, 0]
                
                _x = self.diffusion._sample_categorical(q_xs_x)

                copy_flag_x = (x != self.diffusion.mask_index).to(x.dtype)

                x_next = copy_flag_x * x + (1 - copy_flag_x) * _x

                if (not torch.allclose(x_next, x)):
                    p_x0_cache_x = None

                x = x_next

            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            sigma = self.noiser(t)[0].squeeze().to(self.device)
            x = self.model(x,torch.zeros_like(sigma), lengths) # (batch_size, seq_len)
            
            x = x.argmax(dim=-1) 
            
            index_to_token = {i: token for token, i in self.trainer.datamodule.token_dict.items()}

            x = x.detach().cpu().numpy()

            lengths = lengths.detach().cpu().numpy()
            sequences = []
            for i, (seq, length) in enumerate(zip(x, lengths)):
                sequence = ''.join(index_to_token.get(token, '?') for token in seq)
                sequences.append(sequence)

            sample_path = f"logs/generated_samples.txt"
            with open(sample_path, "a") as f:
                f.write(f"=== Epoch {self.current_epoch} ===\n")
                for length, sequence in zip(lengths, sequences):
                    f.write(f"{length}   {sequence}\n")
                f.write("\n")

        self.model.train()
        self.noiser.train()

        return [{"length": int(length), "sequence": sequence} for length, sequence in zip(lengths, sequences)]
    def generate_sample(self, tokens_dict, num_samples=256, max_length=130):

        self.model.eval()
        self.noiser.eval()

        eps = 1e-5

        with torch.no_grad():
            x = torch.full((num_samples, max_length),
                fill_value=self.diffusion.mask_index,
                dtype=torch.int64,
                device=self.device)

            lengths = torch.randint(low=20, high=32, size=(x.shape[0],), device=self.device, dtype=torch.int32)

            timesteps = torch.linspace(1, eps, self.num_steps + 1, device=self.device)
            dt = (1 - eps) / self.num_steps
            p_x0_cache_x = None

            for i in range(self.num_steps):
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
                sigma_t, _ = self.noiser(t)
                if t.ndim > 1:
                    t = t.squeeze(-1)

                move_chance_t_x = t[:, None, None]
                move_chance_s_x = (t - dt)[:, None, None]

                if p_x0_cache_x is None:
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        # output = self.forward(x, c.to(self.device), lengths, s)
                        output = self.forward(x, torch.zeros_like(sigma_t.squeeze()).to(self.device), lengths)
                    p_x0_cache_x = output.exp()

                q_xs_x = p_x0_cache_x * (move_chance_t_x - move_chance_s_x)

                q_xs_x[:, :, self.diffusion.mask_index] = move_chance_s_x[:, :, 0]

                _x = self.diffusion._sample_categorical(q_xs_x)

                copy_flag_x = (x != self.diffusion.mask_index).to(x.dtype)

                x_next = copy_flag_x * x + (1 - copy_flag_x) * _x


                if (not torch.allclose(x_next, x)):
                    p_x0_cache_x = None

                x = x_next

            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            sigma = self.noiser(t)[0].squeeze().to(self.device)
            x = self.model(x,torch.zeros_like(sigma), lengths) # (batch_size, seq_len)
            
            x = x.argmax(dim=-1) 

            index_to_token = {i: token for token, i in tokens_dict.items()}

            x = x.detach().cpu().numpy()
            s = s.detach().cpu().numpy()
            lengths = lengths.detach().cpu().numpy()

            sequences = []
            for i, (seq, length) in enumerate(zip(x, lengths)):
                sequence = ''.join(index_to_token.get(token, '?') for token in seq)
                sequences.append(sequence)

        return sequences, lengths