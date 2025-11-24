import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

import lightning as L

from dataset import AMPDatasets, NonAMPDatasets, BatchSampler 

import itertools
import transformers

import matplotlib.pyplot as plt

from utils import Diffusion, LogLinearNoise
from models import DIT, EMA

import os
import numpy as np

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
                 num_tokens=24,
                 num_categoricals=25,
                 structure=False,
                 accumulate_grad_batches=4,
                 max_length=66,
                 fusion="sigmoid"):

        super().__init__()
        self.blank_weight = blank_weight
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.scheduler_name = scheduler_name
        self.learning_rate = learning_rate
        self.structure = structure
        self.num_categoricals = num_categoricals
        self.num_tokens = num_tokens
        self.accumulate_grad_batches = accumulate_grad_batches
        self.fusion = fusion
        self.diffusion = Diffusion(max_length=max_length, categorical_bin=num_categoricals - 5)
        self.noiser = LogLinearNoise()

        os.system(f"rm -r results_{self.fusion}_{self.num_categoricals - 5}")
        os.system(f"mkdir results_{self.fusion}_{self.num_categoricals - 5}")

        if model_name == "DiT":
            # TODO: change 40 to real vocab size
            self.model = DIT(vocab_size=self.num_tokens, fusion=self.fusion, num_categoricals=self.num_categoricals)

        self.ema = EMA(itertools.chain(self.model.parameters(), self.noiser.parameters()),
                               decay = 0.9999)
        self.samples_idx = None
        self.sigma = None
        self.dsigma = None
        self.xt = None

        self.automatic_optimization = False

        

    def forward(self, x, sigma, lengths, distance_map=None):
        return self.model(x, sigma, lengths, distance_map)
    def setup(self, stage=None):
        self.model.to(self.device)
        self.noiser.to(self.device)
    def on_fit_start(self):
        self.ema.move_shadow_params_to_device(self.device)
        # self.num_tokens = len(self.trainer.datamodule.full_dataset.tokens)
    def extract_lower_triangle(self, dist):
        if len(dist.shape) == 4:
            # dist: (BS, L, L, 25)
            BS, L, _, C = dist.shape  # C = 25
            tril_idx = torch.tril_indices(L, L, offset=-1) 
            
            lower_tri = dist[:, tril_idx[0], tril_idx[1], :] 

        elif len(dist.shape) == 3:
            # dist: (BS, L, L)
            BS, L, _ = dist.shape  # C = 25
            tril_idx = torch.tril_indices(L, L, offset=-1) 
            
            lower_tri = dist[:, tril_idx[0], tril_idx[1]] 
            
        return lower_tri            
    def compute_nll_loss(self, output, strc=False):
        if not strc:
            output[:, :, self.diffusion.mask_index] = -float("inf") 

            loss = F.cross_entropy(
                output.view(-1, self.num_tokens),
                self.samples_idx.view(-1),
                reduction='none'
            )


            mask = (
                (self.xt.view(-1) == self.diffusion.mask_index) |
                (self.samples_idx.view(-1) == self.diffusion.blank_index)
            )

        else:
            lower_output = self.extract_lower_triangle(output)
            lower_output[:, :, self.diffusion.distance_mask_index] = -float("inf")
            lower_target = self.extract_lower_triangle(self.distance_map)
            lower_masked = self.extract_lower_triangle(self.xt_distance_map)

            loss = F.cross_entropy(
                lower_output.view(-1, self.num_categoricals),
                lower_target.view(-1),
                reduction='none'
            )

            mask = (
                (lower_masked.view(-1) == self.diffusion.distance_mask_index) |
                (lower_target.view(-1) == self.diffusion.distance_blank_index)
            )


        lengths = self.pad_indices.sum(dim=1)
        target_len = 30.0
        sigma = 25
        weights = torch.exp(-0.5 * ((lengths.float() - target_len) / sigma) ** 2)
        seq_weights = weights[:, None].expand_as(self.pad_indices)
        flat_weights = seq_weights.reshape(-1)

        if strc:
            seq_weights = weights[:, None, None].expand_as(self.distance_map)
            lower_seq_weights = self.extract_lower_triangle(seq_weights)
            flat_weights = lower_seq_weights.reshape(-1)

        weighted_loss = loss[mask] * flat_weights[mask]
        if weighted_loss.numel() == 0:
            return torch.tensor(0.0, device=output.device)

        return weighted_loss.sum() / flat_weights[mask].sum()


        # masked_loss = loss[mask]

        # if masked_loss.numel() == 0:
        #     return torch.tensor(0.0, device=output.device)

        # return masked_loss.mean()
    
    def compute_loss(self, weight_tokens=1.5, weight_structure=0.5):
    
        with torch.amp.autocast('cuda', dtype=torch.float32):
            output, maps = self.forward(self.xt, torch.zeros_like(self.sigma).to(self.device), self.pad_indices.sum(dim=1), self.xt_distance_map)

        nll_loss = self.compute_nll_loss(output)
        nll_loss_structure = self.compute_nll_loss(maps, strc=True)

        total_loss = weight_tokens * nll_loss + weight_structure * nll_loss_structure

        return total_loss, nll_loss, nll_loss_structure
    
    def process_noise(self, samples):
        self.samples_idx = torch.argmax(samples['sequence'], dim=1)
        self.pad_indices = (self.samples_idx != self.diffusion.blank_index)

        t = self.diffusion._sample_t(self.samples_idx.shape[0])
        self.sigma, self.dsigma = self.noiser(t)

        self.xt = self.diffusion.q_xt(self.samples_idx, 1 - torch.exp(-self.sigma[:, None].to(self.device)), strc=False)

        self.masked = self.xt == self.diffusion.mask_index
        self.distance_map = samples['map']
        BS, L, _ = self.distance_map.shape

        flat_input = self.distance_map.view(BS, -1)
        sigma_input = 1 - torch.exp(-self.sigma[:, None].to(self.device))

        xt_flat = self.diffusion.q_xt(flat_input, sigma_input, strc=True)
        
        xt_distance_map = xt_flat.view(BS, L, L).long()

        # Zero out diagonal
        xt_distance_map = xt_distance_map * (1 - torch.eye(L, device=xt_distance_map.device).long())

        # Take the lower triangle (excluding diagonal)
        lower = torch.tril(xt_distance_map, diagonal=-1)

        # Construct symmetric matrix by adding its transpose
        self.xt_distance_map = lower + lower.transpose(1, 2)

        # distance_map = distance_map.masked_fill(self.masked.unsqueeze(2), self.diffusion.distance_mask_index)
        # distance_map = distance_map.masked_fill(self.masked.unsqueeze(1), self.diffusion.distance_mask_index)
        

    def training_step(self, batch, batch_idx):

        self.process_noise(batch)
        loss, nll_loss, nll_loss_structure = self.compute_loss()
        
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
            self.log("nll_loss (structure)", nll_loss_structure)
        else:
            self.log("train_loss", loss * self.accumulate_grad_batches)
            self.log("nll_loss", nll_loss)
            self.log("nll_loss (structure)", nll_loss_structure)

        return loss

    def on_train_epoch_end(self):
        if self.current_epoch % 50 == 0 and self.current_epoch != 0:
            torch.save({
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "ema_state_dict": self.ema.state_dict(),
                "optimizer_state_dict": self.trainer.optimizers[0].state_dict(),
                "lr_scheduler_state_dict": self.trainer.lr_scheduler_configs[0].scheduler.state_dict(),
                "noiser_state_dict": self.noiser.state_dict()}, f"logs/{self.fusion}_{self.num_categoricals - 5}/epoch{self.current_epoch}.pt")
            
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
            s = torch.full((self.num_samples, self.trainer.datamodule.max_length, self.trainer.datamodule.max_length),
                fill_value=self.diffusion.distance_mask_index,
                dtype=torch.int64,
                device=self.device)
            # c = torch.tensor([3,1,1,0,0,0,3,3,2,2,2,3,2,3,1,1,3]).to(device)
            # c = c.unsqueeze(0).repeat(5, 1)
            s = s * (1 - torch.eye(max_length, device=s.device).long())
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

                move_chance_t_s = t[:, None, None, None]
                move_chance_s_s = (t - dt)[:, None, None, None]

                if p_x0_cache_x is None:
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        # output = self.forward(x, c.to(self.device), lengths, s)
                        output = self.forward(x, torch.zeros_like(sigma_t.squeeze()).to(self.device), lengths, s)
                    p_x0_cache_x = output[0].exp()
                    p_x0_cache_s = output[1].exp()
                q_xs_x = p_x0_cache_x * (move_chance_t_x - move_chance_s_x)
                q_xs_s = p_x0_cache_s * (move_chance_t_s - move_chance_s_s)

                q_xs_x[:, :, self.diffusion.mask_index] = move_chance_s_x[:, :, 0]
                q_xs_s[:, :, :, self.diffusion.distance_mask_index] = move_chance_s_s[:, :, :, 0]
                
                _x = self.diffusion._sample_categorical(q_xs_x)
                _s = self.diffusion._sample_categorical(q_xs_s)

                copy_flag_x = (x != self.diffusion.mask_index).to(x.dtype)
                copy_flag_s = (s != self.diffusion.distance_mask_index).to(s.dtype)

                x_next = copy_flag_x * x + (1 - copy_flag_x) * _x
                s_next = copy_flag_s * s + (1 - copy_flag_s) * _s  

                if (not torch.allclose(x_next, x)):
                    p_x0_cache_x = None
                s = s_next
                s = s * (1 - torch.eye(self.trainer.datamodule.max_length, device=s.device).long())
                lower = torch.tril(s, diagonal=-1)
                s = lower + lower.transpose(1, 2)
                x = x_next

            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            sigma = self.noiser(t)[0].squeeze().to(self.device)
            x, s = self.model(x,torch.zeros_like(sigma), lengths, s) # (batch_size, seq_len)
            

            
            x = x.argmax(dim=-1) 
            s = s.argmax(dim=-1)

            s = s * (1 - torch.eye(self.trainer.datamodule.max_length, device=s.device).long())
            lower = torch.tril(s, diagonal=-1)
            s = lower + lower.transpose(1, 2)
            
            index_to_token = {i: token for token, i in self.trainer.datamodule.full_dataset.tokens_dict.items()}

            x = x.detach().cpu().numpy()
            s = s.detach().cpu().numpy()
            lengths = lengths.detach().cpu().numpy()

            

            sequences = []
            for i, (seq, dis, length) in enumerate(zip(x, s, lengths)):
                sequence = ''.join(index_to_token.get(token, '?') for token in seq)
                sequences.append(sequence)

                os.makedirs(f"results_{self.fusion}_{self.num_categoricals - 5}/{self.current_epoch}", exist_ok=True)
                plt.figure(figsize=(6, 5))

                # Show the full matrix using viridis
                plt.imshow(dis[:length, :length], cmap="viridis", vmin=0, vmax=self.num_categoricals - 1)

                # Overlay red squares for value == 21
                highlight_indices = np.argwhere(dis == self.num_categoricals - 4)
                for y, x in highlight_indices:
                    if x < length and y < length:
                        plt.gca().add_patch(
                            plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                        edgecolor='red', facecolor='none', linewidth=2)
                        )

                plt.colorbar()

                # Save and close
                save_path = f"results_{self.fusion}_{self.num_categoricals - 5}/{self.current_epoch}/{i}.png"
                plt.savefig(save_path)
                plt.close()


            sample_path = f"logs/generated_samples_{self.fusion}_{self.num_categoricals - 5}.txt"
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
            s = torch.full((num_samples, max_length, max_length),
                fill_value=self.diffusion.distance_mask_index,
                dtype=torch.int64,
                device=self.device)
            # Zero out diagonal
            s = s * (1 - torch.eye(max_length, device=s.device).long())

            # c = torch.tensor([3,1,1,0,0,0,3,3,2,2,2,3,2,3,1,1,3]).to(device)
            # c = c.unsqueeze(0).repeat(5, 1)
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

                move_chance_t_s = t[:, None, None, None]
                move_chance_s_s = (t - dt)[:, None, None, None]

                if p_x0_cache_x is None:
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        # output = self.forward(x, c.to(self.device), lengths, s)
                        output = self.forward(x, torch.zeros_like(sigma_t.squeeze()).to(self.device), lengths, s)
                    p_x0_cache_x = output[0].exp()
                    p_x0_cache_s = output[1].exp()
                q_xs_x = p_x0_cache_x * (move_chance_t_x - move_chance_s_x)
                q_xs_s = p_x0_cache_s * (move_chance_t_s - move_chance_s_s)

                q_xs_x[:, :, self.diffusion.mask_index] = move_chance_s_x[:, :, 0]
                q_xs_s[:, :, :, self.diffusion.distance_mask_index] = move_chance_s_s[:, :, :, 0]
                
                _x = self.diffusion._sample_categorical(q_xs_x)
                _s = self.diffusion._sample_categorical(q_xs_s)

                copy_flag_x = (x != self.diffusion.mask_index).to(x.dtype)
                copy_flag_s = (s != self.diffusion.distance_mask_index).to(s.dtype)

                x_next = copy_flag_x * x + (1 - copy_flag_x) * _x
                s_next = copy_flag_s * s + (1 - copy_flag_s) * _s  

                if (not torch.allclose(x_next, x)):
                    p_x0_cache_x = None
                s = s_next
                s = s * (1 - torch.eye(max_length, device=s.device).long())
                lower = torch.tril(s, diagonal=-1)
                s = lower + lower.transpose(1, 2)
                x = x_next

            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            sigma = self.noiser(t)[0].squeeze().to(self.device)
            x, s = self.model(x,torch.zeros_like(sigma), lengths, s) # (batch_size, seq_len)
            
            x = x.argmax(dim=-1) 
            s = s.argmax(dim=-1)

            s = s * (1 - torch.eye(max_length, device=s.device).long())
            lower = torch.tril(s, diagonal=-1)
            s = lower + lower.transpose(1, 2)
            
            index_to_token = {i: token for token, i in tokens_dict.items()}

            x = x.detach().cpu().numpy()
            s = s.detach().cpu().numpy()
            lengths = lengths.detach().cpu().numpy()

            sequences = []
            for i, (seq, dis, length) in enumerate(zip(x, s, lengths)):
                sequence = ''.join(index_to_token.get(token, '?') for token in seq)
                sequences.append(sequence)

        return sequences, s, lengths

