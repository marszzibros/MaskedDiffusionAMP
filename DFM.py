import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
import transformers
from models import DIT, EMA
import os

class DiscreteFlowMatching(L.LightningModule):
    def __init__(self, 
                 model_name="DiT", 
                 num_epochs=301,
                 warmup_ratio=0.05,
                 num_samples=5,
                 num_steps=256,
                 learning_rate=2e-5,
                 scheduler_name="linear",
                 num_tokens=49,
                 accumulate_grad_batches=4,
                 max_length=68,
                 mask_token_id=None, 
                 pad_token_id=None, 
                 eta=0.0,
                 output_dir=None): 

        super().__init__()
        self.save_hyperparameters()
        
        if mask_token_id is None:
            raise ValueError("You must provide the mask_token_id (integer) from your vocabulary.")
            
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id 
        self.eta = eta 

        if model_name == "DiT":
            # Note: Paper suggests model output should be S-1.
            # We handle this by masking logits rather than changing architecture dimensions.
            self.model = DIT(vocab_size=num_tokens, seq_length=max_length)

        self.ema = EMA(self.model.parameters(), decay=0.9999)
        self.automatic_optimization = False
        
        # if self.global_rank == 0:
        #     os.makedirs("results", exist_ok=True)
        #     os.makedirs("logs", exist_ok=True)
        
    def on_save_checkpoint(self, checkpoint):
        """
        Manually save the EMA state into the checkpoint dictionary.
        This ensures that when you call trainer.save_checkpoint(), 
        the EMA weights go with it.
        """
        if self.ema is not None:
            # Assuming your EMA class has a state_dict() method
            # If it doesn't, you need to save its internal shadow params list
            checkpoint['ema_state_dict'] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        """
        Manually load the EMA state from the checkpoint dictionary.
        """
        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
            # Ensure EMA device matches model device after loading
            self.ema.move_shadow_params_to_device(self.device)

    def forward(self, x, t, lengths):
        return self.model(x, t, lengths)

    def on_fit_start(self):
        self.ema.move_shadow_params_to_device(self.device)

    def training_step(self, batch, batch_idx):
        if batch['sequence'].ndim == 3:
            x_1 = torch.argmax(batch['sequence'], dim=1)
        else:
            x_1 = batch['sequence']
            
        # TODO: Add condition
        print(batch['condition'][0])
            
        batch_size = x_1.shape[0]
        seq_len = x_1.shape[1]
        
        lengths = (x_1 != self.hparams.pad_token_id).sum(dim=1)

        # t ~ U[0, 1]
        t = torch.rand(batch_size, device=self.device)

        # Masking Interpolant
        mask_prob = 1.0 - t.unsqueeze(-1)
        
        # Determine which tokens to mask
        random_mask = torch.rand(batch_size, seq_len, device=self.device) < mask_prob
        
        if self.pad_token_id is not None:
            is_not_padding = (x_1 != self.pad_token_id)
            mask_mask = random_mask & is_not_padding
        else:
            mask_mask = random_mask
        
        # Create x_t: corrupted sequence
        x_t = torch.where(mask_mask, 
                          torch.tensor(self.mask_token_id, device=self.device), 
                          x_1)

        # Predict x_1
        logits = self.forward(x_t, t, lengths)


        logits[:, :, self.mask_token_id] = -float('inf')
        if self.pad_token_id is not None:
            logits[:, :, self.pad_token_id] = -float('inf')


        ignore_idx = self.pad_token_id
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_idx)
        
        nll = loss_fct(logits.transpose(1, 2), x_1)
        
        # Calculate loss only on MASKED tokens
        active_loss_mask = mask_mask.float()
        num_masked = active_loss_mask.sum()
        
        if num_masked > 0:
            # Standard optimization
            loss = (nll * active_loss_mask).sum() / num_masked
        else:
            loss = nll.sum() * 0.0 

        loss = loss / self.hparams.accumulate_grad_batches
        self.manual_backward(loss)

        if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
            # Gradient clipping is important for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
            
            self.ema.update(self.model.parameters())
            self.log("train_loss", loss * self.hparams.accumulate_grad_batches, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self):
        if self.global_rank == 0:
            if hasattr(self.trainer.datamodule, 'token_dict'):
                tokens_dict = self.trainer.datamodule.token_dict
            else:
                print("Warning: 'token_dict' not found in DataModule. Skipping generation.")
                return

            sequences = self.generate_sample(
                tokens_dict=tokens_dict,
                num_samples=self.hparams.num_samples,
                max_length=self.hparams.max_length
            )

            path = f"{self.hparams.output_dir}/generated_samples.txt"
            with open(path, "a") as f:
                f.write(f"\n=== Epoch {self.current_epoch} ===\n")
                for seq in sequences:
                    f.write(f"{seq}\n")

    @torch.no_grad()
    def generate_sample(self, tokens_dict, num_samples=5, max_length=68, eta=None):
        if eta is None:
            eta = self.eta

        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())
        
        self.model.eval()
        device = self.device
        
        try:
            x = torch.full((num_samples, max_length), 
                           self.mask_token_id, 
                           dtype=torch.long, 
                           device=device)
            
            lengths = torch.randint(low=20, high=32, size=(num_samples,), device=device, dtype=torch.int32)
            
            t = 0.0
            steps = self.hparams.num_steps
            dt = 1.0 / steps
            
            index_to_token = {i: token for token, i in tokens_dict.items()}
            
            for _ in range(steps):
                t_tensor = torch.full((num_samples,), t, device=device)
                
                # Passed lengths to forward
                logits = self.model(x, t_tensor, lengths)
                
                logits = logits.to(torch.float32)
                
                logits[:, :, self.mask_token_id] = -float('inf')
                if self.pad_token_id is not None:
                    logits[:, :, self.pad_token_id] = -float('inf')
                
                x1_probs = F.softmax(logits, dim=-1)
                x1_sample = Categorical(x1_probs).sample()
                
                # Unmasking rate 
                unmask_rate = dt * (1 + eta * t) / (1 - t + 1e-6)
                unmask_rate = min(unmask_rate, 1.0)
                
                should_unmask = torch.rand_like(x.float()) < unmask_rate
                
                is_masked = (x == self.mask_token_id)
                
                # Only unmask currently masked positions [cite: 1237]
                x = torch.where(is_masked & should_unmask, x1_sample, x)
                
                # CTMC Stochasticity (Eq 34/Listing 3) [cite: 1303]
                if eta > 0 and (t + dt < 1.0):
                    remask_rate = dt * eta
                    remask_rate = min(remask_rate, 1.0)
                    
                    should_remask = torch.rand_like(x.float()) < remask_rate
                    
                    # --- INTEGRATION: Padding Protection ---
                    # We must only re-mask tokens that are revealed AND NOT PADDING.
                    is_revealed = (x != self.mask_token_id)
                    
                    if self.pad_token_id is not None:
                        is_not_padding = (x != self.pad_token_id)
                        is_revealed = is_revealed & is_not_padding
                    
                    # Apply re-masking
                    x = torch.where(is_revealed & should_remask, 
                                    torch.tensor(self.mask_token_id, device=device), 
                                    x)
                
                t += dt

            sequences = []
            x_np = x.cpu().numpy()
            lens_np = lengths.cpu().numpy() 
            
            for seq, length in zip(x_np, lens_np):
                valid_seq = seq[:length] 
                seq_str = ''.join(index_to_token.get(idx, '?') for idx in valid_seq)
                sequences.append(seq_str)
                
            return sequences

        finally:
            self.ema.restore(self.model.parameters())
            self.model.train()

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.hparams.learning_rate,
                                           weight_decay=1e-5)
        
        if self.trainer.max_epochs is not None:
             total_steps = (len(self.trainer.datamodule.train_dataloader()) // self.hparams.accumulate_grad_batches) * self.trainer.max_epochs
        else:
             total_steps = self.trainer.max_steps

        warmup_steps = int(self.hparams.warmup_ratio * total_steps)
        
        if self.hparams.scheduler_name == "linear":
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        elif self.hparams.scheduler_name == "cosine":
             self.scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        else:
            self.scheduler = None
            
        return [self.optimizer]