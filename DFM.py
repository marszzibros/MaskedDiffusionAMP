import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
import transformers
from models import EMA
from models.DiTwithCondition import DIT
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
                 max_length=None,
                 mask_token_id=None, 
                 pad_token_id=None, 
                 eta=0.0,
                 output_dir=None,
                 cond_dropout=0.1,
                 species_dim=6,
                 groups_dim=5,
                 mic_dim=10,
                 hidden_size=1536,
                 n_blocks=24,
                 n_heads=12):

        super().__init__()
        self.save_hyperparameters()
        
        if mask_token_id is None:
            raise ValueError("You must provide the mask_token_id (integer) from your vocabulary.")

        if max_length is None:
            raise ValueError("You must provide max_length; take it from the datamodule "
                             "after setup() rather than hardcoding it.")
            
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id 
        self.eta = eta 
        self.cond_dropout = cond_dropout
        
        # Store dimensions for generation helper
        self.dims = {'species': species_dim, 'groups': groups_dim, 'mic': mic_dim}

        if model_name == "DiT":
            # Initialize DIT with specific vector dimensions
            self.model = DIT(vocab_size=num_tokens,
                             seq_length=max_length,
                             species_dim=species_dim,
                             groups_dim=groups_dim,
                             mic_dim=mic_dim,
                             hidden_size=hidden_size,
                             n_blocks=n_blocks,
                             n_heads=n_heads)

        self.ema = EMA(self.model.parameters(), decay=0.9999)
        self.automatic_optimization = False
        
    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
            self.ema.move_shadow_params_to_device(self.device)

    def forward(self, x, t, lengths, **kwargs):
        # Forward pass now accepts **kwargs for vectors and masks
        return self.model(x, t, lengths, **kwargs)

    def on_fit_start(self):
        self.ema.move_shadow_params_to_device(self.device)

    def training_step(self, batch, batch_idx):
        if batch['sequence'].ndim == 3:
            x_1 = torch.argmax(batch['sequence'], dim=1)
        else:
            x_1 = batch['sequence']
            
        
        # "first 6 is species, next 5 is group, ignore next 5, last 10 is mic"
        cond_tensor = batch['condition'].float() # Ensure float for embeddings
        
        species_vec = cond_tensor[:, :6]
        groups_vec  = cond_tensor[:, 6:11]
        # Skip 11:16 because they are objects / unused
        mic_vec     = cond_tensor[:, 16:26]
            
        batch_size = x_1.shape[0]
        seq_len = x_1.shape[1]
        
        lengths = (x_1 != self.hparams.pad_token_id).sum(dim=1)

        # --- GENERATE CFG DROPOUT MASKS ---
        # True = Drop Condition (Use Null Embedding)
        # False = Keep Condition (Use Data Vector)
        drop_species = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        drop_groups  = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        drop_mic     = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        if self.cond_dropout > 0:
            # Split batch into 4 portions: [0, q1), [q1, q2), [q2, q3), [q3, batch_size)
            q1 = batch_size // 4
            q2 = batch_size // 2
            q3 = (3 * batch_size) // 4

            # Portion 1: Drop everything
            drop_species[:q1] = True
            drop_groups[:q1]  = True
            drop_mic[:q1]     = True

            # Portion 2: Only keep species (drop groups & mic)
            drop_groups[q1:q2] = True
            drop_mic[q1:q2]    = True

            # Portion 3: Only keep groups (drop species & mic)
            drop_species[q2:q3] = True
            drop_mic[q2:q3]     = True

            # Portion 4: Only keep mic (drop species & groups)
            drop_species[q3:] = True
            drop_groups[q3:]  = True

        t = torch.rand(batch_size, device=self.device)
        mask_prob = 1.0 - t.unsqueeze(-1)
        random_mask = torch.rand(batch_size, seq_len, device=self.device) < mask_prob
        
        if self.pad_token_id is not None:
            is_not_padding = (x_1 != self.pad_token_id)
            mask_mask = random_mask & is_not_padding
        else:
            mask_mask = random_mask
        
        x_t = torch.where(mask_mask, 
                          torch.tensor(self.mask_token_id, device=self.device), 
                          x_1)

        # --- PASS VECTORS AND MASKS TO MODEL ---
        logits = self.forward(x_t, t, lengths,
                              species_vec=species_vec,
                              species_mask=drop_species,
                              groups_vec=groups_vec,
                              groups_mask=drop_groups,
                              mic_vec=mic_vec,
                              mic_mask=drop_mic)

        logits[:, :, self.mask_token_id] = -float('inf')
        if self.pad_token_id is not None:
            logits[:, :, self.pad_token_id] = -float('inf')

        ignore_idx = self.pad_token_id if self.pad_token_id is not None else -100
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_idx)
        nll = loss_fct(logits.transpose(1, 2), x_1)
        
        active_loss_mask = mask_mask.float()
        num_masked = active_loss_mask.sum()
        
        if num_masked > 0:
            loss = (nll * active_loss_mask).sum() / num_masked
        else:
            loss = nll.sum() * 0.0 

        loss = loss / self.hparams.accumulate_grad_batches
        self.manual_backward(loss)

        if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
            
            self.ema.update(self.model.parameters())
            self.log("train_loss", loss * self.hparams.accumulate_grad_batches, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self):
        # num_samples=0 disables per-epoch generation. Used during pretraining,
        # where sampling costs 4 forward passes per step and the validity signal
        # is about AMP chemistry, not generic peptides.
        if self.hparams.num_samples <= 0:
            return
        if self.global_rank == 0:
            if hasattr(self.trainer.datamodule, 'token_dict'):
                tokens_dict = self.trainer.datamodule.token_dict
            else:mask_drop
                print("Warning: 'token_dict' not found in DataModule. Skipping generation.")
                return

            # Example conditions for validation generation
            example_cond = {'species': [1], 'groups': [2], 'mic': 7}
            example_scales = {'species': 1.5, 'groups': 1.5, 'mic': 3.0}

            datamodule = self.trainer.datamodule
            sequences = self.generate_sample(
                tokens_dict=tokens_dict,
                conditions=example_cond,
                scales=example_scales,
                num_samples=self.hparams.num_samples,
                max_length=self.hparams.max_length,
                length_pool=getattr(datamodule, 'length_pool', None),
                decode_fn=getattr(datamodule, 'decode', None),
            )

            # For SAFE, log whether each sample is a chemically valid molecule --
            # a token string can be well-formed and still not decode.
            to_smiles = getattr(datamodule, 'smiles_from_safe', None)
            path = f"{self.hparams.output_dir}/generated_samples.txt"
            with open(path, "a") as f:
                f.write(f"\n=== Epoch {self.current_epoch} ===\n")
                valid = 0
                for seq in sequences:
                    smiles = to_smiles(seq) if to_smiles is not None else None
                    valid += smiles is not None
                    f.write(f"{seq}\n")
                    if to_smiles is not None:
                        f.write(f"  -> {smiles if smiles else 'INVALID'}\n")
                if to_smiles is not None:
                    f.write(f"valid: {valid}/{len(sequences)}\n")
                    self.log("sample_validity", valid / max(len(sequences), 1), prog_bar=True)

    def _decode_to_string(self, x_np, lens_np, index_to_token):
        sequences = []
        for seq, length in zip(x_np, lens_np):
            valid_seq = seq[:length] 
            seq_str = ''.join(index_to_token.get(idx, '?') for idx in valid_seq)
            sequences.append(seq_str)
        return sequences

    def _prepare_vector(self, indices_or_vec, dim, batch_size):
        """Helper to convert inputs into vectors (B, dim)"""
        out_vec = torch.zeros(batch_size, dim, device=self.device)
        
        # If input is already a tensor (e.g. provided by user)
        if isinstance(indices_or_vec, torch.Tensor):
            return indices_or_vec.to(self.device).float()

        # If input is int (one-hot)
        if isinstance(indices_or_vec, int):
             out_vec[:, indices_or_vec] = 1.0
             return out_vec
             
        # If input is list (multi-hot)
        if isinstance(indices_or_vec, (list, tuple)):
            for idx in indices_or_vec:
                out_vec[:, idx] = 1.0
            return out_vec
            
        return out_vec

    @torch.no_grad()
    def generate_sample(self, tokens_dict, conditions, scales, num_samples=5, max_length=None, eta=None, temperature=1.0, k_samples=1, shortest_length=14, longest_length=36, length_pool=None, decode_fn=None, score_fn=None):
        """
        max_length:  defaults to the trained max_length from hparams. Never
                     hardcode it -- a too-small value silently yields stubs.
        length_pool: array of real token lengths to draw generation lengths from.
                     Falls back to uniform [shortest_length, longest_length),
                     which is a peptide-residue range and wrong for SAFE.
        decode_fn:   ids -> string. Defaults to joining raw vocab tokens.
        score_fn:    string -> (is_valid, score), used to rank the k_samples
                     candidates. Defaults to the peptide net-charge filter.
        """
        if eta is None:
            eta = self.eta
        if max_length is None:
            max_length = self.hparams.max_length

        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())
        
        self.model.eval()
        device = self.device
        
        # 1. Prepare Feature Vectors (B, Dim)
        vec_species = self._prepare_vector(
            conditions.get('species', []), 
            self.dims['species'],
            num_samples)
        
        vec_groups = self._prepare_vector(
            conditions.get('groups', []),
            self.dims['groups'],
            num_samples)
        
        vec_mic = self._prepare_vector(
            conditions.get('mic', []), 
            self.dims['mic'], 
            num_samples)

        # 2. Prepare CFG Masks
        # mask_drop (True) -> Use Null Embedding
        # mask_keep (False) -> Use Real Embedding
        mask_drop = torch.ones(num_samples, dtype=torch.bool, device=device)
        mask_keep = torch.zeros(num_samples, dtype=torch.bool, device=device)
        
        try:
            x = torch.full((num_samples, max_length), 
                           self.mask_token_id, 
                           dtype=torch.long, 
                           device=device)
            
            if length_pool is not None and len(length_pool) > 0:
                # Draw from the real corpus length distribution. A SAFE molecule
                # is ~300 tokens; a uniform [14, 36mask_drop) draw would only ever produce
                # truncated fragments.
                pool = torch.as_tensor(np.asarray(length_pool), dtype=torch.int32)
                picks = torch.randint(low=0, high=pool.numel(), size=(num_samples,))
                lengths = pool[picks].to(device)
            else:
                lengths = torch.randint(low=shortest_length, high=longest_length, size=(num_samples,), device=device, dtype=torch.int32)
            lengths = lengths.clamp(max=max_length).to(torch.int32)

            t = 0.0
            steps = self.hparams.num_steps
            dt = 1.0 / steps
            
            index_to_token = {i: token for token, i in tokens_dict.items()}
            
            for _ in range(steps):mask_drop
                t_tensor = torch.full((num_samples,), t, device=device)
                
                # --- 4-PASS COMPOSITIONAL GUIDANCE ---
                
                # Pass 1: Unconditional (All Dropped)
                logits_uncond = self.model(x, 
                                           t_tensor, 
                                           lengths,
                                           species_vec=vec_species, 
                                           species_mask=mask_drop,
                                           groups_vec=vec_groups,
                                           groups_mask=mask_drop,
                                           mic_vec=vec_mic,
                                           mic_mask=mask_drop)

                # Pass 2: Species Only (Keep Species, Drop others)
                logits_species = self.model(x, 
                                            t_tensor, 
                                            lengths,
                                            species_vec=vec_species, 
                                            species_mask=mask_keep,
                                            groups_vec=vec_groups,   
                                            groups_mask=mask_drop,
                                            mic_vec=vec_mic,         
                                            mic_mask=mask_drop)

                # Pass 3: Groups Only
                logits_groups = self.model(x, 
                                           t_tensor, 
                                           lengths,
                                           species_vec=vec_species, 
                                           species_mask=mask_drop,
                                           groups_vec=vec_groups,   
                                           groups_mask=mask_keep,
                                           mic_vec=vec_mic,         
                                           mic_mask=mask_drop)

                # Pass 4: MIC Only
                logits_mic = self.model(x, t_tensor, lengths,
                                        species_vec=vec_species, 
                                        species_mask=mask_drop,
                                        groups_vec=vec_groups,   
                                        groups_mask=mask_drop,
                                        mic_vec=vec_mic,         
                                        mic_mask=mask_keep)

                # Combine Guidance Vectors
                g_spec = scales.get('species', 1.0) * (logits_species - logits_uncond)
                g_grp  = scales.get('groups', 1.0) * (logits_groups - logits_uncond)
                g_mic  = scales.get('mic', 1.0) * (logits_mic - logits_uncond)
                
                # Final Logits
                logits = logits_uncond + g_spec + g_grp + g_mic
                logits = logits.to(torch.float32)
                
                if temperature != 1.0:
                    logits = logits / temperature
                
                # --- STANDARD SAMPLING LOGIC ---
                logits[:, :, self.mask_token_id] = -float('inf')
                if self.pad_token_id is not None:
                    logits[:, :, self.pad_token_id] = -float('inf')
                
                x1_probs = F.softmax(logits, dim=-1)
                
                x1_sample = Categorical(x1_probs).sample()
                
                unmask_rate = dt * (1 + eta * t) / (1 - t + 1e-6)
                unmask_rate = min(unmask_rate, 1.0)
                
                should_unmask = torch.rand_like(x.float()) < unmask_rate
                
                is_masked = (x == self.mask_token_id)
                
                x = torch.where(is_masked & should_unmask, x1_sample, x)
                
                if eta > 0 and (t + dt < 1.0):
                    remask_rate = dt * eta
                    remask_rate = min(remask_rate, 1.0)
                    
                    should_remask = torch.rand_like(x.float()) < remask_rate
                    
                    is_revealed = (x != self.mask_token_id)
                    
                    if self.pad_token_id is not None:
                        is_not_padding = (x != self.pad_token_id)
                        is_revealed = is_revealed & is_not_padding
                    
                    x = torch.where(is_revealed & should_remask,  torch.tensor(self.mask_token_id, device=device),  x)
                
                t += dt

            if decode_fn is not None:
                x_np, lens_np = x.cpu().numpy(), lengths.cpu().numpy()
                return [decode_fn(seq[:length]) for seq, length in zip(x_np, lens_np)]

            return self._decode_to_string(x.cpu().numpy(), lengths.cpu().numpy(), index_to_token)

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