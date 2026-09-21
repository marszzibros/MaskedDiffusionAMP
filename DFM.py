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
                 topology_token_ids=None,
                 chemical_token_ids=None,
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
        
        if topology_token_ids is None:
            topology_token_ids = []
        if chemical_token_ids is None:
            chemical_token_ids = []
            
        self.register_buffer("topo_ids", torch.tensor(topology_token_ids, dtype=torch.long), persistent=False)
        self.register_buffer("chem_ids", torch.tensor(chemical_token_ids, dtype=torch.long), persistent=False)

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

    def _get_mask_prob(self, t, t_start, t_end):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        diff = t_end - t_start
        if diff == 0:
            diff = 1e-6
        val = torch.clamp((t - t_start) / diff, 0.0, 1.0)
        return 0.5 * (1.0 + torch.cos(torch.pi * val))
        
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
        t_unsq = t.unsqueeze(-1)
        
        base_mask_prob = 1.0 - t_unsq
        topo_mask_prob = self._get_mask_prob(t_unsq, 0.0, 0.5)
        chem_mask_prob = self._get_mask_prob(t_unsq, 0.3, 1.0)
        
        is_topo = torch.isin(x_1, self.topo_ids)
        is_chem = torch.isin(x_1, self.chem_ids)
        
        mask_prob = torch.where(is_topo, topo_mask_prob, base_mask_prob)
        mask_prob = torch.where(is_chem, chem_mask_prob, mask_prob)
        
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
            else:
                print("Warning: 'token_dict' not found in DataModule. Skipping generation.")
                return

            # Example conditions for validation generation
            example_cond = {'species': [1], 'groups': [2], 'mic': 7}
            example_scales = {'species': 1.0, 'groups': 1.0, 'mic': 1.0}

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
            
            for step in range(steps):
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

                # logits[:, :, 1] = -float('inf')  # Prevent sampling the CLS token
                # logits[:, :, 2] = -float('inf')  # Prevent sampling the SEP token
                # logits[:, :, 4] = -float('inf')  # Prevent sampling the UNK token
                
                x1_probs = F.softmax(logits, dim=-1)
                
                x1_sample = Categorical(x1_probs).sample()
                
                t_scalar = torch.tensor(t, device=device)
                t_next = torch.tensor(t + dt, device=device)
                
                # 1. 计算拓扑基础解掩码率 (Base Rate = (P_t - P_{t+dt}) / P_t)
                p_topo_t = self._get_mask_prob(t_scalar, 0.0, 0.5)
                p_topo_next = self._get_mask_prob(t_next, 0.0, 0.5)
                base_unmask_topo = (p_topo_t - p_topo_next) / (p_topo_t + 1e-6)
                
                # 2. 计算化学基础解掩码率
                p_chem_t = self._get_mask_prob(t_scalar, 0.3, 1.0)
                p_chem_next = self._get_mask_prob(t_next, 0.3, 1.0)
                base_unmask_chem = (p_chem_t - p_chem_next) / (p_chem_t + 1e-6)
                
                # 3. 依赖模型预测来分派 Rate
                is_pred_topo = torch.isin(x1_sample, self.topo_ids)
                is_pred_chem = torch.isin(x1_sample, self.chem_ids)
                is_pred_base = ~(is_pred_topo | is_pred_chem)
                
                # --- Hardcoded hybrid strategy for testing ---
                topo_eta = 10.0
                chem_eta = 5.0
                base_eta = 5.0
                
                topo_noise_scale = 1.0
                chem_noise_scale = 1.0
                base_noise_scale = 1.0
                # ---------------------------------------------
                
                # 默认回退 rate
                base_unmask_rate = torch.full_like(x.float(), dt / (1 - t + 1e-6))
                base_unmask_rate = torch.where(is_pred_topo, base_unmask_topo, base_unmask_rate)
                base_unmask_rate = torch.where(is_pred_chem, base_unmask_chem, base_unmask_rate)
                
                # 4. 施加 eta (Langevin 噪声) 调节
                eta_map = torch.full_like(x.float(), base_eta)
                eta_map = torch.where(is_pred_topo, torch.full_like(eta_map, topo_eta), eta_map)
                eta_map = torch.where(is_pred_chem, torch.full_like(eta_map, chem_eta), eta_map)
                
                unmask_rate = base_unmask_rate * (1 + eta_map * t)
                unmask_rate = torch.clamp(unmask_rate, 0.0, 1.0)

                is_masked = (x == self.mask_token_id)
                
                # 5. Calculate stochastic confidence
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(x.float()) + 1e-9) + 1e-9)
                
                # Apply token-specific noise scale for unmasking
                unmask_noise_scale = torch.full_like(x.float(), base_noise_scale)
                unmask_noise_scale = torch.where(is_pred_topo, torch.full_like(unmask_noise_scale, topo_noise_scale), unmask_noise_scale)
                unmask_noise_scale = torch.where(is_pred_chem, torch.full_like(unmask_noise_scale, chem_noise_scale), unmask_noise_scale)
                
                # For unmasking, use confidence of the predicted token
                pred_probs = x1_probs.gather(dim=-1, index=x1_sample.unsqueeze(-1)).squeeze(-1)
                pred_conf = torch.log(pred_probs + 1e-9) + gumbel_noise * unmask_noise_scale * (1.0 - t)
                
                def get_topk_mask(mask_group, num_to_select, conf):
                    group_conf = torch.where(mask_group, conf, torch.full_like(conf, -float('inf')))
                    sorted_indices = torch.argsort(group_conf, dim=-1, descending=True)
                    batch_size, seq_len = mask_group.shape
                    batch_indices = torch.arange(batch_size, device=mask_group.device).unsqueeze(1)
                    rank = torch.empty_like(sorted_indices)
                    rank[batch_indices, sorted_indices] = torch.arange(seq_len, device=mask_group.device).unsqueeze(0)
                    selected_mask = rank < num_to_select.unsqueeze(1)
                    return selected_mask & mask_group
                
                is_masked_topo = is_masked & is_pred_topo
                is_masked_chem = is_masked & is_pred_chem
                is_masked_base = is_masked & is_pred_base
                
                num_unmask_topo = (is_masked_topo.float() * unmask_rate).sum(dim=1).round().long()
                num_unmask_chem = (is_masked_chem.float() * unmask_rate).sum(dim=1).round().long()
                num_unmask_base = (is_masked_base.float() * unmask_rate).sum(dim=1).round().long()
                
                should_unmask_topo = get_topk_mask(is_masked_topo, num_unmask_topo, pred_conf)
                should_unmask_chem = get_topk_mask(is_masked_chem, num_unmask_chem, pred_conf)
                should_unmask_base = get_topk_mask(is_masked_base, num_unmask_base, pred_conf)
                
                should_unmask = should_unmask_topo | should_unmask_chem | should_unmask_base
                
                x = torch.where(should_unmask, x1_sample, x)
                
                if eta > 0 and (t + dt < 1.0):
                    # 1. 识别当前 x 中已经显露的 Token 属于什么类型
                    is_curr_topo = torch.isin(x, self.topo_ids)
                    is_curr_chem = torch.isin(x, self.chem_ids)
                    is_curr_base = ~(is_curr_topo | is_curr_chem)
                    
                    curr_eta_map = torch.full_like(x.float(), base_eta)
                    curr_eta_map = torch.where(is_curr_topo, torch.full_like(curr_eta_map, topo_eta), curr_eta_map)
                    curr_eta_map = torch.where(is_curr_chem, torch.full_like(curr_eta_map, chem_eta), curr_eta_map)
                    
                    # 2. 用各自的 Schedule 概率来动态缩放基础的加噪率
                    # 当 p_topo_t 降到 0 时（即 t>0.5），拓扑 Token 的重掩码率自动降为 0（绝对锁定）
                    remask_topo = dt * curr_eta_map * p_topo_t
                    remask_chem = dt * curr_eta_map * p_chem_t
                    
                    # 对于 Padding 或未分类的 Token，给一个跟随 1-t 衰减的基础退火率
                    remask_base_rate = dt * curr_eta_map * (1.0 - t)
                    
                    # 3. 组合出像素级/Token级的动态重掩码率
                    remask_rate = torch.full_like(x.float(), 0.0)
                    remask_rate = torch.where(is_curr_topo, remask_topo, remask_rate)
                    remask_rate = torch.where(is_curr_chem, remask_chem, remask_rate)
                    remask_rate = torch.where(is_curr_base, remask_base_rate, remask_rate)
                    
                    remask_rate = torch.clamp(remask_rate, 0.0, 1.0)
                    
                    # 4. 只对非 MASK 且非 PAD 的 Token 执行重掩码
                    is_revealed = (x != self.mask_token_id)
                    if self.pad_token_id is not None:
                        is_revealed = is_revealed & (x != self.pad_token_id)
                        
                    is_revealed_topo = is_revealed & is_curr_topo
                    is_revealed_chem = is_revealed & is_curr_chem
                    is_revealed_base = is_revealed & is_curr_base
                    
                    num_remask_topo = (is_revealed_topo.float() * remask_rate).sum(dim=1).round().long()
                    num_remask_chem = (is_revealed_chem.float() * remask_rate).sum(dim=1).round().long()
                    num_remask_base = (is_revealed_base.float() * remask_rate).sum(dim=1).round().long()
                    
                    # Apply token-specific noise scale for remasking
                    remask_noise_scale = torch.full_like(x.float(), base_noise_scale)
                    remask_noise_scale = torch.where(is_curr_topo, torch.full_like(remask_noise_scale, topo_noise_scale), remask_noise_scale)
                    remask_noise_scale = torch.where(is_curr_chem, torch.full_like(remask_noise_scale, chem_noise_scale), remask_noise_scale)

                    # For remasking, use confidence of the CURRENTLY REVEALED token
                    safe_x = torch.where(x == self.mask_token_id, torch.zeros_like(x), x)
                    if self.pad_token_id is not None:
                        safe_x = torch.where(safe_x == self.pad_token_id, torch.zeros_like(safe_x), safe_x)
                    curr_probs = x1_probs.gather(dim=-1, index=safe_x.unsqueeze(-1)).squeeze(-1)
                    curr_conf = torch.log(curr_probs + 1e-9) + gumbel_noise * remask_noise_scale * (1.0 - t)
                    
                    # Remask the LEAST confident tokens, so we pass -curr_conf
                    should_remask_topo = get_topk_mask(is_revealed_topo, num_remask_topo, -curr_conf)
                    should_remask_chem = get_topk_mask(is_revealed_chem, num_remask_chem, -curr_conf)
                    should_remask_base = get_topk_mask(is_revealed_base, num_remask_base, -curr_conf)
                    
                    should_remask = should_remask_topo | should_remask_chem | should_remask_base
                    
                    x = torch.where(should_remask,  torch.tensor(self.mask_token_id, device=device),  x)
                
                t += dt

                # # Debugging: Print the first two sequences at specific time steps
                # if t < 0.3 and t > 0.25:
                #     if decode_fn is not None:
                #         x_np, lens_np = x.cpu().numpy(), lengths.cpu().numpy()
                #         print("---------------------------------------------------------------------------")
                #         print('early')
                #         print([decode_fn(seq[:length], skip_special_tokens=False) for seq, length in zip(x_np[0:2], lens_np[0:2])])
                #         print("---------------------------------------------------------------------------")
                # if t < 0.5 and t > 0.45:
                #     if decode_fn is not None:
                #         x_np, lens_np = x.cpu().numpy(), lengths.cpu().numpy()
                #         print("---------------------------------------------------------------------------")
                #         print('mid')
                #         print([decode_fn(seq[:length], skip_special_tokens=False) for seq, length in zip(x_np[0:2], lens_np[0:2])])
                #         print("---------------------------------------------------------------------------")
                # if t < 0.8 and t > 0.75:
                #     if decode_fn is not None:
                #         x_np, lens_np = x.cpu().numpy(), lengths.cpu().numpy()
                #         print("---------------------------------------------------------------------------")
                #         print('late')
                #         print([decode_fn(seq[:length], skip_special_tokens=False) for seq, length in zip(x_np[0:2], lens_np[0:2])])
                #         print("---------------------------------------------------------------------------")


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