import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
import transformers
from .models import DIT, EMA
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
                 topo_eta=10.0,
                 chem_eta=5.0,
                 base_eta=5.0,
                 output_dir=None,
                 cond_dropout=0.1,
                 species_dim=6,
                 groups_dim=5,
                 mic_dim=10,
                 hidden_size=1536,
                 n_blocks=24,
                 n_heads=12,
                 # 'fa2' (flash-attn; needs a wheel built for the device arch)
                 # | 'sdpa' (pure torch, runs anywhere) | 'auto' (probe & pick).
                 attn_backend='fa2',
                 # 'pooled': one mean over every masked token. 'class_balanced': the mean of the
                 # per-class means, so a class masked less often is not outweighed.
                 loss_mode='pooled'):

        super().__init__()
        self.save_hyperparameters()
        
        if mask_token_id is None:
            raise ValueError("You must provide the mask_token_id (integer) from your vocabulary.")

        if max_length is None:
            raise ValueError("You must provide max_length; take it from the datamodule "
                             "after setup() rather than hardcoding it.")
            
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id 
        self.topo_eta = topo_eta
        self.chem_eta = chem_eta
        self.base_eta = base_eta
        self.cond_dropout = cond_dropout

        if loss_mode not in ("pooled", "class_balanced"):
            raise ValueError(f"loss_mode must be 'pooled' or 'class_balanced', not {loss_mode!r}")
        self.loss_mode = loss_mode

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
                             n_heads=n_heads,
                             attn_backend=attn_backend)

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

        if self.loss_mode == "class_balanced" and num_masked > 0 and self.topo_ids.numel() > 0:
            # Average the per-class means instead of pooling every masked token. Under the
            # two-schedule the classes are masked over different windows, so pooling gives the
            # class that unmasks earlier less of the loss. Measured -0.41 strict in both
            # tokenizers, so this is available for replication, not recommended.
            groups = [mask_mask & is_topo, mask_mask & is_chem, mask_mask & ~(is_topo | is_chem)]
            terms = []
            for g in groups:
                n_g = g.sum()
                if n_g > 0:
                    terms.append((nll * g.float()).sum() / n_g)
            loss = sum(terms) / len(terms)
        elif num_masked > 0:
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

    def _get_grammar_tokens(self, tokens_dict):
        import re
        linkage_token_ids = set()
        open_paren_id = None
        close_paren_id = None
        dot_token_id = None
        for token_str, token_id in tokens_dict.items():
            if re.match(r'^([1-9]|%\d{2}|%\(\d+\))$', str(token_str)):
                linkage_token_ids.add(token_id)
            elif token_str == '(':
                open_paren_id = token_id
            elif token_str == ')':
                close_paren_id = token_id
            elif token_str == '.':
                dot_token_id = token_id
        return open_paren_id, close_paren_id, dot_token_id, linkage_token_ids

    def _check_and_mask_grammar(
        self,
        seq_tensor,
        length_tensor,
        open_paren_id,
        close_paren_id,
        dot_token_id,
        linkage_token_ids,
    ):
        """One-shot topological graph repair at t = 0.50.

        Enforces:
          1. Syntactically valid parentheses & dot boundaries (no `(4)`, `()`, `..`, `.4`, or unclosed parens).
          2. No self-looped amino acid fragments (`N%k...C%k=O`) or intra/inter label collisions.
          3. Every inter-fragment linkage appears in exactly 2 distinct fragments (pairs up orphans and excess duplicates).
          4. Connected fragment graph: bridges disconnected components and prunes trailing redundant fragments.
        """
        B, L = seq_tensor.shape
        batch_invalid_mask = torch.zeros_like(seq_tensor, dtype=torch.bool)
        any_invalid = False

        seq_np = seq_tensor.cpu().numpy().copy()
        len_np = length_tensor.cpu().numpy()
        ordered_link_ids = sorted(int(t) for t in linkage_token_ids)

        for i in range(B):
            L_i = int(len_np[i])
            seq = seq_np[i, :L_i]

            # --- Step 1: Positional syntax cleanup (dots & parentheses) ---
            # Mask leading/trailing dots and consecutive dots '..'
            for j in range(L_i):
                tok = int(seq[j])
                if dot_token_id is not None and tok == dot_token_id:
                    if j == 0 or j == L_i - 1 or (j > 0 and int(seq[j - 1]) == dot_token_id):
                        batch_invalid_mask[i, j] = True
                        any_invalid = True
                        seq[j] = self.mask_token_id

            # Mask linkages immediately following '(' (e.g. hallucinated `(4)` inside `(C)` sidechain)
            for j in range(1, L_i):
                if int(seq[j]) in linkage_token_ids and int(seq[j - 1]) == open_paren_id:
                    batch_invalid_mask[i, j] = True
                    any_invalid = True
                    seq[j] = self.mask_token_id

            # Balance parentheses per fragment and disallow literal empty '()' (j == open_pos + 1)
            stack = []
            invalid_parens = []
            for j in range(L_i):
                tok = int(seq[j])
                if dot_token_id is not None and tok == dot_token_id:
                    invalid_parens.extend(stack)
                    stack.clear()
                elif tok == open_paren_id:
                    stack.append(j)
                elif tok == close_paren_id:
                    if len(stack) > 0:
                        open_pos = stack.pop()
                        if j == open_pos + 1:
                            invalid_parens.append(open_pos)
                            invalid_parens.append(j)
                    else:
                        invalid_parens.append(j)
            invalid_parens.extend(stack)

            for j in invalid_parens:
                batch_invalid_mask[i, j] = True
                any_invalid = True
                seq[j] = self.mask_token_id

            # --- Step 2: Segment into fragments and classify intra vs inter linkages ---
            frags = []  # list of (start, end)
            f_start = 0
            for j in range(L_i):
                if dot_token_id is not None and int(seq[j]) == dot_token_id:
                    frags.append((f_start, j))
                    f_start = j + 1
            frags.append((f_start, L_i))
            num_frags = len(frags)

            frag_token_pos = {}  # (f_id, tok_id) -> [pos, ...]
            used_toks = set()
            for f_id, (s_pos, e_pos) in enumerate(frags):
                for j in range(s_pos, e_pos):
                    tok = int(seq[j])
                    if tok in linkage_token_ids:
                        frag_token_pos.setdefault((f_id, tok), []).append(j)
                        used_toks.add(tok)

            inter_positions = {}  # tok_id -> [(pos, f_id), ...]
            unpaired = []  # [(pos, f_id, tok_id), ...]

            def get_free_tok():
                for cand in ordered_link_ids:
                    if cand not in used_toks:
                        used_toks.add(cand)
                        return cand
                return ordered_link_ids[-1]

            for (f_id, tok), pos_list in frag_token_pos.items():
                if len(pos_list) % 2 == 1:
                    # Odd count in fragment: last occurrence is an inter-fragment attachment
                    inter_positions.setdefault(tok, []).append((pos_list[-1], f_id))

            # Detect self-looped isolated fragments (e.g. `N7[C@@H](CC(C)C)C7=O` which has 0 odd linkages
            # and one linkage appearing twice) so we can splice them into the main chain in Step 4B.
            self_loop_frags = []  # [(f_id, p1, p2, tok), ...]
            if num_frags > 1:
                for f_id in range(num_frags):
                    f_toks = [(tok, pos_list) for (f2, tok), pos_list in frag_token_pos.items() if f2 == f_id]
                    has_odd = any(len(pl) % 2 == 1 for _, pl in f_toks)
                    if not has_odd and len(f_toks) == 1 and len(f_toks[0][1]) == 2:
                        tok, pl = f_toks[0]
                        self_loop_frags.append((f_id, pl[0], pl[1], tok))

            for tok in inter_positions:
                inter_positions[tok].sort(key=lambda x: x[0])

            # --- Step 3: Build initial Union-Find from unambiguous 2-occurrence linkages ---
            parent = list(range(num_frags))
            comp_size = [1] * num_frags
            frag_lens = [e_pos - s_pos for s_pos, e_pos in frags]
            cycle_edges = []  # [(tok, (pos1, f1), (pos2, f2)), ...]
            active_pairs = {}  # tok -> [(pos1, f1), (pos2, f2)]

            def find(x_f):
                while parent[x_f] != x_f:
                    parent[x_f] = parent[parent[x_f]]
                    x_f = parent[x_f]
                return x_f

            def union(a_f, b_f):
                ra, rb = find(a_f), find(b_f)
                if ra != rb:
                    if comp_size[ra] < comp_size[rb]:
                        ra, rb = rb, ra
                    parent[rb] = ra
                    comp_size[ra] += comp_size[rb]
                    return True
                return False

            for tok, pf_list in inter_positions.items():
                if len(pf_list) == 2:
                    if pf_list[0][1] != pf_list[1][1]:
                        active_pairs[tok] = pf_list
                        if not union(pf_list[0][1], pf_list[1][1]):
                            cycle_edges.append((tok, pf_list[0], pf_list[1]))
                    else:
                        unpaired.append((pf_list[0][0], pf_list[0][1], tok))
                        unpaired.append((pf_list[1][0], pf_list[1][1], tok))
                elif len(pf_list) == 1:
                    unpaired.append((pf_list[0][0], pf_list[0][1], tok))

            # Now resolve linkages with >= 3 occurrences:
            # Among even-indexed outlier candidates (which preserve adjacent DFS pairs),
            # prefer the candidate that maximizes inter-component merges and minimizes pair token distance.
            for tok, pf_list in inter_positions.items():
                if len(pf_list) % 2 == 1 and len(pf_list) >= 3:
                    best_idx = 0
                    best_key = (-1, float('inf'))
                    for cand_idx in range(0, len(pf_list), 2):
                        rem = pf_list[:cand_idx] + pf_list[cand_idx + 1 :]
                        merges = sum(
                            1 for k in range(0, len(rem), 2)
                            if rem[k][1] != rem[k + 1][1] and find(rem[k][1]) != find(rem[k + 1][1])
                        )
                        cost = sum(rem[k + 1][0] - rem[k][0] for k in range(0, len(rem), 2))
                        key = (merges, -cost)
                        if key > best_key:
                            best_key = key
                            best_idx = cand_idx
                    unpaired.append((pf_list[best_idx][0], pf_list[best_idx][1], tok))
                    rem = pf_list[:best_idx] + pf_list[best_idx + 1 :]
                    for k in range(0, len(rem), 2):
                        if rem[k][1] != rem[k + 1][1]:
                            use_t = tok if k == 0 else get_free_tok()
                            used_toks.add(use_t)
                            seq_tensor[i, rem[k][0]] = use_t
                            seq_tensor[i, rem[k + 1][0]] = use_t
                            active_pairs[use_t] = [rem[k], rem[k + 1]]
                            if not union(rem[k][1], rem[k + 1][1]):
                                cycle_edges.append((use_t, rem[k], rem[k + 1]))
                        else:
                            unpaired.append((rem[k][0], rem[k][1], tok))
                            unpaired.append((rem[k + 1][0], rem[k + 1][1], tok))
                elif len(pf_list) >= 4 and len(pf_list) % 2 == 0:
                    for k in range(0, len(pf_list), 2):
                        if pf_list[k][1] != pf_list[k + 1][1]:
                            use_t = tok if k == 0 else get_free_tok()
                            used_toks.add(use_t)
                            seq_tensor[i, pf_list[k][0]] = use_t
                            seq_tensor[i, pf_list[k + 1][0]] = use_t
                            active_pairs[use_t] = [pf_list[k], pf_list[k + 1]]
                            if not union(pf_list[k][1], pf_list[k + 1][1]):
                                cycle_edges.append((use_t, pf_list[k], pf_list[k + 1]))
                        else:
                            unpaired.append((pf_list[k][0], pf_list[k][1], tok))
                            unpaired.append((pf_list[k + 1][0], pf_list[k + 1][1], tok))

            # --- Step 4: Component-aware pairing of unpaired attachments ---
            unpaired.sort(key=lambda x: x[0])
            used_unpaired = [False] * len(unpaired)

            # Pass 4A: Pair unpaired attachments across different components, prioritizing:
            #   1. Merging two multi-fragment chains (comp_size[f1] > 1 and comp_size[f2] > 1)
            #   2. Merging any two different components where at least one is a chain (comp_size > 1)
            #   3. Closer token distance
            for idx1 in range(len(unpaired)):
                if used_unpaired[idx1]:
                    continue
                pos1, f1, tok1 = unpaired[idx1]
                best_idx2 = None
                best_key = (-1, -1, -1, -float('inf'))
                for idx2 in range(idx1 + 1, len(unpaired)):
                    if used_unpaired[idx2]:
                        continue
                    pos2, f2, tok2 = unpaired[idx2]
                    if f2 == f1:
                        continue
                    diff_comp = int(find(f1) != find(f2))
                    both_chains = int(comp_size[find(f1)] > 1 and comp_size[find(f2)] > 1)
                    any_chain = int(comp_size[find(f1)] > 1 or comp_size[find(f2)] > 1)
                    key = (diff_comp, both_chains, any_chain, -(pos2 - pos1))
                    if key > best_key:
                        best_key = key
                        best_idx2 = idx2
                if best_idx2 is not None:
                    pos2, f2, tok2 = unpaired[best_idx2]
                    # Pick a token ID not already used as an active pair or intra-fragment ring in f1/f2
                    f1_rings = {t for (ff, t) in frag_token_pos if ff == f1 and t != tok1}
                    f2_rings = {t for (ff, t) in frag_token_pos if ff == f2 and t != tok2}
                    if tok1 not in active_pairs and tok1 not in f2_rings:
                        target_tok = tok1
                    elif tok2 not in active_pairs and tok2 not in f1_rings:
                        target_tok = tok2
                    else:
                        target_tok = get_free_tok()
                    used_toks.add(target_tok)
                    seq_tensor[i, pos1] = target_tok
                    seq_tensor[i, pos2] = target_tok
                    pair_entry = [(pos1, f1), (pos2, f2)]
                    active_pairs[target_tok] = pair_entry
                    if not union(f1, f2):
                        cycle_edges.append((target_tok, pair_entry[0], pair_entry[1]))
                    used_unpaired[idx1] = True
                    used_unpaired[best_idx2] = True

            # If a single leftover unpaired attachment remains on an isolated 2-token cap fragment (e.g. `.N%14`),
            # absorb it into the adjacent fragment by masking its dot separator and orphan linkage token
            # so it doesn't decode into a disconnected `.N` (ammonia) molecule.
            for idx1 in range(len(unpaired)):
                if not used_unpaired[idx1]:
                    pos1, f1, tok1 = unpaired[idx1]
                    if num_frags > 1 and comp_size[find(f1)] == 1 and frag_lens[f1] <= 2:
                        s_pos, e_pos = frags[f1]
                        dot_pos = s_pos - 1 if s_pos > 0 else e_pos
                        if 0 <= dot_pos < L_i:
                            batch_invalid_mask[i, dot_pos] = True
                        batch_invalid_mask[i, pos1] = True
                        any_invalid = True

            # Pass 4B: Splice any self-looped isolated residue (`N7[C@@H](...)C7=O`) into the nearest active edge
            if self_loop_frags and active_pairs:
                for f_id, p1, p2, orig_tok in self_loop_frags:
                    if comp_size[find(f_id)] > 1:
                        continue
                    # Find the active edge closest in token position to p1
                    best_tokA = min(
                        active_pairs.keys(),
                        key=lambda tA: min(abs(active_pairs[tA][0][0] - p1), abs(active_pairs[tA][1][0] - p1)),
                    )
                    (pA1, fA1), (pA2, fA2) = active_pairs[best_tokA]
                    new_tok = orig_tok if orig_tok not in active_pairs else get_free_tok()
                    used_toks.add(new_tok)
                    # Splice fA1 --(best_tokA)-- f_id(p1, p2) --(new_tok)-- fA2
                    seq_tensor[i, p1] = best_tokA
                    seq_tensor[i, p2] = new_tok
                    seq_tensor[i, pA2] = new_tok
                    active_pairs[best_tokA] = [(pA1, fA1), (p1, f_id)]
                    active_pairs[new_tok] = [(p2, f_id), (pA2, fA2)]
                    union(fA1, f_id)

            # --- Step 5: Bridge disconnected components using cycle edges ---
            # Swapping a cycle edge (u1-v1) in component A with any edge (u2-v2) in component B
            # opens the cycle and splices it into component B without disconnecting component A!
            if num_frags > 1 and cycle_edges:
                for c_tok, (pC1, fC1), (pC2, fC2) in list(cycle_edges):
                    r_cycle = find(fC1)
                    # Look for any active edge in a different component
                    other_edge = None
                    for tokB, ((pB1, fB1), (pB2, fB2)) in active_pairs.items():
                        if find(fB1) != r_cycle:
                            other_edge = (tokB, (pB1, fB1), (pB2, fB2))
                            break
                    if other_edge is not None:
                        tokB, (pB1, fB1), (pB2, fB2) = other_edge
                        seq_tensor[i, pC2] = tokB
                        seq_tensor[i, pB1] = c_tok
                        active_pairs[c_tok] = [(pC1, fC1), (pB1, fB1)]
                        active_pairs[tokB] = [(pC2, fC2), (pB2, fB2)]
                        union(fC1, fB1)

        return batch_invalid_mask, any_invalid

    @torch.no_grad()
    def generate_sample(self, tokens_dict, conditions, scales, num_samples=5, max_length=None, temperature=1.0, shortest_length=14, longest_length=36, length_pool=None, decode_fn=None, topo_noise_scale=1.0, chem_noise_scale=1.0, base_noise_scale=1.0, use_grammar_check=True):
        """
        max_length:  defaults to the trained max_length from hparams. Never
                     hardcode it -- a too-small value silently yields stubs.
        length_pool: array of real token lengths to draw generation lengths from.
                     Falls back to uniform [shortest_length, longest_length),
                     which is a peptide-residue range and wrong for SAFE.
        decode_fn:   ids -> string. Defaults to joining raw vocab tokens.
        """
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

            # [改动] 单次拓扑锁定语法修复 (t = 0.50) 与拓扑重掩码截止阈值
            checked_at_half = False
            topo_remask_cutoff = 0.45

            if use_grammar_check:
                open_paren_id, close_paren_id, dot_token_id, linkage_token_ids = self._get_grammar_tokens(tokens_dict)

            for step in range(steps):
                # 在 t >= 0.50 正式锁定拓扑前做一次性结构完整性与连通性修复
                if use_grammar_check and t >= 0.50 - 1e-6 and not checked_at_half:
                    checked_at_half = True
                    invalid_mask, any_invalid = self._check_and_mask_grammar(
                        x, lengths, open_paren_id, close_paren_id, dot_token_id, linkage_token_ids
                    )
                    if any_invalid:
                        x = torch.where(invalid_mask, torch.tensor(self.mask_token_id, device=device), x)

                t_tensor = torch.full((num_samples,), t, device=device)
                
                # --- 4-PASS COMPOSITIONAL GUIDANCE ---
                logits_uncond = self.model(x, t_tensor, lengths,
                                           species_vec=vec_species, species_mask=mask_drop,
                                           groups_vec=vec_groups, groups_mask=mask_drop,
                                           mic_vec=vec_mic, mic_mask=mask_drop)

                logits_species = self.model(x, t_tensor, lengths,
                                            species_vec=vec_species, species_mask=mask_keep,
                                            groups_vec=vec_groups, groups_mask=mask_drop,
                                            mic_vec=vec_mic, mic_mask=mask_drop)

                logits_groups = self.model(x, t_tensor, lengths,
                                           species_vec=vec_species, species_mask=mask_drop,
                                           groups_vec=vec_groups, groups_mask=mask_keep,
                                           mic_vec=vec_mic, mic_mask=mask_drop)

                logits_mic = self.model(x, t_tensor, lengths,
                                        species_vec=vec_species, species_mask=mask_drop,
                                        groups_vec=vec_groups, groups_mask=mask_drop,
                                        mic_vec=vec_mic, mic_mask=mask_keep)

                g_spec = scales.get('species', 1.0) * (logits_species - logits_uncond)
                g_grp  = scales.get('groups', 1.0) * (logits_groups - logits_uncond)
                g_mic  = scales.get('mic', 1.0) * (logits_mic - logits_uncond)
                
                logits = logits_uncond + g_spec + g_grp + g_mic
                logits = logits.to(torch.float32)
                
                if temperature != 1.0:
                    logits = logits / temperature
                
                # --- STANDARD SAMPLING LOGIC ---
                logits[:, :, self.mask_token_id] = -float('inf')
                if self.pad_token_id is not None:
                    logits[:, :, self.pad_token_id] = -float('inf')

                if t >= 0.5:
                    logits[:, :, self.topo_ids] = -float('inf')

                x1_probs = F.softmax(logits, dim=-1)
                x1_sample = Categorical(x1_probs).sample()
                
                t_scalar = torch.tensor(t, device=device)
                t_next = torch.tensor(t + dt, device=device)
                
                # 1. 计算拓扑基础解掩码率
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
                
                base_unmask_rate = torch.full_like(x.float(), dt / (1 - t + 1e-6))
                base_unmask_rate = torch.where(is_pred_topo, base_unmask_topo, base_unmask_rate)
                base_unmask_rate = torch.where(is_pred_chem, base_unmask_chem, base_unmask_rate)
                
                # 4. 施加 eta 调节
                eta_map = torch.full_like(x.float(), self.base_eta)
                eta_map = torch.where(is_pred_topo, torch.full_like(eta_map, self.topo_eta), eta_map)
                eta_map = torch.where(is_pred_chem, torch.full_like(eta_map, self.chem_eta), eta_map)
                
                unmask_rate = base_unmask_rate * (1 + eta_map * t)
                unmask_rate = torch.clamp(unmask_rate, 0.0, 1.0)

                is_masked = (x == self.mask_token_id)
                
                # 5. Calculate stochastic confidence for unmasking
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(x.float()) + 1e-9) + 1e-9)
                
                unmask_noise_scale = torch.full_like(x.float(), base_noise_scale)
                unmask_noise_scale = torch.where(is_pred_topo, torch.full_like(unmask_noise_scale, topo_noise_scale), unmask_noise_scale)
                unmask_noise_scale = torch.where(is_pred_chem, torch.full_like(unmask_noise_scale, chem_noise_scale), unmask_noise_scale)
                
                topo_decay = max(0.0, 0.5 - t)
                unmask_decay = torch.full_like(x.float(), 1.0 - t)
                unmask_decay = torch.where(is_pred_topo, torch.full_like(unmask_decay, topo_decay), unmask_decay)

                pred_probs = x1_probs.gather(dim=-1, index=x1_sample.unsqueeze(-1)).squeeze(-1)
                pred_conf = torch.log(pred_probs + 1e-9) + gumbel_noise * unmask_noise_scale * unmask_decay
                
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
                
                # [改动 2] 在 t >= topo_remask_cutoff 收尾阶段，仅对高置信度拓扑槽位 (>0.5) 使用 ceil()，
                # 防止 Categorical 隨機抽到 0.1% 概率拓扑 token 时被强制塞进化学侧链槽位 (如 `(4)`)
                raw_unmask_topo = (is_masked_topo.float() * unmask_rate).sum(dim=1)
                if t >= topo_remask_cutoff:
                    confident_topo = is_masked_topo & (pred_probs > 0.5)
                    raw_conf_topo = (confident_topo.float() * unmask_rate).sum(dim=1)
                    num_unmask_topo = torch.max(raw_unmask_topo.round().long(), raw_conf_topo.ceil().long())
                else:
                    num_unmask_topo = raw_unmask_topo.round().long()

                num_unmask_chem = (is_masked_chem.float() * unmask_rate).sum(dim=1).round().long()
                num_unmask_base = (is_masked_base.float() * unmask_rate).sum(dim=1).round().long()
                
                should_unmask_topo = get_topk_mask(is_masked_topo, num_unmask_topo, pred_conf)
                should_unmask_chem = get_topk_mask(is_masked_chem, num_unmask_chem, pred_conf)
                should_unmask_base = get_topk_mask(is_masked_base, num_unmask_base, pred_conf)
                
                should_unmask = should_unmask_topo | should_unmask_chem | should_unmask_base
                
                x = torch.where(should_unmask, x1_sample, x)
                
                if (self.topo_eta > 0 or self.chem_eta > 0 or self.base_eta > 0) and (t + dt < 1.0):
                    is_curr_topo = torch.isin(x, self.topo_ids)
                    is_curr_chem = torch.isin(x, self.chem_ids)
                    is_curr_base = ~(is_curr_topo | is_curr_chem)
                    
                    curr_eta_map = torch.full_like(x.float(), self.base_eta)
                    curr_eta_map = torch.where(is_curr_topo, torch.full_like(curr_eta_map, self.topo_eta), curr_eta_map)
                    curr_eta_map = torch.where(is_curr_chem, torch.full_like(curr_eta_map, self.chem_eta), curr_eta_map)
                    
                    remask_topo = dt * curr_eta_map * p_topo_t
                    remask_chem = dt * curr_eta_map * p_chem_t
                    remask_base_rate = dt * curr_eta_map * (1.0 - t)
                    
                    remask_rate = torch.full_like(x.float(), 0.0)
                    remask_rate = torch.where(is_curr_topo, remask_topo, remask_rate)
                    remask_rate = torch.where(is_curr_chem, remask_chem, remask_rate)
                    remask_rate = torch.where(is_curr_base, remask_base_rate, remask_rate)
                    
                    remask_rate = torch.clamp(remask_rate, 0.0, 1.0)
                    
                    # [改动 4A] 保护当前步刚刚解封的 Token (~should_unmask)，避免用更新前的旧概率立即误杀
                    is_revealed = (x != self.mask_token_id) & (~should_unmask)
                    if self.pad_token_id is not None:
                        is_revealed = is_revealed & (x != self.pad_token_id)
                        
                    is_revealed_topo = is_revealed & is_curr_topo
                    is_revealed_chem = is_revealed & is_curr_chem
                    is_revealed_base = is_revealed & is_curr_base
                    
                    # [改动 1] 在 t >= topo_remask_cutoff (0.40) 时停止挖新的拓扑坑，留出纯收敛缓冲期
                    if t >= topo_remask_cutoff:
                        num_remask_topo = torch.zeros(num_samples, dtype=torch.long, device=device)
                    else:
                        num_remask_topo = (is_revealed_topo.float() * remask_rate).sum(dim=1).round().long()

                    num_remask_chem = (is_revealed_chem.float() * remask_rate).sum(dim=1).round().long()
                    num_remask_base = (is_revealed_base.float() * remask_rate).sum(dim=1).round().long()
                    
                    remask_noise_scale = torch.full_like(x.float(), base_noise_scale)
                    remask_noise_scale = torch.where(is_curr_topo, torch.full_like(remask_noise_scale, topo_noise_scale), remask_noise_scale)
                    remask_noise_scale = torch.where(is_curr_chem, torch.full_like(remask_noise_scale, chem_noise_scale), remask_noise_scale)

                    topo_decay = max(0.0, 0.5 - t)
                    remask_decay = torch.full_like(x.float(), 1.0 - t)
                    remask_decay = torch.where(is_curr_topo, torch.full_like(remask_decay, topo_decay), remask_decay)

                    # [改动 4B] 为 Remask 独立采样一份新的 Gumbel 噪声，避免与 Unmask 噪声耦合
                    gumbel_noise_remask = -torch.log(-torch.log(torch.rand_like(x.float()) + 1e-9) + 1e-9)

                    safe_x = torch.where(x == self.mask_token_id, torch.zeros_like(x), x)
                    if self.pad_token_id is not None:
                        safe_x = torch.where(safe_x == self.pad_token_id, torch.zeros_like(safe_x), safe_x)
                    curr_probs = x1_probs.gather(dim=-1, index=safe_x.unsqueeze(-1)).squeeze(-1)
                    curr_conf = torch.log(curr_probs + 1e-9) + gumbel_noise_remask * remask_noise_scale * remask_decay
                    
                    should_remask_topo = get_topk_mask(is_revealed_topo, num_remask_topo, -curr_conf)
                    should_remask_chem = get_topk_mask(is_revealed_chem, num_remask_chem, -curr_conf)
                    should_remask_base = get_topk_mask(is_revealed_base, num_remask_base, -curr_conf)
                    
                    should_remask = should_remask_topo | should_remask_chem | should_remask_base
                    
                    x = torch.where(should_remask, torch.tensor(self.mask_token_id, device=device), x)
                
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