import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import safe as sf
from custom_tokenizer import OrthogonalSafeTokenizer
from rdkit import Chem, RDLogger

# SAFE decoding of a half-trained model produces a lot of invalid fragments;
# RDKit logs every one of them at error level otherwise.
RDLogger.DisableLog('rdApp.*')


# Condition layout expected by DiscreteFlowMatching.training_step:
#   [0:6]   species  one-hot
#   [6:11]  groups   multi-hot
#   [11:16] objects  multi-hot  (sliced out and ignored by the model)
#   [16:26] MIC bin  one-hot
TARGET_SPECIES = ['escherichia coli', 'pseudomonas aeruginosa', 'klebsiella pneumoniae',
                  'staphylococcus aureus', 'bacillus subtilis', 'staphylococcus epidermidis']
TARGET_GROUPS = ['GRAM-', 'GRAM+', 'MAMMALIAN CELL', 'FUNGUS', 'OTHER']
TARGET_OBJECTS = ['LIPID BILAYER', 'DNA / RNA', 'CYTOPLASMIC PROTEIN', 'MEMBRANE PROTEIN', 'OTHER']

CONDITION_DIM = len(TARGET_SPECIES) + len(TARGET_GROUPS) + len(TARGET_OBJECTS) + 10
TOKENIZER_PATH = "tokenizer_vocab.csv"

# arm -> (tokenizer, SAFE corpus), all built by run.sh.
#
# Only brics and recap: the other slicers blow the SMILES %99 ring-label ceiling
# and their corpora decode at 2-62%. Only splitter=safe: with splitter=none BPE
# bakes absolute ring numbers into its merges. The _reorder arms are the same
# corpus with fragments re-emitted depth-first (scripts/reorder_safe.py), which
# cuts median pair span ~16x at identical token count and vocabulary.
ARMS = {
    "brics_safe":         ("tokenizers/tok_brics_safe.json",         "tokenizers/safe_brics.csv"),
    "brics_reorder_safe": ("tokenizers/tok_brics_reorder_safe.json", "tokenizers/safe_brics_reorder.csv"),
    "recap_safe":         ("tokenizers/tok_recap_safe.json",         "tokenizers/safe_recap.csv"),
    "recap_reorder_safe": ("tokenizers/tok_recap_reorder_safe.json", "tokenizers/safe_recap_reorder.csv"),
}


class _SafeJsonTokenizer:
    """SAFETokenizer wearing OrthogonalSafeTokenizer's interface.

    The two disagree on three things -- constructor, vocab accessor and decode --
    so this adapts the .json format onto the surface dataset.py already uses
    (.token2id / .encode / .decode) rather than forking every call site.
    """

    def __init__(self, path):
        from safe import SAFETokenizer
        self._tok = SAFETokenizer.load(path)
        self._hf = self._tok.get_pretrained()
        self.token2id = dict(self._hf.get_vocab())

    def encode(self, safe_string, add_special_tokens=True):
        # SAFETokenizer.encode always adds them; it has no opt-out.
        return self._tok.encode(safe_string)

    def decode(self, ids, skip_special_tokens=True):
        return self._hf.decode(list(ids), skip_special_tokens=skip_special_tokens)


def load_tokenizer(path):
    """Load a SAFETokenizer .json or an OrthogonalSafeTokenizer .csv vocab.

    Dispatch on extension because OrthogonalSafeTokenizer.load_csv() given a
    .json returns an EMPTY vocabulary without raising -- a silent failure that
    otherwise surfaces much later as garbage decoding.
    """
    tok = _SafeJsonTokenizer(path) if str(path).endswith(".json") \
        else OrthogonalSafeTokenizer.load_csv(path)
    if not tok.token2id:
        raise ValueError(
            f"tokenizer at {path!r} loaded an empty vocabulary -- wrong format "
            f"for this loader?")
    return tok


def encode_safe_strings(tokenizer, safe_strings):
    return [tokenizer.encode(safe_string, add_special_tokens=True)
            for safe_string in safe_strings]


def parse_list(value):
    """amp.csv stores targetGroups / targetObjects as stringified python lists."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    return [item.strip().strip("'\"") for item in value.strip('[]').split(',') if item.strip()]


class AMPSafeConditions:
    """Joins the peptide table, the per-species activity table and the SAFE strings."""

    def __init__(self, data_path, mic_bins=10, safe_csv=None):
        dbaasp_dir = os.path.join(data_path, "dbaasp")
        safe_dir = os.path.join(data_path, "safe")

        amp = pd.read_csv(os.path.join(dbaasp_dir, "amp.csv"))
        activity = pd.read_csv(os.path.join(dbaasp_dir, "amp_activity.csv"))
        safe = pd.read_csv(safe_csv or os.path.join(safe_dir, "modified_amp_safe.csv"))

        amp['targetGroups'] = amp['targetGroups'].apply(parse_list)
        amp['targetObjects'] = amp['targetObjects'].apply(parse_list)

        safe = safe.dropna(subset=['safe'])
        safe = safe[safe['safe'].str.len() > 0]

        # One row per (peptide, species): species and MIC are genuine per-example
        # labels rather than something collapsed onto the peptide.
        df = activity.merge(amp[['amp_id', 'targetGroups', 'targetObjects']], on='amp_id')
        df = df.merge(safe[['amp_id', 'safe']], on='amp_id')
        df = df[df['species'].isin(TARGET_SPECIES)]
        df = df.dropna(subset=['mic_ug_per_ml'])

        self.species_dict = {name: i for i, name in enumerate(TARGET_SPECIES)}
        self.groups_dict = {name: i for i, name in enumerate(TARGET_GROUPS)}
        self.objects_dict = {name: i for i, name in enumerate(TARGET_OBJECTS)}

        # MIC is log-normal across four orders of magnitude, so bin on the log.
        df['MIC_log'] = np.log(df['mic_ug_per_ml'] + 1e-6)
        df['MIC_category'], self.mic_bin_edges = pd.qcut(
            df['MIC_log'], q=mic_bins, labels=False, retbins=True, duplicates='drop')
        self.mic_bins = int(df['MIC_category'].max()) + 1

        self.df = df.reset_index(drop=True)


class AMPSafeDataset(Dataset):
    """SAFE-encoded peptides conditioned on (species, target groups, MIC bin)."""

    def __init__(self, data_path="molecular_dataset/dataset/data/", max_length=None,
                 mic_bins=10, arm=None):
        # max_length=None keeps every molecule: the cutoff is set to the longest
        # arm=None keeps the previous behaviour (the CSV vocab + modified_amp_safe).
        tokenizer_path, safe_csv = ARMS.get(arm, (TOKENIZER_PATH, None))
        if arm is not None and arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; known arms: {sorted(ARMS)}")
        self.arm = arm

        self.conditions = AMPSafeConditions(data_path, mic_bins=mic_bins, safe_csv=safe_csv)

        self.tokenizer = load_tokenizer(tokenizer_path)

        # Read special ids from the loaded vocabulary rather than hardcoding them.
        self.pad_token_id = self.tokenizer.token2id.get("[PAD]")
        self.mask_token_id = self.tokenizer.token2id.get("[MASK]")
        self.token_dict = self.tokenizer.token2id
        self.num_tokens = len(self.token_dict)

        if self.pad_token_id is None or self.mask_token_id is None:
            raise ValueError("Tokenizer is missing a [PAD] or [MASK] token.")

        df = self.conditions.df

        # Tokenize each distinct molecule once, then index per activity row.
        unique = df[['amp_id', 'safe']].drop_duplicates('amp_id').reset_index(drop=True)
        encoded = encode_safe_strings(self.tokenizer, unique['safe'].tolist())

        lengths = np.array([len(ids) for ids in encoded])
        self.max_length = int(lengths.max()) if max_length is None else int(max_length)

        keep = lengths <= self.max_length
        dropped = int((~keep).sum())
        if dropped:
            print(f"[AMPSafeDataset] dropped {dropped}/{len(keep)} molecules "
                  f"longer than max_length={self.max_length} "
                  f"(longest {int(lengths.max())} tokens)")

        kept_ids = unique.loc[keep, 'amp_id'].to_numpy()
        # Stored unpadded; the collate function pads each batch to its own max.
        self.sequences = [np.asarray(encoded[idx], dtype=np.int64) for idx in np.flatnonzero(keep)]

        self.row_of_amp_id = {amp_id: row for row, amp_id in enumerate(kept_ids)}
        df = df[df['amp_id'].isin(self.row_of_amp_id)].reset_index(drop=True)
        self.df = df

        self.sequence_index = df['amp_id'].map(self.row_of_amp_id).to_numpy()
        self.conditions_array = self._build_conditions(df)
        # Per-example token lengths, for length bucketing and for drawing
        # realistic lengths at sampling time.
        self.token_lengths = np.array([len(self.sequences[i]) for i in self.sequence_index])

    def _build_conditions(self, df):
        out = np.zeros((len(df), CONDITION_DIM), dtype=np.float32)
        group_offset = len(TARGET_SPECIES)
        object_offset = group_offset + len(TARGET_GROUPS)
        mic_offset = object_offset + len(TARGET_OBJECTS)

        for i, row in enumerate(df.itertuples(index=False)):
            out[i, self.conditions.species_dict[row.species]] = 1.0
            for group in row.targetGroups:
                if group in self.conditions.groups_dict:
                    out[i, group_offset + self.conditions.groups_dict[group]] = 1.0
            for obj in row.targetObjects:
                if obj in self.conditions.objects_dict:
                    out[i, object_offset + self.conditions.objects_dict[obj]] = 1.0
            out[i, mic_offset + int(row.MIC_category)] = 1.0

        return out

    def __len__(self):
        return len(self.sequence_index)

    def __getitem__(self, idx):
        return {
            "sequence": torch.from_numpy(self.sequences[self.sequence_index[idx]]),
            "condition": torch.from_numpy(self.conditions_array[idx]),
        }

    # ---- decoding -------------------------------------------------------

    def decode(self, ids):
        """Token ids -> SAFE string, with [PAD]/[CLS]/[SEP] removed."""
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return self.tokenizer.decode(list(ids), skip_special_tokens=True).replace(' ', '')

    def decode_to_smiles(self, ids):
        """Token ids -> (safe_string, smiles or None).

        smiles is None when the generated token string is not a decodable SAFE
        molecule -- unbalanced ring closures, a fragment cut off mid-attachment,
        an unparseable atom block. This is the validity signal worth logging.
        """
        safe_str = self.decode(ids)
        return safe_str, safe_to_smiles(safe_str)


class SafeDecoder:
    """Tokenizer-only decoder, for sampling without loading the whole dataset.

    Everything the sampler needs to turn model output back into chemistry:
    ids -> SAFE string -> SMILES, plus a validity score for candidate ranking.
    """

    def __init__(self, tokenizer_path=TOKENIZER_PATH):
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.pad_token_id = self.tokenizer.token2id.get("[PAD]")
        self.mask_token_id = self.tokenizer.token2id.get("[MASK]")
        self.token_dict = self.tokenizer.token2id

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return self.tokenizer.decode(list(ids), skip_special_tokens=True).replace(' ', '')

    def smiles_from_safe(self, safe_str):
        return safe_to_smiles(safe_str)

    def score(self, safe_str):
        """(is_valid, score) for ranking k_samples candidates in generate_sample.

        Score is heavy-atom count, so among decodable candidates the larger
        molecule wins rather than a trivial one-fragment answer.
        """
        smiles = safe_to_smiles(safe_str)
        if smiles is None:
            return False, 0.0
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, 0.0
        return True, float(mol.GetNumHeavyAtoms())

    def length_pool(self, safe_csv_path):
        """Token lengths of the real corpus, to draw generation lengths from."""
        df = pd.read_csv(safe_csv_path).dropna(subset=['safe'])
        encoded = encode_safe_strings(self.tokenizer, df['safe'].tolist())
        return np.array([len(ids) for ids in encoded])


def safe_to_smiles(safe_str):
    """Decode a SAFE string to canonical SMILES, or None if it is not valid."""
    if not safe_str:
        return None
    try:
        smiles = sf.decode(safe_str, canonical=True, ignore_errors=False)
    except Exception:
        return None
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def collate_pad(batch, pad_token_id, multiple_of=32):
    """Pad a batch to its own longest member, rounded up to `multiple_of`.

    Rounding keeps the number of distinct sequence lengths small, which matters
    because the DiT's rotary embedding caches per length (models/DiTwithCondition.py).
    """
    lengths = [item['sequence'].shape[0] for item in batch]
    width = max(lengths)
    if multiple_of > 1:
        width = -(-width // multiple_of) * multiple_of

    sequences = torch.full((len(batch), width), pad_token_id, dtype=torch.long)
    for i, item in enumerate(batch):
        sequences[i, :lengths[i]] = item['sequence']

    return {
        "sequence": sequences,
        "condition": torch.stack([item['condition'] for item in batch]),
    }


class AMPSafeDataModule(L.LightningDataModule):
    def __init__(self, file_path="molecular_dataset/dataset/data/", max_length=None,
                 batch_size=16, mic_bins=10, num_workers=4, arm=None):
        super().__init__()
        self.arm = arm
        self.file_path = file_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.mic_bins = mic_bins
        self.num_workers = num_workers
        self.full_dataset = None

    def setup(self, stage=None):
        if self.full_dataset is not None:
            return
        self.full_dataset = AMPSafeDataset(
            data_path=self.file_path, max_length=self.max_length,
            mic_bins=self.mic_bins, arm=self.arm)

        # Surfaced for trainer.py and for DiscreteFlowMatching's decoding.
        self.token_dict = self.full_dataset.token_dict
        self.num_tokens = self.full_dataset.num_tokens
        self.mask_token_id = self.full_dataset.mask_token_id
        self.pad_token_id = self.full_dataset.pad_token_id
        self.max_length = self.full_dataset.max_length
        # Empirical token-length distribution, used to draw sampling lengths.
        self.length_pool = self.full_dataset.token_lengths

    def decode(self, ids):
        return self.full_dataset.decode(ids)

    def decode_to_smiles(self, ids):
        return self.full_dataset.decode_to_smiles(ids)

    def smiles_from_safe(self, safe_str):
        return safe_to_smiles(safe_str)

    def train_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_pad(batch, self.pad_token_id),
        )


class UniProtSafeDataset(Dataset):
    """SAFE-encoded generic peptides from UniRef50, for pretraining.
    
    pretrain with cond_dropout=1.0 and VectorEmbedder substitutes its learned
    null_embedding for all three conditions (see DiTwithCondition.VectorEmbedder).

    Build the CSV with build_uniprot_corpus.py
    """

    def __init__(self, csv_path, tokenizer_path=TOKENIZER_PATH,
                 max_length=None, limit=None):
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.pad_token_id = self.tokenizer.token2id.get("[PAD]")
        self.mask_token_id = self.tokenizer.token2id.get("[MASK]")
        self.token_dict = self.tokenizer.token2id
        self.num_tokens = len(self.token_dict)
        if self.pad_token_id is None or self.mask_token_id is None:
            raise ValueError("Tokenizer is missing a [PAD] or [MASK] token.")

        df = pd.read_csv(csv_path)
        if 'safe' not in df.columns:
            raise ValueError(f"{csv_path} has no 'safe' column -- build it with "
                             f"build_uniprot_corpus.py")
        df = df.dropna(subset=['safe'])
        df = df[df['safe'].str.len() > 0].drop_duplicates('safe').reset_index(drop=True)
        if limit:
            df = df.head(limit)

        encoded = encode_safe_strings(self.tokenizer, df['safe'].tolist())

        lengths = np.array([len(ids) for ids in encoded])
        # max_length need not match the AMP run: DDitFinalLayer stores seq_length
        # but never builds a shape-dependent parameter, so checkpoints transfer
        # across different values. Passing the AMP value just keeps configs tidy.
        self.max_length = int(lengths.max()) if max_length is None else int(max_length)

        keep = lengths <= self.max_length
        dropped = int((~keep).sum())
        if dropped:
            print(f"[UniProtSafeDataset] dropped {dropped}/{len(keep)} peptides "
                  f"longer than max_length={self.max_length} "
                  f"(longest {int(lengths.max())} tokens)")

        self.sequences = [np.asarray(encoded[i], dtype=np.int64)
                          for i in np.flatnonzero(keep)]
        self.token_lengths = np.array([len(s) for s in self.sequences])
        # One shared zero vector: the conditions are dropped, never read.
        self._null_condition = torch.zeros(CONDITION_DIM, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": torch.from_numpy(self.sequences[idx]),
            "condition": self._null_condition,
        }

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return self.tokenizer.decode(list(ids), skip_special_tokens=True).replace(' ', '')

    def decode_to_smiles(self, ids):
        safe_str = self.decode(ids)
        return safe_str, safe_to_smiles(safe_str)


class UniProtSafeDataModule(L.LightningDataModule):
    """Drop-in replacement for AMPSafeDataModule during pretraining."""

    def __init__(self, csv_path, tokenizer_path=TOKENIZER_PATH,
                 max_length=None, batch_size=16, num_workers=4, limit=None):
        super().__init__()
        self.csv_path = csv_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit = limit
        self.full_dataset = None

    def setup(self, stage=None):
        if self.full_dataset is not None:
            return
        self.full_dataset = UniProtSafeDataset(
            csv_path=self.csv_path, tokenizer_path=self.tokenizer_path,
            max_length=self.max_length, limit=self.limit)

        self.token_dict = self.full_dataset.token_dict
        self.num_tokens = self.full_dataset.num_tokens
        self.mask_token_id = self.full_dataset.mask_token_id
        self.pad_token_id = self.full_dataset.pad_token_id
        self.max_length = self.full_dataset.max_length
        self.length_pool = self.full_dataset.token_lengths

    def decode(self, ids):
        return self.full_dataset.decode(ids)

    def decode_to_smiles(self, ids):
        return self.full_dataset.decode_to_smiles(ids)

    def smiles_from_safe(self, safe_str):
        return safe_to_smiles(safe_str)

    def train_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_pad(batch, self.pad_token_id),
        )