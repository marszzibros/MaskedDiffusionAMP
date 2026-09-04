"""Make an APETokenizer look like the SAFE/HuggingFace tokenizer dataset.py expects.

dataset.py asks its tokenizer for five things: `pad_token_id`, `mask_token_id`,
`get_vocab()`, a batch `__call__` returning {"input_ids": [[...], ...]}, and
`decode(ids, skip_special_tokens=True)`. APETokenizer has none of them in that
shape -- its `__call__` takes one string at a time and it has no `decode`.

It also fixes the encode cost. APETokenizer.encode looks for the longest
vocabulary entry at each position by scanning `for j in range(len(text), i, -1)`
-- from the end of the *whole string* downward. On a 1300-character SAFE string
that is ~845,000 substring lookups per molecule, or 17 billion over this corpus.
No vocabulary entry is longer than the longest one, so the scan can start there
instead; `assert_matches_reference` below checks the two agree token for token.

    from ape.ape_adapter import APESafeTokenizer
    tok = APESafeTokenizer("ape/vocab_1024.json")
    ids = tok(["CCO", "CCN"], add_special_tokens=True)["input_ids"]
    tok.decode(ids[0])
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ape_tokenizer import APETokenizer


class APESafeTokenizer:
    def __init__(self, vocab_path):
        self.tok = APETokenizer()
        self.tok.load_vocabulary(vocab_path)
        self.vocab = self.tok.vocabulary
        self.inv = {i: t for t, i in self.vocab.items()}
        self.max_token_len = max(len(t) for t in self.vocab)
        self.specials = set(self.tok.special_tokens)

    # ---- the bits dataset.py reads ------------------------------------

    @property
    def pad_token_id(self):
        return self.tok.pad_token_id

    @property
    def mask_token_id(self):
        return self.tok.mask_token_id

    def get_vocab(self):
        return dict(self.vocab)

    def __len__(self):
        return len(self.vocab)

    # ---- encoding ------------------------------------------------------

    def encode_one(self, text, add_special_tokens=False):
        """Greedy longest match, same rule as APETokenizer.encode, bounded by the
        longest vocabulary entry instead of by the length of the input."""
        vocab, out = self.vocab, []
        if add_special_tokens:
            out.append(self.tok.bos_token_id)
        i, n = 0, len(text)
        while i < n:
            hi = min(n, i + self.max_token_len)
            for j in range(hi, i, -1):
                piece = text[i:j]
                if piece in vocab:
                    out.append(vocab[piece])
                    i = j
                    break
            else:
                out.append(vocab[self.tok.unk_token])
                i += 1
        if add_special_tokens:
            out.append(self.tok.eos_token_id)
        return out

    def __call__(self, text, add_special_tokens=False, padding=False,
                 truncation=False, max_length=None, return_attention_mask=False,
                 **_):
        one = isinstance(text, str)
        texts = [text] if one else list(text)
        ids = [self.encode_one(t, add_special_tokens) for t in texts]
        if truncation and max_length:
            ids = [x[:max_length] for x in ids]
        if padding and max_length:
            pad = self.pad_token_id
            ids = [x + [pad] * (max_length - len(x)) for x in ids]
        out = {"input_ids": ids[0] if one else ids}
        if return_attention_mask:
            masks = [[int(t != self.pad_token_id) for t in x] for x in ids]
            out["attention_mask"] = masks[0] if one else masks
        return out

    # ---- decoding ------------------------------------------------------

    def decode(self, ids, skip_special_tokens=True):
        toks = []
        for i in ids:
            t = self.inv.get(int(i))
            if t is None or (skip_special_tokens and t in self.specials):
                continue
            toks.append(t)
        return "".join(toks)

    def convert_ids_to_tokens(self, ids):
        return [self.inv.get(int(i), self.tok.unk_token) for i in ids]

    # ---- checks --------------------------------------------------------

    def assert_matches_reference(self, texts):
        """The bounded scan must give the same ids as the upstream unbounded one."""
        for t in texts:
            mine = self.encode_one(t)
            theirs = self.tok.encode(t, add_special_tokens=False)
            if mine != theirs:
                raise AssertionError(
                    f"bounded scan disagrees with APETokenizer.encode on:\n  {t[:80]}")
        return True

    def assert_roundtrip(self, texts):
        for t in texts:
            back = self.decode(self.encode_one(t))
            if back != t:
                raise AssertionError(f"round trip lost information:\n  in  {t[:80]}\n  out {back[:80]}")
        return True
