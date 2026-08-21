import re
import pandas as pd
import io
from collections import Counter


TOKENIZER_PATTERN = re.compile(r'\[[^\]]+\]|[%0-9\.\(\)]|[^%0-9\.\(\)\[\]]+')

class OrthogonalSafeTokenizer:
    SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]

    def __init__(self, appearance_number=None):
        self.vocab_counter = Counter()
        # preset fixed topology symbols that are always included in the vocabulary
        self.topo_symbols = {'.', '(', ')', '%', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
        self.appearance_number = appearance_number
        self.breakdown_map = {}
        self.decomp_pool_sorted = []
        self.token2id = {}
        self.id2token = {}
    
    def decompose_c_sequence(self, k):
        """Decompose a sequence of k consecutive uppercase 'C's into allowed C tokens."""
        valid_long = [18, 16, 14, 12, 8]
        res = []
        while k >= 7:
            found = False
            for b in valid_long:
                if b <= k:
                    res.append('C' * b)
                    k -= b
                    found = True
                    break
            if not found:
                break
        if k > 0:
            if k <= 6:
                res.append('C' * k)
            elif k == 7:
                res.append('C' * 6)
                res.append('C')
        return res

    def is_indivisible(self, token):
        """Check if a token cannot be broken down further."""
        if len(token) <= 1:
            return True
        if token.startswith('[') and token.endswith(']'):
            return True
        if token in {'Br', 'Cl'}:
            return True
        if token in self.topo_symbols:
            return True
        return False

    def decompose_token(self, token, decomp_pool_sorted):
        """Decompose a low-frequency token into higher-frequency tokens or basic elements."""
        if self.is_indivisible(token):
            return [token]
        
        res = []
        i = 0
        n = len(token)
        while i < n:
            matched = False
            for sub in decomp_pool_sorted:
                sub_len = len(sub)
                if i + sub_len <= n and token[i:i+sub_len] == sub:
                    res.append(sub)
                    i += sub_len
                    matched = True
                    break
            if not matched:
                res.append(token[i])
                i += 1
        return res

    def tokenize_first_round(self, safe_string):
        tokens = TOKENIZER_PATTERN.findall(safe_string)
        res = []
        for t in tokens:
            if not t:
                continue
            if re.fullmatch(r'c+', t):
                res.extend(list(t))
                continue

            # Check if N appears after more than 5 Cs (e.g., CCCCCCN, CCCCCCC...N)
            m_cn = re.match(r'^(C{6,})(N.*)$', t)
            if m_cn:
                c_part = m_cn.group(1)
                rest_part = m_cn.group(2)
                res.extend(self.decompose_c_sequence(len(c_part)))
                if re.fullmatch(r'C+', rest_part):
                    res.extend(self.decompose_c_sequence(len(rest_part)))
                else:
                    res.append(rest_part)
                continue

            # Check for pure uppercase C sequence
            if re.fullmatch(r'C+', t):
                res.extend(self.decompose_c_sequence(len(t)))
                continue

            res.append(t)
        return res

    def build_vocab(self):
        """Construct deterministic token2id and id2token mapping."""
        self.token2id = {}
        self.id2token = {}
        
        # 1. Special tokens
        for idx, tok in enumerate(self.SPECIAL_TOKENS):
            self.token2id[tok] = idx
            self.id2token[idx] = tok

        current_id = len(self.SPECIAL_TOKENS)

        # 2. Topology tokens
        full_vocab = list(self.vocab_counter.keys())
        topo_tokens = sorted([t for t in full_vocab if t in self.topo_symbols])
        for t in sorted(self.topo_symbols):
            if t not in topo_tokens:
                topo_tokens.append(t)
        
        for t in topo_tokens:
            if t not in self.token2id:
                self.token2id[t] = current_id
                self.id2token[current_id] = t
                current_id += 1

        # 3. Chemistry tokens sorted by frequency descending
        chem_tokens = [t for t, _ in self.vocab_counter.most_common() if t not in self.token2id]
        for t in chem_tokens:
            if t not in self.token2id:
                self.token2id[t] = current_id
                self.id2token[current_id] = t
                current_id += 1

    def fit(self, safe_strings, appearance_number=None, min_freq=None, min_frequency=None):
        if appearance_number is not None:
            self.appearance_number = appearance_number
        elif min_freq is not None:
            self.appearance_number = min_freq
        elif min_frequency is not None:
            self.appearance_number = min_frequency

        # First round tokenization
        first_round_token_lists = [self.tokenize_first_round(s) for s in safe_strings]
        first_counter = Counter()
        for t_list in first_round_token_lists:
            first_counter.update(t_list)

        if self.appearance_number is None or self.appearance_number <= 1:
            self.vocab_counter = first_counter
            self.breakdown_map = {}
            self.build_vocab()
            return

        # Identify high-frequency tokens (freq >= appearance_number)
        high_freq_tokens = {t for t, freq in first_counter.items() if freq >= self.appearance_number}
        basic_tokens = {'C', 'N', 'O', 'S', 'P', 'F', 'I', 'Br', 'Cl', 'c', 'n', 'o', 's', '=', '#', '/', '\\'}
        decomp_pool = high_freq_tokens | basic_tokens | self.topo_symbols
        # Sort candidates longest first, then highest frequency first
        self.decomp_pool_sorted = sorted(list(decomp_pool), key=lambda x: (-len(x), -first_counter.get(x, 0)))

        # Build breakdown_map for low frequency tokens (< appearance_number)
        self.breakdown_map = {}
        for t, freq in first_counter.items():
            if freq < self.appearance_number and not self.is_indivisible(t):
                self.breakdown_map[t] = self.decompose_token(t, self.decomp_pool_sorted)

        # Second round tokenization pass & update vocab_counter
        self.vocab_counter = Counter()
        for t_list in first_round_token_lists:
            second_round_list = []
            for t in t_list:
                if t in self.breakdown_map:
                    second_round_list.extend(self.breakdown_map[t])
                else:
                    second_round_list.append(t)
            self.vocab_counter.update(second_round_list)

        self.build_vocab()

    def tokenize(self, safe_string):
        tokens = self.tokenize_first_round(safe_string)
        res = []
        for t in tokens:
            if t in self.breakdown_map:
                res.extend(self.breakdown_map[t])
            elif t not in self.token2id and not self.is_indivisible(t):
                res.extend(self.decompose_token(t, self.decomp_pool_sorted))
            else:
                res.append(t)
        return res

    def encode(self, safe_string, add_special_tokens=False):
        """Tokenize string and map tokens to IDs."""
        tokens = self.tokenize(safe_string)
        unk_id = self.token2id.get("[UNK]", 4)
        ids = [self.token2id.get(t, unk_id) for t in tokens]
        if add_special_tokens:
            cls_id = self.token2id.get("[CLS]", 1)
            sep_id = self.token2id.get("[SEP]", 2)
            ids = [cls_id] + ids + [sep_id]
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        """Convert token IDs back to a string of tokens."""
        special_ids = {self.token2id[st] for st in self.SPECIAL_TOKENS if st in self.token2id}
        res_tokens = []
        for i in token_ids:
            if skip_special_tokens and i in special_ids:
                continue
            res_tokens.append(self.id2token.get(i, "[UNK]"))
        return "".join(res_tokens)

    def get_vocabulary(self):
        return [token for token, freq in self.vocab_counter.most_common()]

    def save_csv(self, filepath="tokenizer_vocab.csv"):
        """Save the tokenizer vocabulary, IDs, types, and frequencies to a readable CSV file."""
        records = []
        special_set = set(self.SPECIAL_TOKENS)
        
        for tok, idx in sorted(self.token2id.items(), key=lambda x: x[1]):
            if tok in special_set:
                tok_type = "special"
            elif tok in self.topo_symbols:
                tok_type = "topology"
            else:
                tok_type = "chemistry"
            
            freq = self.vocab_counter.get(tok, 0)
            records.append({
                "id": idx,
                "token": tok,
                "type": tok_type,
                "frequency": freq
            })

        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False, encoding="utf-8")
        print(f"Saved vocabulary ({len(df)} tokens) to '{filepath}'.")

    @classmethod
    def load_csv(cls, filepath="tokenizer_vocab.csv"):
        """Load tokenizer from a CSV file, preserving exact token IDs."""
        df = pd.read_csv(filepath, keep_default_na=False)
        tokenizer = cls()
        
        tokenizer.token2id = {}
        tokenizer.id2token = {}
        tokenizer.vocab_counter = Counter()

        decomp_candidates = []

        for _, row in df.iterrows():
            idx = int(row["id"])
            tok = str(row["token"])
            tok_type = str(row["type"])
            freq = int(row["frequency"])

            tokenizer.token2id[tok] = idx
            tokenizer.id2token[idx] = tok
            if freq > 0:
                tokenizer.vocab_counter[tok] = freq

            if tok_type != "special":
                decomp_candidates.append((tok, len(tok), freq))

        # Reconstruct decomp_pool_sorted for handling unseen composite tokens
        # Sort candidates longest first, then highest frequency first
        decomp_candidates.sort(key=lambda x: (-x[1], -x[2]))
        tokenizer.decomp_pool_sorted = [x[0] for x in decomp_candidates]

        print(f"Loaded tokenizer from '{filepath}' with {len(tokenizer.token2id)} tokens.")
        return tokenizer


if __name__ == "__main__":
    import os

    csv_data = "molecular_dataset/dataset/data/safe/amp_safe.csv"

    print("=== Step 1: Fitting OrthogonalSafeTokenizer ===")
    df = pd.read_csv(csv_data)
    tokenizer = OrthogonalSafeTokenizer()
    tokenizer.fit(df['safe'], appearance_number=10)

    full_vocab = tokenizer.get_vocabulary()

    topo_tokens = [t for t in full_vocab if t in tokenizer.topo_symbols]
    chem_tokens = [t for t in full_vocab if t not in tokenizer.topo_symbols]

    print(f"Total Vocabulary Size = {len(full_vocab)}")
    print(f"Total Token Map Size (including special) = {len(tokenizer.token2id)}")
    print(f"Topology Tokens ({len(topo_tokens)}): {sorted(topo_tokens)}")
    print(f"Chemistry Tokens ({len(chem_tokens)}): Top 10 = {chem_tokens[:10]}")

    print("\n=== Step 2: Saving Tokenizer to CSV ===")
    csv_file_path = "tokenizer_vocab.csv"
    tokenizer.save_csv(csv_file_path)

    print("\n=== Step 3: Loading Tokenizer from CSV & Verifying Fixed IDs ===")
    loaded_tokenizer = OrthogonalSafeTokenizer.load_csv(csv_file_path)

    # Verify ID map identity
    assert tokenizer.token2id == loaded_tokenizer.token2id, "ERROR: token2id mismatch after load!"
    assert tokenizer.id2token == loaded_tokenizer.id2token, "ERROR: id2token mismatch after load!"
    print("SUCCESS: Token IDs are 100% identical between saved and loaded tokenizers!")

    print("\n=== Step 4: Testing Encoding & Decoding ===")
    test_cases = [
        # SAFE representation strings
        "C%13(=O)[C@@H]%29CCC(N)=O.C%24(=O)[C@@H]%30CCCCN.C%64(=O)[C@@H]%34CCCCN.[C@H]%45(CCC(=O)O)C%12=O.[C@H]%47(CCCCN)C%14=O.[C@H]%50(CCC(=O)O)C%17=O.[C@H]%51(CCCCN)C%18=O.[C@H]%56(CCCCN)C%23=O.CC[C@H](C)[C@H]%27C7=O.C2(=O)[C@@H]%28CC(C)C.C%65(=O)[C@@H]%36[C@@H](C)CC.C%66(=O)[C@@H]%37[C@@H](C)CC.C3(=O)[C@@H]%38CC(=O)O.[C@@H]%48(C%15=O)[C@@H](C)CC.[C@H]%54(CC(C)C)C%21=O.[C@H]%55(CC(N)=O)C%22=O.[C@H]%58(CC(=O)O)C%25=O.[C@H]%60(CC(N)=O)C(=O)O.C5(=O)[C@@H]%40C(C)C.[C@H]%44(CC%61)C%11=O.[C@@H]%52(C%19=O)C(C)C.C4(=O)[C@@H]%39C%63.c1%63ccccc1.[C@H]%42(C%62)C9=O.C%46(=O)[C@@H]%32C.[C@H]%41(C)C8=O.c1%62cnc[nH]1.[C@H]%43(C)C%10=O.[C@H]%49(C)C%16=O.C%35(=O)C%31.C%57(=O)C%33.C6(=O)CN.C%53C%20=O.C%59C%26=O.N2%27.N%13%28.N%24%29.N%35%30.N%46%31.N%57%32.N%64%33.N%65%34.N%66%36.N3%37.N4%38.N5%39.N6%40.N7%41.N8%42.N9%43.N%10%44.S%61C.N%11%45.N%12%47.N%14%48.N%15%49.N%16%50.N%17%51.N%18%52.N%19%53.N%20%54.N%21%55.N%22%56.N%23%58.N%25%59.N%26%60",
        "[C@@H]1%40NC(=O)CNC(=O)[C@H]%41NC(=O)[C@@H]2CSSC[C@H]%36C(=O)N3CCC[C@H]3C(=O)N[C@@H]%42C(=O)N[C@@H]%43C(=O)N[C@@H]%44C(=O)NCC(=O)N[C@@H](C)C(=O)N[C@H]3CSSC[C@H](NC(=O)[C@H]%45NC(=O)[C@H]%46NC(=O)[C@H]%47NC(=O)[C@H]%49NC1=O)C(=O)N[C@@H]%50C(=O)N[C@H]%39CSSC[C@H](NC(=O)[C@H]%51NC(=O)[C@H]%52NC(=O)[C@H]%53NC3=O)C(=O)N[C@@H]%54C(=O)N[C@@H]%55C(=O)N[C@@H]%56C(=O)N[C@@H]%57C(=O)N[C@@H]%58C(=O)N[C@@H]%60C(=O)NCC(=O)NCC(=O)N[C@@H]%61C(=O)N2.C%37(=O)[C@@H]%20CCCN=C(N)N.C%48(=O)[C@@H]%21CCCN=C(N)N.C%68(=O)[C@@H]%23CCCN=C(N)N.[C@H]%34(CCCN=C(N)N)C%16=O.C7(=O)[C@@H]%29CCC(=O)O.C9(=O)[C@@H]%31CCC(=O)O.C%69(=O)[C@@H]%24CC(C)C.[C@H]%35(CC(N)=O)C(=O)O.C%59(=O)[C@@H]%22C(C)C.C%12(=O)[C@@H](N)C(C)C.c1%66ccc(O)cc1.C%54CCN=C(N)N.C%57CCN=C(N)N.C%58CCN=C(N)N.C%60CCN=C(N)N.c1%67ccc(O)cc1.C%15(=O)[C@@H]%18C%65.c1%65ccccc1.C%70(=O)[C@@H]%25C%62.C8(=O)[C@@H]%30CO.[C@@H]1%38CCCN1%12.[C@H]%33(C%66)C%14=O.c1%62cnc[nH]1.C5(=O)[C@@H]%27C.C6(=O)[C@@H]%28C.C%10(=O)[C@@H]%32C.C%44CC(N)=O.C%46CC(N)=O.C%47CCCN.c1%63cnc[nH]1.c1%64cnc[nH]1.CC[C@@H]%40C.C4(=O)C%17.C%26(=O)C%19.C%42C(C)C.C%43C(N)=O.[C@H]%49(C)CC.C%52C(N)=O.[C@H]%56(C)CC.C%11%38=O.[C@H]%45(C)O.[C@H]%50(C)O.C%13%39=O.C%41O.N4%36.N%15%17.N%26%18.N%37%19.N%48%20.N%59%21.N%68%22.N%69%23.N%70%24.N5%25.N6%27.N7%28.N8%29.N9%30.N%10%31.N%11%32.N%13%33.N%14%34.N%16%35.C%51%63.C%53%64.C%55O.C%61%67",
        "C2(=O)[C@@H]8CCCN=C(N)N.C%13(=O)[C@@H](N)CCCN=C(N)N.[C@H]9(CCCN=C(N)N)C%22=O.[C@H]%10(CCCN=C(N)N)C%24=O.[C@H]%11(CCCN=C(N)N)C%27=O.[C@H]%14(CCCN=C(N)N)C5=O.CC[C@H](C)[C@H]7C%21=O.[C@H]%12(CC(C)C)C3=O.N1%22CCC[C@H]1%16.N1%24CCC[C@H]1%17.N1%25CCC[C@H]1%18.N13CCC[C@H]1%19.N15CCC[C@H]1%20.[C@H]%15(CS)C(=O)O.C%23%16=O.C%25%17=O.C%26%18=O.C4%19=O.C6%20=O.N27.N%138.N%219.N%23%10.N%26%11.N%27%12.N4%14.N6%15",
    ]

    for idx, safe_str in enumerate(test_cases, 1):
        tokens_orig = tokenizer.tokenize(safe_str)
        ids_orig = tokenizer.encode(safe_str)
        decoded_orig = tokenizer.decode(ids_orig)

        tokens_loaded = loaded_tokenizer.tokenize(safe_str)
        ids_loaded = loaded_tokenizer.encode(safe_str)
        decoded_loaded = loaded_tokenizer.decode(ids_loaded)

        assert ids_orig == ids_loaded, f"ERROR: Encoded ID mismatch for test case {idx}!"
        assert tokens_orig == tokens_loaded, f"ERROR: Token list mismatch for test case {idx}!"

        print(f"\nTest Case {idx}: '{safe_str}'")
        print(f"  Tokens: {tokens_loaded}")
        print(f"  Encoded IDs: {ids_loaded}")
        print(f"  Decoded String: '{decoded_loaded}'")