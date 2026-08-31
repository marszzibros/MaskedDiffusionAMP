import re
import io
import safe as sf
from rdkit import Chem, RDLogger

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

def parse_fragment(frag_str):
    # 匹配: 1. [中括号内容]  2. %两位数  3. 单数字  4. 其他纯文本
    pattern = re.compile(r'(\[[^\]]+\])|(%\d{2})|([0-9])|([^\[\]%0-9]+)')
    parts = []
    for match in pattern.finditer(frag_str):
        if match.group(1):
            parts.append(('text', match.group(1)))
        elif match.group(2):
            parts.append(('anchor', match.group(2)))
        elif match.group(3):
            parts.append(('anchor', match.group(3)))
        elif match.group(4):
            parts.append(('text', match.group(4)))
    return parts

def reorder_and_reindex_safe(safe_str):
    frags = safe_str.split('.')
    parsed_frags = [parse_fragment(f) for f in frags]
    
    anchor_to_frags = {}
    for idx, parts in enumerate(parsed_frags):
        for ptype, pval in parts:
            if ptype == 'anchor':
                if pval not in anchor_to_frags:
                    anchor_to_frags[pval] = []
                anchor_to_frags[pval].append(idx)
                
    visited_frags = set()
    new_frags = []
    anchor_map = {}      # 记录: 旧锚点 -> 新锚点
    next_anchor_id = 1   # 新锚点从 1 开始严格递增
    
    def format_anchor(num):
        return str(num) if num < 10 else f"%{num}"
        
    def dfs(frag_idx):
        nonlocal next_anchor_id
        visited_frags.add(frag_idx)
        
        parts = parsed_frags[frag_idx]
        new_frag_str = ""
        outgoing_anchors = []
        
        for ptype, pval in parts:
            if ptype == 'anchor':
                if pval not in anchor_map:
                    anchor_map[pval] = format_anchor(next_anchor_id)
                    next_anchor_id += 1
                new_frag_str += anchor_map[pval]
                outgoing_anchors.append(pval)
            else:
                new_frag_str += pval
                
        new_frags.append(new_frag_str)
        
        # 3. 顺藤摸瓜：沿着锚点去找下一个相连的片段
        for old_anchor in outgoing_anchors:
            neighbors = anchor_to_frags.get(old_anchor, [])
            for neighbor_idx in neighbors:
                if neighbor_idx not in visited_frags:
                    dfs(neighbor_idx)

    # 4. 从第 0 个片段 (通常是 N 端修饰或第一个残基) 开始遍历
    for i in range(len(frags)):
        if i not in visited_frags:
            dfs(i)
            
    return ".".join(new_frags)


original_safe = "[C@@H]15NC(=O)[C@@H]2CSSC[C@@H]%28NC(=O)[C@H]6NC(=O)[C@H](CSSC[C@H]%27C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H]7C(=O)N2)NC(=O)[C@H]8NC(=O)[C@H]9NC(=O)[C@H]%10NC(=O)CNC(=O)[C@H]%11NC(=O)[C@H]%12NC(=O)[C@H]%13NC(=O)CNC(=O)CNC(=O)[C@H]%14NC(=O)[C@H]%16NC(=O)CNC(=O)[C@H]%17NC(=O)[C@H]%18NC1=O.[C@H]%24(CCCN=C(N)N)C(=O)O.C%23(=O)[C@@H](N)CC(=O)O.C9CCN=C(N)N.c1%20ccc(O)cc1.C%16CCN=C(N)N.C%17CCN=C(N)N.c1%21ccccc1.C%15(=O)[C@@H]%25C.C%22(=O)[C@@H]%26C.c1%19cnc[nH]1.C%10CCCN.CC(C)C5.C%11C(N)=O.C4%28=O.C6(C)C.C8(C)C.N4%24.N%15%27.N%22%25.N%23%26.C7%19.C%12S.C%13%20.C%14O.C%18%21"

canonical_safe = reorder_and_reindex_safe(original_safe)

print("Original SAFE:")
print(original_safe)
print("\n" + "-"*50 + "\n")
print("Reordered and Reindexed SAFE:")
print(canonical_safe)