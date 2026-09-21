import re
import safe as sf
from rdkit import Chem

# Supports: %(100), %100 (non-standard 3-digit), %10, and 0-9
LINKAGE_PATTERN = re.compile(r"%\([0-9]+\)|%[0-9]{2}|[0-9]")


def _map_bracket_interiors(safe_str: str):
    in_bracket = [False] * len(safe_str)
    bracket_stack = []
    errors = []

    for idx, char in enumerate(safe_str):
        if char == "[":
            bracket_stack.append(idx)
        elif char == "]":
            if not bracket_stack:
                errors.append(
                    {
                        "pos": idx,
                        "token": "]",
                        "desc": "Unexpected closing bracket ']' with no opening partner.",
                    }
                )
            else:
                open_idx = bracket_stack.pop()
                for i in range(open_idx, idx + 1):
                    in_bracket[i] = True

    for idx in bracket_stack:
        errors.append(
            {"pos": idx, "token": "[", "desc": "Unclosed opening bracket '['."}
        )

    return in_bracket, errors


def find_dangling_ring_bonds(safe_str: str, in_bracket):
    """Accurately maps open/close states per linkage token,

    detecting odd occurrences and illegal overlapping reuse.
    """
    token_records = {}  # canonical_label -> list of (start, end, raw_token, fragment_idx)
    fragment_idx = 0

    i = 0
    n = len(safe_str)
    while i < n:
        if in_bracket[i]:
            i += 1
            continue

        char = safe_str[i]
        if char == ".":
            fragment_idx += 1
            i += 1
            continue

        match = LINKAGE_PATTERN.match(safe_str, i)
        if not match:
            i += 1
            continue

        raw_token = match.group()
        start, end = match.start(), match.end()
        i = end

        canonical_label = raw_token.replace("(", "").replace(")", "")
        token_records.setdefault(canonical_label, []).append(
            (start, end, raw_token, fragment_idx)
        )

    errors = []
    for label, records in token_records.items():
        count = len(records)
        # Linkages in SAFE/SMILES must strictly appear an even number of times (pairs)
        if count % 2 != 0:
            last_start, last_end, last_tok, last_frag = records[-1]
            errors.append(
                {
                    "pos": last_start,
                    "token": last_tok,
                    "fragment": last_frag,
                    "desc": (
                        f"Broken linkage: Label '{label}' occurs {count} time(s) (odd number). "
                        f"Last unclosed instance is here in fragment #{last_frag}."
                    ),
                }
            )

    return errors


def find_safe_syntax_and_linkage_errors(safe_str: str):
    errors = []

    # 1. Mask %(...) linkage parentheses so they don't break branch checking
    linkage_paren_spans = set()
    for match in re.finditer(r"%\([0-9]+\)", safe_str):
        linkage_paren_spans.add(match.start() + 1)
        linkage_paren_spans.add(match.end() - 1)

    # 2. Check Branching Parentheses '(' and ')'
    paren_stack = []
    for idx, char in enumerate(safe_str):
        if idx in linkage_paren_spans:
            continue

        if char == "(":
            paren_stack.append(idx)
        elif char == ")":
            if not paren_stack:
                errors.append(
                    {
                        "pos": idx,
                        "token": ")",
                        "desc": "Unexpected closing branching parenthesis ')' with no opening partner.",
                    }
                )
            else:
                paren_stack.pop()

    for idx in paren_stack:
        errors.append(
            {
                "pos": idx,
                "token": "(",
                "desc": "Unclosed branching parenthesis '('.",
            }
        )

    # 3. Check Square Brackets
    in_bracket, bracket_errors = _map_bracket_interiors(safe_str)
    errors.extend(bracket_errors)

    # 4. Ring / Linkage Bonds
    errors.extend(find_dangling_ring_bonds(safe_str, in_bracket))

    return sorted(errors, key=lambda x: x["pos"] if x["pos"] is not None else -1)


class ValenceLocalizationError(Exception):
    pass


# Comprehensive pattern including elements, aromatic tokens, and wildcards
ATOM_TOKEN_PATTERN = re.compile(
    r"\[[^\]]+\]|Br|Cl|Si|Se|Na|Mg|Al|Ca|Fe|Zn|B|C|N|O|P|S|F|I|b|c|n|o|p|s|\*"
)


def map_atom_positions(safe_str: str, in_bracket):
    positions = []
    i = 0
    n = len(safe_str)
    while i < n:
        if in_bracket[i] and safe_str[i] != "[":
            i += 1
            continue

        match = ATOM_TOKEN_PATTERN.match(safe_str, i)
        if match:
            positions.append((match.start(), match.group()))
            i = match.end()
        else:
            i += 1
    return positions


def pinpoint_chemical_valence_errors(safe_str: str):
    in_bracket, bracket_errors = _map_bracket_interiors(safe_str)
    if bracket_errors:
        raise ValenceLocalizationError(
            "Unbalanced square brackets; fix syntax before valence analysis."
        )

    atom_positions = map_atom_positions(safe_str, in_bracket)
    mol = Chem.MolFromSmiles(safe_str, sanitize=False)
    if mol is None:
        raise ValenceLocalizationError(
            "RDKit could not construct molecule graph even with sanitize=False."
        )

    if mol.GetNumAtoms() != len(atom_positions):
        raise ValenceLocalizationError(
            f"Atom count mismatch: Tokenizer detected {len(atom_positions)} atoms, "
            f"but RDKit constructed {mol.GetNumAtoms()}. Cannot map offsets accurately."
        )

    valence_errors = []
    for idx, atom in enumerate(mol.GetAtoms()):
        pos, token = atom_positions[idx]
        try:
            atom.UpdatePropertyCache(strict=True)
        except Chem.rdchem.MolSanitizeException as e:
            valence_errors.append(
                {
                    "pos": pos,
                    "token": token,
                    "desc": f"{type(e).__name__}: {e}",
                }
            )

    if not valence_errors:
        kekule_error = _check_molecule_level(mol)
        if kekule_error:
            valence_errors.append(kekule_error)

    # Safe sort: Handle None pos by mapping to -1
    return sorted(
        valence_errors, key=lambda x: x["pos"] if x["pos"] is not None else -1
    )


def _check_molecule_level(mol):
    probe = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(probe)
    except Chem.rdchem.KekulizeException as e:
        return {
            "pos": None,
            "token": "(molecule)",
            "desc": f"Kekulization failed: {e}",
        }
    except Chem.rdchem.MolSanitizeException as e:
        return {
            "pos": None,
            "token": "(molecule)",
            "desc": f"{type(e).__name__}: {e}",
        }
    return None


def print_diagnostic_pointer(safe_str: str, error_item: dict):
    pos = error_item["pos"]
    tok = error_item["token"]
    desc = error_item["desc"]

    print(f"Error at index {pos} (token '{tok}'):")
    print(f"  {desc}")
    if pos is not None:
        # Truncate context window for very long peptide strings
        start_win = max(0, pos - 30)
        end_win = min(len(safe_str), pos + 40)
        snippet = safe_str[start_win:end_win]
        caret_offset = pos - start_win

        prefix = "..." if start_win > 0 else ""
        suffix = "..." if end_win < len(safe_str) else ""

        print(f"  Snippet: {prefix}{snippet}{suffix}")
        pointer = " " * (len(prefix) + caret_offset + 11) + "^" * len(tok)
        print(f"{pointer}")
    print()


def locate_safe_errors(safe_str: str):
    print("=" * 80)
    print("DIAGNOSING STRING")
    print("=" * 80)

    # 1. Structural Linkage & Parenthesis Check
    structural_issues = find_safe_syntax_and_linkage_errors(safe_str)
    if structural_issues:
        print(f"[FOUND {len(structural_issues)} SYNTAX/LINKAGE ISSUE(S)]\n")
        for err in structural_issues:
            print_diagnostic_pointer(safe_str, err)
        return

    # 2. Chemical Valence Checks
    try:
        chem_issues = pinpoint_chemical_valence_errors(safe_str)
    except ValenceLocalizationError as e:
        print(f"[COULD NOT LOCALIZE VALENCE ERRORS]: {e}\n")
        return

    if chem_issues:
        print(f"[FOUND {len(chem_issues)} CHEMICAL/VALENCE ISSUE(S)]\n")
        for err in chem_issues:
            print_diagnostic_pointer(safe_str, err)
        return

    # 3. SAFE Library Canonical Decoding Check
    try:
        smiles = sf.decode(
            safe_str, canonical=True, fix=False, ignore_errors=False
        )
        print("Sequence is valid!")
        print(f"SMILES: {smiles}")
    except Exception as e:
        print(f"SAFE decoding error: {e}")


if __name__ == "__main__":
    locate_safe_errors("CCCCCCCC1=O.N12.[C@H]2(C3=O)C(C)C.N34.[C@@H]4(CCN)C5=O.N56.C6C7=O.N78.[C@@H]8(CO)C9=O.N9%10.[C@@H]%10(C%11)C%12=O.c%13%11cnc[nH]%13.N%13%13CCC[C@H]%13%14.c%13%15ccccc%13.[C@H]%16(C%15)C%17=O.N%18%16.[C@@H]%19(CCN)C%18=O.N%20%19.C%20%14=O.N%17%21.[C@H]%21(C%22=O)C(C)C.N%22%23.[C@H]%23(CCC(=O)O)C%24=O.N%24%25.[C@H]%25(C)C%26=O.N%26%27.C%27C%28=O.N%28%29.C%29(CCCCN)C(N%29%30.[C@H]%30(C)C(=O)O")