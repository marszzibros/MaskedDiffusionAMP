
import argparse
import os
from pathlib import Path

import pandas as pd

from safe.converter import SAFEConverter


def encode_smiles(smiles, converter, allow_empty=True):
    """
    Convert one SMILES to a SAFE string.
    """
    try:
        return converter.encoder(smiles, allow_empty=allow_empty)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert amp.csv SMILES to SAFE.")
    parser.add_argument("--in-path", default="dataset/data/dbaasp/amp.csv")
    parser.add_argument("--out-path", default="dataset/data/safe/amp_safe.csv")
    parser.add_argument("--log-path", default="log/safe/encode_failures.log")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--id-col", default="amp_id")
    # The slicer decides where molecules fragment, which sets the ring-closure
    # structure the tokenizer then has to encode -- it matters more than any
    # BPE hyperparameter. "brics" is a general drug-like fragmenter; "recap"
    # and "hr" cut amide bonds, i.e. peptide residue boundaries.
    parser.add_argument("--slicer", default="brics",
                        choices=SAFEConverter.SUPPORTED_SLICERS)
    args = parser.parse_args()

    # run relative to the project root so default paths resolve consistently
    os.chdir(Path(__file__).resolve().parents[1])

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    df = pd.read_csv(args.in_path)
    converter = SAFEConverter(slicer=args.slicer)  # reuse one converter across all rows

    rows = []
    failures = []
    for amp_id, smiles in zip(df[args.id_col], df[args.smiles_col]):
        if not isinstance(smiles, str) or not smiles.strip():
            failures.append((amp_id, "empty smiles", smiles))
            continue
        safe_str = encode_smiles(smiles, converter=converter)
        if safe_str is None:
            failures.append((amp_id, "encode failed", smiles))
            continue
        rows.append({"amp_id": amp_id, "smiles": smiles, "safe": safe_str})

    pd.DataFrame(rows, columns=["amp_id", "smiles", "safe"]).to_csv(args.out_path, index=False)
    with open(args.log_path, "w") as fh:
        for amp_id, reason, smiles in failures:
            fh.write(f"{amp_id}\t{reason}\t{smiles}\n")

    print(f"encoded {len(rows)} / {len(df)} peptides with slicer={args.slicer} "
          f"({len(failures)} failed -> {args.log_path})")
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
