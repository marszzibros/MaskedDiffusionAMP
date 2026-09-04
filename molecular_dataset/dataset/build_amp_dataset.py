"""
Turn dbaasp_new_info.csv into a linked (relational) AMP dataset.

Outputs two CSVs joined by `amp_id`:

  amp.csv           one row per peptide: amp_id, name, sequence, smiles,
                    nTerminus, cTerminus, targetGroups, targetObjects, unusuals
  amp_activity.csv  one row per (amp_id, species): the MIC averaged across all
                    strains/replicates of that species, in ug/ml, plus how many
                    raw measurements were averaged.

Species are normalized to "Genus species" so different strains of the same
species (e.g. "Bacillus cereus ATCC 11778") collapse to one ("Bacillus cereus")
and are averaged together.
"""

import argparse
import ast
import os
import re
from pathlib import Path

import pandas as pd

# Which activity measures count as "MIC". Plain MIC dominates the data; add
# "MIC50"/"MIC90" here if you want the MIC family folded in too.
MIC_MEASURES = {"MIC"}

# Only keep MIC activity for these six species (compared case-insensitively on
# the strain-normalized "genus species" name).
TARGET_SPECIES = [
    "escherichia coli",
    "pseudomonas aeruginosa",
    "klebsiella pneumoniae",
    "staphylococcus aureus",
    "bacillus subtilis",
    "staphylococcus epidermidis",
]
_TARGET_SPECIES_SET = set(TARGET_SPECIES)

# Canonical target categories. Any raw value not listed here maps to "OTHER".
# Raw DBAASP values match these once upper-cased (e.g. "Gram+" -> "GRAM+").
KEEP_GROUPS = {"GRAM-", "GRAM+", "MAMMALIAN CELL", "FUNGUS"}
KEEP_OBJECTS = {"LIPID BILAYER", "DNA / RNA", "CYTOPLASMIC PROTEIN", "MEMBRANE PROTEIN"}


def canonicalize_categories(cell, keep_set):
    """Map a stringified list of raw categories to the canonical set.

    Each raw value is upper-cased; anything outside `keep_set` becomes "OTHER".
    Returns a de-duplicated, sorted list.
    """
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        raw = ast.literal_eval(cell)
    except Exception:
        return []
    out = {(v.upper() if v.upper() in keep_set else "OTHER") for v in raw}
    return sorted(out)


def parse_activities(cell):
    """Parse the stringified list-of-dicts in targetActivities.

    Handles the numpy scalar reprs (e.g. np.float32(13.0)) that pandas wrote
    out, which plain ast.literal_eval cannot evaluate.
    """
    if not isinstance(cell, str) or not cell.strip():
        return []
    cell = re.sub(r"np\.float\d*\(([^)]*)\)", r"\1", cell)
    try:
        value = ast.literal_eval(cell)
    except Exception:
        return []
    return value if isinstance(value, list) else []


def normalize_species(name):
    """Collapse a strain-qualified species name to 'Genus species'.

    "Bacillus cereus ATCC 11778"                    -> "Bacillus cereus"
    "Salmonella enterica subsp. enterica ... 14028" -> "Salmonella enterica"
    A one- or two-token name (already bare) is returned unchanged.
    """
    name = " ".join(str(name).split())  # collapse whitespace
    tokens = name.split(" ")
    if len(tokens) <= 2:
        return name
    return tokens[0] + " " + tokens[1]


def main():
    parser = argparse.ArgumentParser(description="Build linked AMP dataset from DBAASP species CSV.")
    parser.add_argument("--in-path", default="dataset/data/dbaasp/dbaasp_new_info.csv")
    parser.add_argument("--amp-path", default="dataset/data/dbaasp/amp.csv")
    parser.add_argument("--activity-path", default="dataset/data/dbaasp/amp_activity.csv")
    args = parser.parse_args()

    # run relative to the project root so default paths resolve consistently
    os.chdir(Path(__file__).resolve().parents[1])

    df = pd.read_csv(args.in_path).reset_index(drop=True)
    df.insert(0, "amp_id", df.index)

    # collapse target groups/objects to the canonical category sets (-> OTHER)
    df["targetGroups"] = df["targetGroups"].apply(lambda c: canonicalize_categories(c, KEEP_GROUPS))
    df["targetObjects"] = df["targetObjects"].apply(lambda c: canonicalize_categories(c, KEEP_OBJECTS))

    # peptide table (structure + metadata; carries the SMILES)
    peptide_cols = [
        "amp_id", "name", "sequence", "smiles",
        "nTerminus", "cTerminus", "targetGroups", "targetObjects", "unusuals",
    ]
    df[peptide_cols].to_csv(args.amp_path, index=False)

    # activity table: average MIC per target species, per peptide
    records = []
    dropped_nonpositive = 0
    for amp_id, cell in zip(df["amp_id"], df["targetActivities"]):
        buckets = {}  # target species -> list of MIC values
        for act in parse_activities(cell):
            if act.get("species_measure") not in MIC_MEASURES:
                continue
            species = normalize_species(act.get("species_name", "")).lower()
            if species not in _TARGET_SPECIES_SET:  # keep only the six species
                continue
            try:
                conc = float(act.get("concentration"))
            except (TypeError, ValueError):
                continue
            if not conc > 0:  # drop 0 / negative / NaN as invalid MICs
                dropped_nonpositive += 1
                continue
            buckets.setdefault(species, []).append(conc)
        for species, values in buckets.items():
            records.append({
                "amp_id": amp_id,
                "species": species,
                "mic_ug_per_ml": sum(values) / len(values),
                "n_measurements": len(values),
            })

    activity = pd.DataFrame(records, columns=["amp_id", "species", "mic_ug_per_ml", "n_measurements"])
    activity.to_csv(args.activity_path, index=False)

    # summary
    print(f"peptides (amp.csv):            {len(df)}")
    print(f"activity rows (amp_activity):  {len(activity)}")
    print(f"peptides with >=1 target MIC:  {activity['amp_id'].nunique()}")
    print(f"dropped non-positive MIC values: {dropped_nonpositive}")
    print("\nMIC coverage per target species (peptides):")
    counts = activity.groupby("species")["amp_id"].nunique()
    for sp in TARGET_SPECIES:
        print(f"  {counts.get(sp, 0):5d}  {sp}")


if __name__ == "__main__":
    main()
