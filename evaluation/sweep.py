"""Valid and connected rates for every cell of a sampling sweep.

    python evaluation/sweep.py xh_dfs_single_seed0 xh_raw_single      # run names under output/
    python evaluation/sweep.py output/xh_dfs_single_seed0 --out results/valid_connected.csv

valid       share of samples whose SAFE string decodes to a molecule (convertible to SMILES)
connected   share of samples that decode to one connected molecule (valid, and a single piece)

Both are shares of all samples in the file, so connected <= valid. They are the functions run.py reports.
A run folder holds sweep folders (sweep/, sweep_eta/, sweep_matched/), each with one CSV per sampling cell:
steps<S>_eta<E>[_nogrammar].csv or steps<S>_t<T>_c<C>_b<B>[_nogrammar].csv. `repair` is off when the name says nogrammar.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))     # the metric files sit beside this one

from connectivity import connected                           # noqa: E402
from loading import load_samples                             # noqa: E402
from validity import validity                                # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
CELL = re.compile(r"steps(\d+)_(.+?)(_nogrammar)?$")


def run_dir(arg):
    path = Path(arg)
    if path.is_dir():
        return path
    if (OUTPUT / arg).is_dir():
        return OUTPUT / arg
    raise SystemExit(f"{arg}: not a folder, and not a run under {OUTPUT}")


def cells(run):
    """(sweep folder name, csv path) for every sampling cell in a run folder."""
    for path in sorted(run.glob("sweep*/steps*.csv")):
        yield path.parent.name, path


def score_run(run):
    rows = []
    for sweep, path in cells(run):
        steps, eta, off = CELL.fullmatch(path.stem).groups()
        samples = load_samples(path)
        rows.append({
            "run": run.name, "sweep": sweep, "steps": int(steps), "eta": eta, "repair": "off" if off else "on",
            "n": len(samples), "valid": validity(samples), "connected": connected(samples),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders, or names under output/")
    ap.add_argument("--out", default=None, help="write every cell to this CSV")
    args = ap.parse_args()

    rows = [row for arg in args.runs for row in score_run(run_dir(arg))]
    if not rows:
        raise SystemExit("no sweep cells found (expected <run>/sweep*/steps*.csv)")
    df = pd.DataFrame(rows)
    df["_eta"] = df["eta"].str.extract(r"(\d+)", expand=False).astype(int)     # first number: eta, or topology eta
    df = df.sort_values(["run", "sweep", "steps", "_eta", "repair"]).drop(columns="_eta")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, float_format="%.4f")
    short = df.groupby("run")[["valid", "connected"]].agg(["mean", "min", "max"])
    with pd.option_context("display.width", 200, "display.max_rows", None, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
        print("\nacross cells")
        print(short)
    if (df["n"] != df["n"].iloc[0]).any():
        print(f"\nwarning: cells differ in sample count: {sorted(df['n'].unique())}")


if __name__ == "__main__":
    main()
