"""Collect every cell of the 3-seed sweep into one tidy CSV.

    python scripts/seed_sweep/collect.py                         # -> results/seed_sweep.csv
    python scripts/seed_sweep/collect.py --out /tmp/cells.csv
    python scripts/seed_sweep/collect.py --summary               # also print the arm table

Reads the .txt summaries that arm_job.sh wrote, so it needs no GPU and no RDKit.
Run directory names are <tokenizer>_<order>_<schedule>_seed<N>, sweep folders are sweep_seed<N>,
and cells are steps<S>_b<B> (single) or steps<S>_t<T>_c<C>_b<B> (two), with _nogrammar when the
grammar repair was off.
"""
import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RUN = re.compile(r"^(?P<tokenizer>amide|xh)_(?P<order>dfs|raw)_(?P<schedule>single|two)_seed(?P<tseed>\d+)$")
CELL = re.compile(r"^steps(?P<steps>\d+)_(?P<etas>.+?)(?P<nogram>_nogrammar)?$")
ETA3 = re.compile(r"^t(?P<t>\d+)_c(?P<c>\d+)_b(?P<b>\d+)$")
ETA1 = re.compile(r"^b(?P<b>\d+)$")
NUM = re.compile(r"^\s{2}(\w+)\s+([\d.]+)\s*$")
FRAG = re.compile(r"^\s{2}fragments\s+mean ([\d.]+), median ([\d.]+)\s*$")
HEAD = re.compile(r"\((\d+) samples\)")

FIELDS = ["arm", "tokenizer", "order", "schedule", "tseed", "sseed", "steps", "repair", "cell",
          "topo_eta", "chem_eta", "base_eta", "n_samples",
          "validity", "novelty", "backbone", "connected", "strict", "frag_mean", "frag_median"]


def collect(out_root):
    rows, skipped = [], []
    for txt in sorted(Path(out_root).glob("*/sweep_seed*/steps*.txt")):
        run_m = RUN.match(txt.parts[-3])
        if not run_m:
            skipped.append(str(txt))
            continue
        cell_m = CELL.match(txt.stem)
        if not cell_m:
            skipped.append(str(txt))
            continue
        r = dict(run_m.groupdict())
        r["arm"] = f"{r['tokenizer']}_{r['order']}_{r['schedule']}"
        r["tseed"] = int(r["tseed"])
        r["sseed"] = int(txt.parts[-2].removeprefix("sweep_seed"))
        r["steps"] = int(cell_m["steps"])
        r["repair"] = "off" if cell_m["nogram"] else "on"
        r["cell"] = cell_m["etas"]
        if m := ETA3.match(r["cell"]):
            r["topo_eta"], r["chem_eta"], r["base_eta"] = (int(m[k]) for k in ("t", "c", "b"))
        elif m := ETA1.match(r["cell"]):
            # single schedule: no token classes, so only base_eta acts
            r["topo_eta"] = r["chem_eta"] = 0
            r["base_eta"] = int(m["b"])
        else:
            skipped.append(str(txt))
            continue

        text = txt.read_text()
        if h := HEAD.search(text):
            r["n_samples"] = int(h.group(1))
        for line in text.splitlines():
            if f := FRAG.match(line):
                r["frag_mean"], r["frag_median"] = float(f[1]), float(f[2])
            elif m := NUM.match(line):
                r[m.group(1)] = float(m.group(2))
        rows.append(r)
    return rows, skipped


def summarise(rows):
    """Per arm: the mean over cells, and the spread across training seeds at matched cells."""
    print(f"\n{'arm':<22}{'tseeds':>7}{'cells':>7}{'strict':>9}{'best':>8}{'seed spread':>13}")
    print("-" * 68)
    by_arm = defaultdict(list)
    for r in rows:
        if r.get("strict") is not None:
            by_arm[r["arm"]].append(r)
    for arm, rs in sorted(by_arm.items()):
        seeds = sorted({r["tseed"] for r in rs})
        vals = [r["strict"] for r in rs]
        # spread across training seeds, matched on (sseed, steps, cell, repair)
        cellwise = defaultdict(dict)
        for r in rs:
            cellwise[(r["sseed"], r["steps"], r["cell"], r["repair"])][r["tseed"]] = r["strict"]
        spreads = [max(d.values()) - min(d.values()) for d in cellwise.values() if len(d) > 1]
        sp = f"{statistics.mean(spreads):.3f}" if spreads else "n/a"
        print(f"{arm:<22}{len(seeds):>7}{len(rs):>7}{statistics.mean(vals):>9.3f}"
              f"{max(vals):>8.3f}{sp:>13}")
    if any(len({r['tseed'] for r in rs}) > 1 for rs in by_arm.values()):
        print("\n'seed spread' is the mean max-min across training seeds at one matched cell:")
        print("the noise floor any arm-vs-arm claim has to clear.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", default=str(ROOT / "output"))
    ap.add_argument("--out", default=str(ROOT / "results" / "seed_sweep.csv"))
    ap.add_argument("--summary", action="store_true", help="also print the per-arm table")
    args = ap.parse_args()

    rows, skipped = collect(args.out_root)
    if not rows:
        msg = f"no cells found under {args.out_root}/*/sweep_seed*/steps*.txt"
        if skipped:
            msg += (f"\n  {len(skipped)} file(s) skipped: their run directory is not named "
                    f"<tokenizer>_<order>_<schedule>_seed<N>.\n  first: {skipped[0]}")
        raise SystemExit(msg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cells -> {out}")
    for name in ("strict", "connected", "validity"):
        missing = sum(name not in r for r in rows)
        if missing:
            print(f"  warning: {missing} cells have no {name}")
    if skipped:
        print(f"  skipped {len(skipped)} unparsed file(s), first: {skipped[0]}")
    if args.summary:
        summarise(rows)


if __name__ == "__main__":
    main()
