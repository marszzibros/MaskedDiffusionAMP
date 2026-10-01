"""Score a sample file.

    python evaluation/run.py results/samples.csv

validity    share of samples that decode to a molecule
novelty     share of valid samples that are not in the training corpus
backbone    share of valid samples with at least 5 peptide-bonded residues in a row
fragments   SAFE fragments per sample, mean and median
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # the metric files sit beside this one

from backbone import backbone                                # noqa: E402
from fragments import mean_fragments, median_fragments       # noqa: E402
from loading import load_reference, load_samples             # noqa: E402
from novelty import novelty                                  # noqa: E402
from validity import validity                                # noqa: E402


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python evaluation/run.py results/samples.csv")
    samples = load_samples(sys.argv[1])
    if not samples:
        raise SystemExit(f"{sys.argv[1]}: no samples")
    print(f"{sys.argv[1]}  ({len(samples)} samples)")
    print(f"  validity   {validity(samples):.3f}")
    print(f"  novelty    {novelty(samples, load_reference()):.3f}")
    print(f"  backbone   {backbone(samples):.3f}")
    print(f"  fragments  mean {mean_fragments(samples):.1f}, median {median_fragments(samples):.1f}")


if __name__ == "__main__":
    main()
