"""Fraction of samples that decode to a parseable molecule."""


def validity(samples):
    if not samples:
        return float("nan")
    return sum(s.valid for s in samples) / len(samples)
