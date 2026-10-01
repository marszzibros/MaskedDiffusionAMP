"""
Number of fragments in the SAFE string, counted over every sample (valid or not).

"""
import statistics


def fragment_count(safe):
    return len(safe.split("."))


def mean_fragments(samples):
    return statistics.fmean(fragment_count(s.safe) for s in samples) if samples else float("nan")


def median_fragments(samples):
    return statistics.median(fragment_count(s.safe) for s in samples) if samples else float("nan")
