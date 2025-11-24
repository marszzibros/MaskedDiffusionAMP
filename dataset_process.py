import os
import csv
import numpy as np
import pandas as pd
from multiprocessing import Pool

categorical_bin = 10
base_path = "/netfiles/vaillab/distance_maps_long"

# --- Function to process one file ---
def process_file(filename):
    path = os.path.join(base_path, filename)
    matrix = np.load(path)  # assuming these are .npy matrices; adjust if different
    upper_values = matrix[np.triu_indices_from(matrix, k=1)]
    return upper_values.tolist()  # return list so Pool can serialize easily

# --- Parallel map ---
maps = os.listdir(base_path)
with Pool() as pool:
    results = pool.map(process_file, maps)

# Flatten list of lists
all_values = [v for sub in results for v in sub]
all_values = np.array(all_values)

# Bin edges
bin_edges = pd.qcut(all_values, q=categorical_bin, retbins=True, duplicates='drop')[1]
bin_edges = np.insert(bin_edges, 0, 0)  # add 0 at start

# Prepare dicts
distance_dicts = {}
distance_tokens = [0]

# --- Save dict_swissprot_bin ---
with open(f"data/dict_swissprot_bin_{categorical_bin}.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    for i, edge in enumerate(bin_edges):
        writer.writerow([i, edge])

# --- Save dict_swissprot_ids ---
with open(f"data/dict_swissprot_ids_{categorical_bin}.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    for i, edge in enumerate(bin_edges[:-1]):
        distance_dicts[i] = edge
        writer.writerow([edge, i])
        distance_tokens.append(edge)

    # Special tokens
    distance_dicts[0] = 0
    distance_dicts['<MASK>'] = len(bin_edges) - 1
    writer.writerow(['<MASK>', len(bin_edges) - 1])
    distance_tokens.append('<MASK>')

    distance_dicts['<blank>'] = len(bin_edges)
    writer.writerow(['<blank>', len(bin_edges)])
    distance_tokens.append('<blank>')

    distance_dicts['<SOS>'] = len(bin_edges) + 1
    writer.writerow(['<SOS>', len(bin_edges) + 1])
    distance_tokens.append('<SOS>')

    distance_dicts['<EOS>'] = len(bin_edges) + 2
    writer.writerow(['<EOS>', len(bin_edges) + 2])
    distance_tokens.append('<EOS>')
