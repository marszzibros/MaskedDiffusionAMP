import argparse
import os
import re
import csv
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 20 Standard L-Amino Acids
L_AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# 16 D-Amino Acids from dictionary (lowercase single-letter representation)
D_AMINO_ACIDS = ['a', 'c', 'f', 'h', 'i', 'k', 'l', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y']

VALID_AMINO_ACIDS = set(L_AMINO_ACIDS + D_AMINO_ACIDS)

def extract_sequence(seq):
    if not isinstance(seq, str):
        seq = str(seq)
    match = re.search(r'<SOS>(.*?)<EOS>', seq)
    if match:
        return match.group(1)
    return seq.strip()

def is_valid_sequence(seq):
    if not seq or not isinstance(seq, str):
        return False
    if '<' in seq or '>' in seq:
        return False
    return all(c in VALID_AMINO_ACIDS for c in seq)

def load_training_sequences(dbaasp_path):
    training_seqs = []
    if not os.path.exists(dbaasp_path):
        raise FileNotFoundError(f"Training data file not found at: {dbaasp_path}")
        
    df_dbaasp = pd.read_csv(dbaasp_path)
    seq_col = None
    for col in ['modified_sequence', 'sequence', 'Sequence']:
        if col in df_dbaasp.columns:
            seq_col = col
            break
            
    if seq_col is None:
        raise ValueError(f"Could not find sequence column in {dbaasp_path}")
        
    for raw_seq in df_dbaasp[seq_col].dropna():
        extracted = extract_sequence(str(raw_seq))
        if is_valid_sequence(extracted):
            training_seqs.append(extracted)
            
    return training_seqs

def load_sampled_sequences(sampled_csv_path, only_valid=True):
    if not os.path.exists(sampled_csv_path):
        raise FileNotFoundError(f"Sampled sequences file not found at: {sampled_csv_path}")
        
    df_sampled = pd.read_csv(sampled_csv_path, comment='#')
    
    if only_valid and 'Is Valid' in df_sampled.columns:
        valid_mask = df_sampled['Is Valid'].astype(str).str.strip().str.lower().isin(['true', '1'])
        df_sampled = df_sampled[valid_mask]
        
    seq_col = None
    for col in ['Extracted Sequence', 'Generated Sequence', 'sequence', 'Sequence']:
        if col in df_sampled.columns:
            seq_col = col
            break
            
    if seq_col is None:
        # Fallback to the second column or first string column
        seq_col = df_sampled.columns[0]
        
    sampled_seqs = []
    for raw_seq in df_sampled[seq_col].dropna():
        extracted = extract_sequence(str(raw_seq))
        if is_valid_sequence(extracted):
            sampled_seqs.append(extracted)
            
    return sampled_seqs

def compute_frequencies(sequences):
    all_residues = "".join(sequences)
    total_residues = len(all_residues)
    
    counts = Counter(all_residues)
    
    freqs = {}
    for aa in L_AMINO_ACIDS + D_AMINO_ACIDS:
        cnt = counts.get(aa, 0)
        freqs[aa] = (cnt / total_residues * 100.0) if total_residues > 0 else 0.0
        
    l_total_cnt = sum(counts.get(aa, 0) for aa in L_AMINO_ACIDS)
    d_total_cnt = sum(counts.get(aa, 0) for aa in D_AMINO_ACIDS)
    
    return freqs, total_residues, l_total_cnt, d_total_cnt, counts

def plot_aa_frequencies(train_freqs, sample_freqs, output_plot, title_suffix=""):
    fig, (ax_l, ax_d) = plt.subplots(2, 1, figsize=(14, 10))
    
    width = 0.35
    
    # --- 1. L-Amino Acids Subplot ---
    x_l = np.arange(len(L_AMINO_ACIDS))
    train_l = [train_freqs[aa] for aa in L_AMINO_ACIDS]
    sample_l = [sample_freqs[aa] for aa in L_AMINO_ACIDS]
    
    ax_l.bar(x_l - width/2, train_l, width, label='Training Data (DBAASP)', color='#1f77b4', alpha=0.85, edgecolor='black')
    ax_l.bar(x_l + width/2, sample_l, width, label='Sampled Sequences', color='#ff7f0e', alpha=0.85, edgecolor='black')
    
    ax_l.set_ylabel('Frequency (%)', fontsize=12, fontweight='bold')
    ax_l.set_title(f'L-Amino Acid Frequency Comparison {title_suffix}', fontsize=14, fontweight='bold')
    ax_l.set_xticks(x_l)
    ax_l.set_xticklabels(L_AMINO_ACIDS, fontsize=11, fontweight='bold')
    ax_l.legend(fontsize=11)
    ax_l.grid(axis='y', linestyle='--', alpha=0.6)
    
    # --- 2. D-Amino Acids Subplot ---
    x_d = np.arange(len(D_AMINO_ACIDS))
    train_d = [train_freqs[aa] for aa in D_AMINO_ACIDS]
    sample_d = [sample_freqs[aa] for aa in D_AMINO_ACIDS]
    
    d_labels = [f"{aa} (d-{aa.upper()})" for aa in D_AMINO_ACIDS]
    
    ax_d.bar(x_d - width/2, train_d, width, label='Training Data (DBAASP)', color='#1f77b4', alpha=0.85, edgecolor='black')
    ax_d.bar(x_d + width/2, sample_d, width, label='Sampled Sequences', color='#ff7f0e', alpha=0.85, edgecolor='black')
    
    ax_d.set_xlabel('Amino Acid Residue', fontsize=12, fontweight='bold')
    ax_d.set_ylabel('Frequency (%)', fontsize=12, fontweight='bold')
    ax_d.set_title(f'D-Amino Acid Frequency Comparison {title_suffix}', fontsize=14, fontweight='bold')
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(d_labels, fontsize=10, rotation=45, ha='right')
    ax_d.legend(fontsize=11)
    ax_d.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Bar plot comparison figure successfully saved to: {output_plot}")

def main():
    parser = argparse.ArgumentParser(description="Compare per-amino-acid frequency (L- and D-amino acids) between sampled sequences and training data.")
    
    parser.add_argument("--sampled_csv", type=str, required=True, help="Path to sampled sequences CSV file")
    parser.add_argument("--training_csv", type=str, default="./data/dbaasp.csv", help="Path to training DBAASP CSV file")
    parser.add_argument("--output_plot", type=str, default="aa_frequency_comparison.png", help="Path to save the comparison plot figure")
    parser.add_argument("--output_freq_csv", type=str, default=None, help="Optional path to save frequency statistics CSV")
    parser.add_argument("--include_invalid", action="store_true", help="Include invalid sequences from sampled CSV (default: valid only)")
    
    args = parser.parse_args()
    
    print(f"Loading training data from: {args.training_csv}")
    train_seqs = load_training_sequences(args.training_csv)
    print(f"Loaded {len(train_seqs)} valid training sequences.")
    
    print(f"Loading sampled sequences from: {args.sampled_csv}")
    only_valid = not args.include_invalid
    sampled_seqs = load_sampled_sequences(args.sampled_csv, only_valid=only_valid)
    print(f"Loaded {len(sampled_seqs)} sampled sequences (only_valid={only_valid}).")
    
    train_freqs, train_total, train_l_cnt, train_d_cnt, train_counts = compute_frequencies(train_seqs)
    sample_freqs, sample_total, sample_l_cnt, sample_d_cnt, sample_counts = compute_frequencies(sampled_seqs)
    
    print("\n--- Summary Statistics ---")
    print(f"Training Total Residues: {train_total} | L-AA: {train_l_cnt} ({train_l_cnt/train_total*100:.2f}%) | D-AA: {train_d_cnt} ({train_d_cnt/train_total*100:.2f}%)")
    print(f"Sampled  Total Residues: {sample_total} | L-AA: {sample_l_cnt} ({sample_l_cnt/sample_total*100:.2f}%) | D-AA: {sample_d_cnt} ({sample_d_cnt/sample_total*100:.2f}%)")
    
    plot_aa_frequencies(train_freqs, sample_freqs, args.output_plot)
    
    if args.output_freq_csv:
        rows = []
        for aa in L_AMINO_ACIDS:
            rows.append({
                'Amino Acid': aa,
                'Type': 'L',
                'Training Count': train_counts.get(aa, 0),
                'Training Frequency (%)': train_freqs[aa],
                'Sampled Count': sample_counts.get(aa, 0),
                'Sampled Frequency (%)': sample_freqs[aa]
            })
        for aa in D_AMINO_ACIDS:
            rows.append({
                'Amino Acid': aa,
                'Type': 'D',
                'Training Count': train_counts.get(aa, 0),
                'Training Frequency (%)': train_freqs[aa],
                'Sampled Count': sample_counts.get(aa, 0),
                'Sampled Frequency (%)': sample_freqs[aa]
            })
        df_out = pd.DataFrame(rows)
        df_out.to_csv(args.output_freq_csv, index=False)
        print(f"Saved frequency statistics to: {args.output_freq_csv}")

if __name__ == "__main__":
    main()
