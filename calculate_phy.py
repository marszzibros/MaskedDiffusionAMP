import pandas as pd
import numpy as np
import math
import os

# --- 1. Physicochemical Scales ---
EISENBERG_SCALE = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29, 
    'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38, 
    'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12, 
    'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
}

BOMAN_SCALE = {
    'L': -4.92, 'I': -4.92, 'V': -4.04, 'F': -2.98, 'M': -2.35, 
    'W': -2.33, 'C': -1.28, 'A':  0.31, 'G':  0.74, 'T':  1.08, 
    'S':  1.24, 'Y':  1.68, 'P':  2.14, 'H':  3.01, 'Q':  3.17, 
    'N':  3.22, 'E':  3.54, 'D':  4.87, 'K':  4.21, 'R':  4.53
}

# Kyte-Doolittle Hydropathy scale (for GRAVY)
KD_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# --- 2. Scoring Functions ---
def calculate_charge_score(sequence, q_ideal=4.5, sigma=2.0):
    if not sequence or pd.isna(sequence): return 0.0
    q = sequence.count('K') + sequence.count('R') - sequence.count('D') - sequence.count('E')
    return np.exp(-((q - q_ideal)**2) / (2 * sigma**2))

def calculate_hmoment(sequence, angle, requires_cysteines=False):
    if not sequence or pd.isna(sequence): return 0.0
    if requires_cysteines and sequence.count('C') < 2: return 0.0
    
    rad_angle = math.radians(angle)
    sum_cos, sum_sin = 0.0, 0.0
    
    for n, aa in enumerate(sequence, start=1):
        h_val = EISENBERG_SCALE.get(aa, 0.0) 
        sum_cos += h_val * math.cos(rad_angle * n)
        sum_sin += h_val * math.sin(rad_angle * n)
        
    return math.sqrt(sum_cos**2 + sum_sin**2) / len(sequence)

def calculate_boman_index(sequence):
    if not sequence or pd.isna(sequence): return 0.0
    total_solubility = sum(BOMAN_SCALE.get(aa, 0.0) for aa in sequence)
    return total_solubility / len(sequence)

def calculate_gravy(sequence):
    if not sequence or pd.isna(sequence): return 0.0
    total_hydropathy = sum(KD_SCALE.get(aa, 0.0) for aa in sequence)
    return total_hydropathy / len(sequence)

# --- 3. Main Processing Pipeline ---
def process_grid_search(summary_file_path):
    # Load the summary dataframe
    summary_df = pd.read_csv(summary_file_path)
    
    # Dictionary to hold the batch averages for the summary file
    mean_metrics = {'charge': [], 'alpha': [], 'beta': [], 'boman': [], 'gravy': []}
    
    print(f"Processing {len(summary_df)} files from summary...")
    
    for index, row in summary_df.iterrows():
        filepath = row['output_file']
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found -> {filepath}")
            for k in mean_metrics: mean_metrics[k].append(np.nan)
            continue
            
        # Read individual batch CSV, ignoring comment headers
        df = pd.read_csv(filepath, comment='#')
        
        # Clean boolean columns safely
        df['Is Valid'] = df['Is Valid'].astype(str).str.strip().str.lower() == 'true'
        df['In Training Data'] = df['In Training Data'].astype(str).str.strip().str.lower() == 'true'
        
        # Create a mask for valid and novel sequences
        mask = (df['Is Valid'] == True) & (df['In Training Data'] == False)
        
        # Initialize empty score columns
        for col in ['Net_Charge_Score', 'Alpha_Helix_Score', 'Beta_Hairpin_Score', 'Boman_Index', 'GRAVY_Score']:
            df[col] = 0.0
            
        # Standardize sequence format (Handle D-amino acids by converting to uppercase)
        processed_seqs = df.loc[mask, 'Extracted Sequence'].astype(str).str.upper()
        
        # Apply metric calculations to valid rows
        df.loc[mask, 'Net_Charge_Score'] = processed_seqs.apply(calculate_charge_score)
        df.loc[mask, 'Alpha_Helix_Score'] = processed_seqs.apply(lambda x: calculate_hmoment(x, 100, False))
        df.loc[mask, 'Beta_Hairpin_Score'] = processed_seqs.apply(lambda x: calculate_hmoment(x, 160, True))
        df.loc[mask, 'Boman_Index'] = processed_seqs.apply(calculate_boman_index)
        df.loc[mask, 'GRAVY_Score'] = processed_seqs.apply(calculate_gravy)
        
        # Save individual CSV back to disk (no sorting)
        df.to_csv(filepath, index=False)
        
        # Calculate the batch means and store them for the summary
        mean_metrics['charge'].append(df.loc[mask, 'Net_Charge_Score'].mean())
        mean_metrics['alpha'].append(df.loc[mask, 'Alpha_Helix_Score'].mean())
        mean_metrics['beta'].append(df.loc[mask, 'Beta_Hairpin_Score'].mean())
        mean_metrics['boman'].append(df.loc[mask, 'Boman_Index'].mean())
        mean_metrics['gravy'].append(df.loc[mask, 'GRAVY_Score'].mean())
        
    # Append the calculated means as new columns to the summary dataframe
    summary_df['mean_charge_score'] = mean_metrics['charge']
    summary_df['mean_alpha_score'] = mean_metrics['alpha']
    summary_df['mean_beta_score'] = mean_metrics['beta']
    summary_df['mean_boman_index'] = mean_metrics['boman']
    summary_df['mean_gravy_score'] = mean_metrics['gravy']
    
    # Save the updated summary file
    summary_df.to_csv(summary_file_path, index=False)
    print("Processing complete. Summary and individual files have been updated with raw scores.")

# --- 4. Execution ---
if __name__ == "__main__":
    process_grid_search('./output/20251224-123317/grid_search_scales/grid_search_summary.csv')