import argparse
import numpy as np
import matplotlib.pyplot as plt
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def extract_charges(file_path):
    charges = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            first_close_idx = line.find('>')
            second_open_idx = line.find('<', first_close_idx)
            second_close_idx = line.find('>', second_open_idx)
            
            seq_start_idx = second_close_idx + 1
            third_open_idx = line.find('<', seq_start_idx)
            
            if seq_start_idx > 0 and third_open_idx > seq_start_idx:
                raw_seq = line[seq_start_idx:third_open_idx]
            else:
                import re
                raw_seq = re.sub(r'<[^>]+>', '', line)
            
            raw_seq = raw_seq.strip().upper()
            clean_aa = "".join([c for c in raw_seq if c in "ACDEFGHIKLMNPQRSTVWY"])
            
            if not clean_aa:
                continue
                
            try:
                pa = ProteinAnalysis(clean_aa)
                charges.append(pa.charge_at_pH(7.0))
            except Exception:
                pass
                
    return charges

def analyze_and_plot(file_no_filter, file_with_filter, output_plot="charge_comparison.png"):
    charges_no_filter = extract_charges(file_no_filter)
    charges_filter = extract_charges(file_with_filter)
    
    if not charges_no_filter or not charges_filter:
        print("Error: Could not extract enough valid charges from one or both files.")
        return
        
    mean_no_filter = np.mean(charges_no_filter)
    var_no_filter = np.var(charges_no_filter)
    
    mean_filter = np.mean(charges_filter)
    var_filter = np.var(charges_filter)
    
    print(f"--- Without Filter ({file_no_filter}) ---")
    print(f"Count: {len(charges_no_filter)}")
    print(f"Average: {mean_no_filter:.4f}")
    print(f"Variance: {var_no_filter:.4f}\n")
    
    print(f"--- With Filter ({file_with_filter}) ---")
    print(f"Count: {len(charges_filter)}")
    print(f"Average: {mean_filter:.4f}")
    print(f"Variance: {var_filter:.4f}\n")
    
    # Plotting Mean and Variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    labels = ['No Filter', 'Charge Filtered']
    means = [mean_no_filter, mean_filter]
    variances = [var_no_filter, var_filter]
    
    colors = ['#1f77b4', '#ff7f0e']
    
    # Plot Means
    bars1 = ax1.bar(labels, means, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Average Net Charge (pH 7.0)', fontsize=14)
    ax1.set_ylabel('Mean Charge', fontsize=12)
    ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Lower Bound (+2)')
    ax1.axhline(y=9.0, color='red', linestyle=':', alpha=0.7, label='Upper Bound (+9)')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels on bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot Variances
    bars2 = ax2.bar(labels, variances, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Variance of Net Charge', fontsize=14)
    ax2.set_ylabel('Variance', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels on bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Comparison of Generated Sequences', fontsize=16, y=1.05)
    plt.tight_layout()
    
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Saved bar plot comparison to {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average and variance of net charge and plot comparisons.")
    parser.add_argument("--file_no_filter", type=str, required=True, help="Path to the generated_samples file without filtering")
    parser.add_argument("--file_with_filter", type=str, required=True, help="Path to the generated_samples file with charge filtering")
    parser.add_argument("--output_plot", type=str, default="charge_comparison.png", help="Filename for the output plot")
    
    args = parser.parse_args()
    analyze_and_plot(args.file_no_filter, args.file_with_filter, args.output_plot)
