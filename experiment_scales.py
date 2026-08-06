import argparse
import torch
import os
import math
import re
import csv
import pandas as pd
from tqdm import tqdm
from DFM import DiscreteFlowMatching

def load_vocab(vocab_path):
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}. You need the token_dict to decode outputs.")
    
    token_dict = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                token, idx = line.strip().split(',')
                token_dict[token] = int(idx)
            except ValueError:
                continue
    
    return token_dict

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy")

def extract_sequence(seq):
    match = re.search(r'<SOS>(.*?)<EOS>', seq)
    if match:
        return match.group(1)
    return seq.strip()

def is_valid_sequence(extracted_seq):
    if not extracted_seq:
        return False
    if '<' in extracted_seq or '>' in extracted_seq:
        return False
    return all(c in VALID_AMINO_ACIDS for c in extracted_seq)

def load_training_data(dbaasp_path, non_amps_path):
    training_seqs = set()
    
    if os.path.exists(dbaasp_path):
        df_dbaasp = pd.read_csv(dbaasp_path)
        if 'modified_sequence' in df_dbaasp.columns:
            for seq in df_dbaasp['modified_sequence'].dropna():
                extracted = extract_sequence(str(seq))
                if is_valid_sequence(extracted):
                    training_seqs.add(extracted)
                
    if os.path.exists(non_amps_path):
        df_non_amps = pd.read_csv(non_amps_path)
        if 'Sequence' in df_non_amps.columns:
            for seq in df_non_amps['Sequence'].dropna():
                if '<SOS>' in str(seq):
                    extracted = extract_sequence(str(seq))
                else:
                    extracted = str(seq).strip()
                if is_valid_sequence(extracted):
                    training_seqs.add(extracted)
                
    return training_seqs

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {args.checkpoint_path}")
    model = DiscreteFlowMatching.load_from_checkpoint(args.checkpoint_path)
    model.to(device)
    if hasattr(model, 'ema') and model.ema is not None:
        model.ema.move_shadow_params_to_device(device)
    model.eval()

    current_eta = args.eta if args.eta is not None else model.eta
    print(f"Sampling with stochasticity (eta): {current_eta}")

    token_dict = load_vocab(args.vocab_path)
    print(f"Loaded vocabulary size: {len(token_dict)}")

    training_seqs = load_training_data(args.dbaasp_path, args.non_amps_path)
    print(f"Loaded {len(training_seqs)} unique training sequences.")

    total_samples = args.num_samples
    batch_size = args.batch_size
    num_batches = math.ceil(total_samples / batch_size)

    scales = {'species': args.species_scale, 'groups': args.groups_scale, 'mic': args.mic_scale}
    print(f"Using CFG scales: {scales}")

    print(f"Generating {total_samples} samples in {num_batches} batches...")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    generated_sequences = []
    
    import random
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Sampling"):
            samples_generated_so_far = i * batch_size
            samples_remaining = total_samples - samples_generated_so_far
            current_batch_n = min(batch_size, samples_remaining)
            
            model.hparams.num_steps = args.steps
            
            mic_tensor = torch.zeros(current_batch_n, model.dims['mic'], device=device)
            for b in range(current_batch_n):
                m = random.randint(max(0, args.mic - 1), min(model.dims['mic'] - 1, args.mic + 1))
                mic_tensor[b, m] = 1.0

            conditions = {
                'species': args.species, 
                'groups': args.groups, 
                'mic': mic_tensor
            }

            sequences = model.generate_sample(
                tokens_dict=token_dict,
                conditions=conditions,
                scales=scales,
                num_samples=current_batch_n,
                max_length=model.hparams.max_length,
                eta=current_eta,
                temperature=args.temperature,
                k_samples=args.k_samples,
                use_charge_filter=args.use_charge_filter
            )

            generated_sequences.extend(sequences)

    # Evaluate validity (success rate), novelty, and repeat rate
    valid_count = 0
    novel_count = 0
    seen_valid_seqs = set()
    results = []
    for seq in generated_sequences:
        extracted = extract_sequence(seq)
        valid = is_valid_sequence(extracted)
        in_train = False
        is_repeat = False
        if valid:
            valid_count += 1
            if extracted in seen_valid_seqs:
                is_repeat = True
            else:
                seen_valid_seqs.add(extracted)

            in_train = extracted in training_seqs
            if not in_train:
                novel_count += 1
        results.append((seq, extracted, valid, in_train, is_repeat))

    unique_valid_count = len(seen_valid_seqs)
    repeat_count = valid_count - unique_valid_count

    success_rate = valid_count / total_samples * 100 if total_samples > 0 else 0
    novelty_rate = novel_count / total_samples * 100 if total_samples > 0 else 0
    novelty_of_valid_rate = novel_count / valid_count * 100 if valid_count > 0 else 0
    repeat_rate = repeat_count / valid_count * 100 if valid_count > 0 else 0

    with open(args.output_file, "w", newline="", encoding="utf-8") as f_out:
        f_out.write(f"# Success rate (Valid sequences): {valid_count}/{total_samples} ({success_rate:.2f}%)\n")
        f_out.write(f"# Novelty rate (Valid & Novel / Total): {novel_count}/{total_samples} ({novelty_rate:.2f}%)\n")
        f_out.write(f"# Novelty rate among valid sequences: {novel_count}/{valid_count} ({novelty_of_valid_rate:.2f}%)\n")
        f_out.write(f"# Repeat rate among valid sequences: {repeat_count}/{valid_count} ({repeat_rate:.2f}%)\n")
        writer = csv.writer(f_out)
        writer.writerow(["Generated Sequence", "Extracted Sequence", "Is Valid", "In Training Data", "Is Repeat"])
        for seq, extracted, valid, in_train, is_repeat in results:
            writer.writerow([seq, extracted, valid, in_train, is_repeat])

    print(f"Valid sequences (Success rate): {valid_count}/{total_samples} ({success_rate:.2f}%)")
    print(f"Novel sequences (Novelty rate): {novel_count}/{total_samples} ({novelty_rate:.2f}%)")
    if valid_count > 0:
        print(f"Novelty rate among valid sequences: {novel_count}/{valid_count} ({novelty_of_valid_rate:.2f}%)")
        print(f"Repeat rate among valid sequences: {repeat_count}/{valid_count} ({repeat_rate:.2f}%)")
    print(f"Results saved to {args.output_file}")

    if args.summary_file:
        summary_dir = os.path.dirname(os.path.abspath(args.summary_file))
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        file_exists = os.path.exists(args.summary_file)
        with open(args.summary_file, "a") as f_sum:
            if not file_exists:
                f_sum.write("species_scale,groups_scale,mic_scale,valid_count,novel_count,repeat_count,total_samples,success_rate,novelty_rate,repeat_rate,output_file\n")
            f_sum.write(f"{args.species_scale},{args.groups_scale},{args.mic_scale},{valid_count},{novel_count},{repeat_count},{total_samples},{success_rate:.2f},{novelty_rate:.2f},{repeat_rate:.2f},{args.output_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment with different CFG scales for AMP generation.")
    
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the .ckpt file")
    parser.add_argument("--vocab_path", type=str, default="./data/dict.csv", help="Path to the vocab csv file")
    parser.add_argument("--dbaasp_path", type=str, default="./data/dbaasp.csv", help="Path to dbaasp.csv")
    parser.add_argument("--non_amps_path", type=str, default="./data/non_amps.csv", help="Path to non_amps.csv")
    parser.add_argument("--num_samples", type=int, required=True, help="Total number of samples to generate")
    parser.add_argument("--output_file", type=str, default="experiment_results.csv", help="Where to save the results")
    parser.add_argument("--summary_file", type=str, default=None, help="Path to a summary CSV file to append grid search novelty results")
    
    # CFG Scales
    parser.add_argument("--species_scale", type=float, default=1.0, help="CFG scale for species")
    parser.add_argument("--groups_scale", type=float, default=1.0, help="CFG scale for groups")
    parser.add_argument("--mic_scale", type=float, default=1.0, help="CFG scale for MIC")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--k_samples", type=int, default=1)
    parser.add_argument("--use_charge_filter", action="store_true")
    parser.add_argument("--species", type=int, nargs="+", default=[0])
    parser.add_argument("--groups", type=int, nargs="+", default=[0])
    parser.add_argument("--mic", type=int, default=2)

    args = parser.parse_args()
    main(args)
