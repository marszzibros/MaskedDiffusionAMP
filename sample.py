import argparse
import torch
import os
import math
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
            # Split by comma: "<blank>,0" -> ["<blank>", "0"]
            try:
                token, idx = line.strip().split(',')
                token_dict[token] = int(idx)
            except ValueError:
                print(f"Skipping malformed line in vocab: {line}")
                continue
    
    return token_dict

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {args.checkpoint_path}")
    model = DiscreteFlowMatching.load_from_checkpoint(args.checkpoint_path)
    model.to(device)
    model.eval()

    current_eta = args.eta if args.eta is not None else model.eta
    print(f"Sampling with stochasticity (eta): {current_eta}")

    token_dict = load_vocab(args.vocab_path)
    print(f"Loaded vocabulary size: {len(token_dict)}")

    total_samples = args.num_samples
    batch_size = args.batch_size
    num_batches = math.ceil(total_samples / batch_size)

    print(f"Generating {total_samples} samples in {num_batches} batches...")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    conditions = {'species': [1,2], 'groups': [2,5], 'mic': 7}
    scales = {'species': 1.5, 'groups': 1.5, 'mic': 3.0}
    
    with open(args.output_file, "w") as f_out:
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Sampling"):
                # Determine how many samples to generate in this specific batch
                # (The last batch might be smaller than batch_size)
                samples_generated_so_far = i * batch_size
                samples_remaining = total_samples - samples_generated_so_far
                current_batch_n = min(batch_size, samples_remaining)
                
                model.hparams.num_steps = 500

                # Call the model's generation method
                sequences = model.generate_sample(
                    tokens_dict=token_dict,
                    conditions=conditions,
                    scales=scales,
                    num_samples=current_batch_n,
                    max_length=model.hparams.max_length,
                    eta=current_eta
                )

                # Write to file
                for seq in sequences:
                    f_out.write(seq + "\n")

    print(f"Done! {total_samples} samples saved to: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate protein samples from trained DFM model.")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the .ckpt file")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocab csv file (e.g. ./data/vocab.csv)")
    parser.add_argument("--num_samples", type=int, required=True, help="Total number of samples to generate")
    
    # Optional arguments
    parser.add_argument("--output_file", type=str, default="generated_samples.txt", help="Where to save the results")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for generation (adjust based on GPU memory)")
    parser.add_argument("--eta", type=float, default=None, help="Override the stochasticity parameter (default uses model's trained eta)")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps for generation")

    args = parser.parse_args()
    main(args)