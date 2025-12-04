import argparse
import torch
import json
import os
from tqdm import tqdm
from math import ceil
from DFM import DiscreteFlowMatching

def load_vocab(vocab_path):
    """Loads the token dictionary from a JSON file."""
    with open(vocab_path, 'r') as f:
        token_dict = json.load(f)
    return token_dict

def main(args):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    print(f"Loading checkpoint from: {args.checkpoint}")
    model = DiscreteFlowMatching.load_from_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()

    # Override hyperparameters if provided in args
    if args.eta is not None:
        model.eta = args.eta
        print(f"Overriding stochasticity parameter eta to: {args.eta}")

    # 3. Load Vocab
    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {args.vocab_path}. Please save your token_dict as a JSON file.")
    
    token_dict = load_vocab(args.vocab_path)
    print(f"Loaded vocabulary with {len(token_dict)} tokens.")

    # 4. Prepare Batches
    num_samples = args.num_samples
    batch_size = args.batch_size
    num_batches = ceil(num_samples / batch_size)
    
    print(f"Generating {num_samples} samples in {num_batches} batches...")

    # 5. Generation Loop
    all_sequences = []
    
    with open(args.output_file, 'w') as f_out:
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Generating"):
                # Calculate current batch size (handle last batch)
                current_batch_size = min(batch_size, num_samples - (i * batch_size))
                
                # Generate samples
                # Note: max_length uses the model's saved hparam unless overridden
                sequences = model.generate_sample(
                    tokens_dict=token_dict,
                    num_samples=current_batch_size,
                    max_length=args.max_length if args.max_length else model.hparams.max_length,
                    eta=model.eta
                )
                
                # Write immediately to file to save memory
                for seq in sequences:
                    f_out.write(seq + '\n')
                    
    print(f"Successfully saved {num_samples} samples to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from Discrete Flow Matching model")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .ckpt model file")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocab/token_dict JSON file")
    parser.add_argument("--num_samples", type=int, required=True, help="Total number of samples to generate")
    parser.add_argument("--output_file", type=str, default="generated_samples.txt", help="Output text file path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--eta", type=float, default=None, help="Override stochasticity (eta) value")
    parser.add_argument("--max_length", type=int, default=None, help="Override sequence length")

    args = parser.parse_args()
    main(args)