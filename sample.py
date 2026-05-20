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
    if hasattr(model, 'ema') and model.ema is not None:
        model.ema.move_shadow_params_to_device(device)
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
    
    scales = {'species': 1.0, 'groups': 1.0, 'mic': 1.0}
    
    import random
    with open(args.output_file, "w") as f_out:
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Sampling"):
                # Determine how many samples to generate in this specific batch
                # (The last batch might be smaller than batch_size)
                samples_generated_so_far = i * batch_size
                samples_remaining = total_samples - samples_generated_so_far
                current_batch_n = min(batch_size, samples_remaining)
                
                model.hparams.num_steps = args.steps
                
                # Prepare varying MIC tensor
                mic_tensor = torch.zeros(current_batch_n, model.dims['mic'], device=device)
                for b in range(current_batch_n):
                    m = random.randint(max(0, args.mic - 1), min(model.dims['mic'] - 1, args.mic + 1))
                    mic_tensor[b, m] = 1.0

                conditions = {
                    'species': args.species, 
                    'groups': args.groups, 
                    'mic': mic_tensor
                }

                # Call the model's generation method
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
    parser.add_argument("--output_file", type=str, default="test.txt", help="Where to save the results")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for generation (adjust based on GPU memory)")
    parser.add_argument("--eta", type=float, default=None, help="Override the stochasticity parameter (default uses model's trained eta)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling. Lower = more confident, higher = more diverse")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps for generation")
    parser.add_argument("--k_samples", type=int, default=1, help="Number of candidate samples to generate per token step when filtering by charge")
    parser.add_argument("--use_charge_filter", action="store_true", help="Turn on the Biopython net charge filter (requires k_samples > 1 for diversity selection)")
    parser.add_argument("--species", type=int, nargs="+", default=[0], help="List of species indices")
    parser.add_argument("--groups", type=int, nargs="+", default=[0], help="List of groups indices")
    parser.add_argument("--mic", type=int, default=2, help="Base MIC value")

    args = parser.parse_args()
    main(args)