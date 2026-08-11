import argparse
import torch
import os
import math
import numpy as np
from tqdm import tqdm
from DFM import DiscreteFlowMatching
from dataset import SafeDecoder

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

    decoder = SafeDecoder(args.tokenizer_path)
    token_dict = decoder.token_dict
    print(f"Loaded vocabulary size: {len(token_dict)}")

    length_pool = None
    if args.safe_csv:
        length_pool = decoder.length_pool(args.safe_csv)
        print(f"Length pool: {len(length_pool)} molecules, median {int(np.median(length_pool))} tokens")
    else:
        print("No --safe_csv given: falling back to uniform lengths, which will "
              "produce truncated fragments for SAFE. Pass amp_safe.csv.")

    total_samples = args.num_samples
    batch_size = args.batch_size
    num_batches = math.ceil(total_samples / batch_size)

    print(f"Generating {total_samples} samples in {num_batches} batches...")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    scales = {'species': 1.0, 'groups': 1.0, 'mic': 1.0}
    
    import random
    n_valid = n_total = 0
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
                    use_charge_filter=args.use_charge_filter,
                    length_pool=length_pool,
                    decode_fn=decoder.decode,
                    score_fn=decoder.score if args.k_samples > 1 else None,
                )

                # A SAFE string that does not decode is not a molecule; record
                # the SMILES so downstream tools get something usable.
                for seq in sequences:
                    smiles = decoder.smiles_from_safe(seq)
                    n_valid += smiles is not None
                    n_total += 1
                    f_out.write(f"{seq}\t{smiles if smiles else 'INVALID'}\n")

    pct = 100 * n_valid / max(n_total, 1)
    print(f"Done! {total_samples} samples saved to: {args.output_file}")
    print(f"Valid molecules: {n_valid}/{n_total} ({pct:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate protein samples from trained DFM model.")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the .ckpt file")
    parser.add_argument("--tokenizer_path", type=str,
                        default="molecular_dataset/dataset/data/safe/tokenizer.json",
                        help="Path to the trained SAFE tokenizer.json")
    parser.add_argument("--safe_csv", type=str,
                        default="molecular_dataset/dataset/data/safe/amp_safe.csv",
                        help="Corpus used to draw realistic generation lengths; pass '' to disable")
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