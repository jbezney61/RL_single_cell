import pandas as pd
import pickle
import numpy as np
import time
import torch
import platform
import argparse
import multiprocessing as mp
from tqdm import tqdm

# Import your custom utility classes and functions
from search_utils import AverageCellPerturbationSearch, search_to_df
from rl_utils import QNetwork # Assuming QNetwork is in rl_utils.py

# --- Worker Process Globals ---
# These will hold shared, read-only objects in each worker process
worker_search = None
worker_q_net = None
worker_device = None

def init_worker(search_obj, q_net, device):
    """Initializes each worker process with shared objects."""
    global worker_search, worker_q_net, worker_device
    worker_search = search_obj
    worker_q_net = q_net
    worker_device = device

def run_dqn_search(args):
    """The core function executed by each worker process."""
    # Unpack the arguments for a single search task
    sid, start, end, k, n_steps, strategy, threshold, blend = args
    
    # Call the search method using the globally available objects
    return worker_search.search_path_dqn(
        search_id=sid,
        starting_cl=start,
        ending_cl=end,
        q_network=worker_q_net, # Use the model from the worker's global scope
        device=worker_device,   # Use the device from the worker's global scope
        n_steps=n_steps,
        strategy=strategy,
        k=k,
        threshold=threshold,
        blend=blend
    )

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run a parallelized, DQN-guided k-path search.")
    parser.add_argument('--target', type=str, required=True, help='The target cell line to guide the search.')
    parser.add_argument('--k', type=int, default=5, help='Beam width for the search.')
    parser.add_argument('--n_steps', type=int, default=5, help='Number of steps in the search path.')
    parser.add_argument('--strategy', type=str, default='beam', help='Search strategy to use.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Progress threshold for recording paths.')
    parser.add_argument('--blend', type=int, default=2, help='Number of nearest centroids to blend for displacement.')
    parser.add_argument('--n_workers', type=int, default=16, help='Number of parallel worker processes.')
    args = parser.parse_args()

    # --- 2. System and Data Setup ---
    # Set multiprocessing start method based on OS
    system = platform.system()
    if system == 'Darwin':
        # 'fork' is generally fine and faster on macOS
        mp.set_start_method("fork", force=True)
    elif system == 'Linux':
        # 'spawn' is safer for Linux with PyTorch/CUDA
        mp.set_start_method("spawn", force=True)
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    # Load required data files
    print("Loading data files...")
    with open('../data_and_models/cleaned_data.pkl', 'rb') as f:
        df_averaged = pickle.load(f)
    with open('conversion_dict.pkl', 'rb') as f:
        conversion_dict = pickle.load(f)

    # Initialize the main search environment object
    search_obj = AverageCellPerturbationSearch(df_averaged)

    # --- 3. Load the Trained DQN Model ---
    print(f"Loading trained DQN model for target: {args.target}")
    device = torch.device("cpu")
    
    state_dims = len(search_obj.embedding_cols) + 1
    n_actions = len(search_obj.drugs)
    trained_net = QNetwork(state_dims, n_actions)
    
    model_path = f'../data_and_models/dqn_model_{args.target}.pth'
    try:
        trained_net.load_state_dict(torch.load(model_path, map_location=device))
        trained_net.to(device)
        trained_net.eval() # CRITICAL: Set model to evaluation mode
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at '{model_path}'. Cannot proceed.")
        return

    # --- 4. Prepare All Search Tasks ---
    starting_cells = conversion_dict.get(args.target)
    if not starting_cells:
        print(f"FATAL ERROR: No starting cells found for target '{args.target}' in conversion_dict.")
        return

    search_args = []
    search_id = 0
    for start_cl in starting_cells:
        if start_cl == args.target:
            continue
        search_args.append((
            search_id, start_cl, args.target, args.k, 
            args.n_steps, args.strategy, args.threshold, args.blend
        ))
        search_id += 1
    
    print(f"Prepared {len(search_args)} search tasks to run on {args.n_workers} workers.")

    # --- 5. Run Searches in Parallel ---
    start_time = time.time()
    
    # The initializer passes the read-only objects to each worker once
    init_args = (search_obj, trained_net, device)
    with mp.Pool(processes=args.n_workers, initializer=init_worker, initargs=init_args) as pool:
        results = list(tqdm(pool.imap_unordered(run_dqn_search, search_args), total=len(search_args)))

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds")

    # --- 6. Aggregate and Save Results ---
    if results:
        search_df = pd.concat([search_to_df(r) for r in results if r], ignore_index=True)
        output_path = f'../data_and_models/dqn_search_results_{args.target}_k{args.k}_n{args.n_steps}.pkl'
        search_df.to_pickle(output_path)
        print(f"Generated a final DataFrame with {len(search_df)} rows.")
        print(f"Results saved to {output_path}")
    else:
        print("Search complete, but no results were generated.")

if __name__ == "__main__":
    main()