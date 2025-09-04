import pandas as pd
import pickle
import numpy as np
import time
import torch
import platform
import argparse
import multiprocessing as mp
from tqdm import tqdm
import pathlib

from search_utils import AverageCellPerturbationSearch, search_to_df
from rl_utils import QNetwork 

# --- Worker Process Globals ---
worker_search = None
worker_q_net = None
worker_device = None

# debug
mod = 'mainrun'
typ = 'tree'


def init_worker(search_obj, q_net, device):
    """Initializes each worker process with shared, read-only objects."""
    global worker_search, worker_q_net, worker_device
    worker_search = search_obj
    worker_q_net = q_net
    worker_device = device

def run_dqn_search(args):
    """The core function executed by each worker process."""
    sid, start, end, k, n_steps, strategy, threshold, blend = args
    
    return worker_search.search_path_dqn(
        search_id=sid,
        starting_cl=start,
        ending_cl=end,
        q_network=worker_q_net,
        device=worker_device,
        n_steps=n_steps,
        strategy=strategy,
        k=k,
        threshold=threshold,
        blend=blend
    )

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run a parallelized, DQN-guided k-path search.")
    parser.add_argument('--target', type=str, required=True, help='The target cell line.')
    parser.add_argument('--k', type=int, default=5, help='Beam width or branching factor.')
    parser.add_argument('--n_steps', type=int, default=5, help='Number of steps in the search path.')
    parser.add_argument('--strategy', type=str, default='beam', choices=['beam', 'tree'], help='Search strategy.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Progress threshold for recording paths.')
    parser.add_argument('--blend', type=int, default=2, help='Number of nearest centroids to blend.')
    parser.add_argument('--n_workers', type=int, default=16, help='Number of parallel worker processes.')
    args = parser.parse_args()

    # --- 2. System and Data Setup ---
    system = platform.system()
    if system == 'Darwin' or system == 'Linux':
        mp.set_start_method("fork", force=True)
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    print("Loading data files...")
    script_dir = pathlib.Path(__file__).parent
    data_dir = script_dir.parent / 'data_and_models'
    
    with open(data_dir / 'cleaned_data.pkl', 'rb') as f:
        df_averaged = pickle.load(f)

    search_obj = AverageCellPerturbationSearch(df_averaged)

    # --- 3. Load the Trained DQN Model ---
    print(f"Loading trained DQN model for target: {args.target}")
    device = torch.device("cpu")
    
    # State dimensions must match the trained model
    state_dims = len(search_obj.embedding_cols) + args.n_steps
    n_actions = len(search_obj.drugs)
    
    trained_net = QNetwork(state_dims, n_actions)
    
    model_path = data_dir / f'dqn_model_{args.target}_{mod}.pth'
    try:
        trained_net.load_state_dict(torch.load(model_path, map_location=device))
        trained_net.to(device)
        trained_net.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at '{model_path}'. Cannot proceed.")
        return
    except RuntimeError as e:
        print(f"FATAL ERROR: Failed to load model. The model architecture might not match the saved file.")
        print(f"Ensure 'n_steps' ({args.n_steps}) matches the 'MAX_PATH_STEPS' used for training.")
        print(f"PyTorch error: {e}")
        return

    # --- 4. Prepare All Search Tasks ---
    all_possible_cells = search_obj.cell_lines
    
    search_args = []
    search_id = 0
    for start_cl in all_possible_cells:
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
    
    init_args = (search_obj, trained_net, device)
    with mp.Pool(processes=args.n_workers, initializer=init_worker, initargs=init_args) as pool:
        results = list(tqdm(pool.imap_unordered(run_dqn_search, search_args), total=len(search_args)))

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds")

    # --- 6. Aggregate and Save Results ---
    if results:
        # The call to search_to_df remains the same and now uses the imported function
        search_df = pd.concat([search_to_df(r) for r in results if r], ignore_index=True)
        if not search_df.empty:
            output_path = data_dir / f'dqn_search_results_{args.target}_k{args.k}_n{args.n_steps}_{typ}_{mod}.pkl'
            search_df.to_pickle(output_path)
            print(f"Generated a final DataFrame with {len(search_df)} rows.")
            print(f"Results saved to {output_path}")
        else:
            print("Search complete, but no valid paths were found to save.")
    else:
        print("Search complete, but no results were generated.")

if __name__ == "__main__":
    main()