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

# IMPORTANT: Import from your production utils to match architecture
from search_utils import AverageCellPerturbationSearch, search_to_df
from rl_utils_production import QNetwork 

# --- Worker Process Globals ---
worker_search = None
worker_q_net = None
worker_device = None

# Metadata for naming
MOD_NAME = 'mainrun_with_jitter'

def init_worker(search_obj, q_net, device):
    global worker_search, worker_q_net, worker_device
    worker_search = search_obj
    worker_q_net = q_net
    worker_device = device

def run_dqn_search(args):
    sid, start, end, k, n_steps, strategy, threshold, blend = args
    return worker_search.search_path_dqn(
        search_id=sid, starting_cl=start, ending_cl=end,
        q_network=worker_q_net, device=worker_device,
        n_steps=n_steps, strategy=strategy, k=k,
        threshold=threshold, blend=blend
    )

def main():
    parser = argparse.ArgumentParser(description="Parallelized Search for Jitter-Trained Model.")
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--k', type=int, default=4, help='Branching factor (k=4 for tree, k=190 for beam).')
    parser.add_argument('--n_steps', type=int, default=5)
    parser.add_argument('--strategy', type=str, default='tree', choices=['beam', 'tree'])
    parser.add_argument('--threshold', type=float, default=0.5, help='Progress threshold for recording paths (0.5 = 50% progress).')
    parser.add_argument('--blend', type=int, default=2)
    parser.add_argument('--n_workers', type=int, default=8)
    args = parser.parse_args()

    # System Setup
    if platform.system() in ['Darwin', 'Linux']:
        mp.set_start_method("fork", force=True)

    # 1. Load Data
    current_dir = pathlib.Path(__file__).parent
    data_dir = current_dir.parent / 'data_and_models'
    with open(data_dir / 'cleaned_data.pkl', 'rb') as f:
        df_averaged = pickle.load(f)
    search_obj = AverageCellPerturbationSearch(df_averaged)

    # 2. Load the Production Model
    print(f"Loading Production Model (Jitter=0.1) for target: {args.target}")
    device = torch.device("cpu")
    state_dims = len(search_obj.embedding_cols) + args.n_steps
    n_actions = len(search_obj.drugs)
    
    trained_net = QNetwork(state_dims, n_actions)
    
    # Updated to search in current directory for your production file
    model_filename = f'production_model_{args.target}.pth'
    model_path = current_dir / model_filename 
    
    try:
        trained_net.load_state_dict(torch.load(model_path, map_location=device))
        trained_net.eval()
        print(f"Successfully loaded {model_filename}")
    except FileNotFoundError:
        print(f"ERROR: {model_filename} not found in {current_dir}")
        return

    # 3. Prepare Tasks
    search_args = []
    for i, start_cl in enumerate(search_obj.cell_lines):
        if start_cl == args.target: continue
        search_args.append((i, start_cl, args.target, args.k, args.n_steps, args.strategy, args.threshold, args.blend))
    
    # 4. Execute Search
    start_time = time.time()
    init_args = (search_obj, trained_net, device)
    with mp.Pool(processes=args.n_workers, initializer=init_worker, initargs=init_args) as pool:
        results = list(tqdm(pool.imap_unordered(run_dqn_search, search_args), total=len(search_args)))

    # 5. Save in Current Directory
    if results:
        search_df = pd.concat([search_to_df(r) for r in results if r], ignore_index=True)
        output_filename = f'dqn_results_{args.target}_{args.strategy}_k{args.k}_{MOD_NAME}.pkl'
        output_path = current_dir / output_filename
        search_df.to_pickle(output_path)
        print(f"\nSaved {len(search_df)} paths to {output_filename}")

if __name__ == "__main__":
    main()