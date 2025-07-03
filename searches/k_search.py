import pandas as pd
import pickle
import numpy as np
import time
import multiprocessing as mp
import argparse
import platform
from tqdm import tqdm
from search_utils import AverageCellPerturbationSearch, search_to_df

# Local variable to hold the shared object in each worker
worker_search = None

# Initialization function for worker processes
def init_worker(search_obj):
    global worker_search
    worker_search = search_obj

# Core search function used in each worker
def run_search(args):
    sid, start, end, k, n_steps, strategy = args
    return worker_search.search_path_k_paths(
        search_id=sid,
        starting_cl=start,
        ending_cl=end,
        strategy=strategy,
        n_steps=n_steps,
        k=k
    )

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True, help='Number of top paths to return')
    parser.add_argument('--n_steps', type=int, required=True, help='Number of steps in the search')
    parser.add_argument('--strategy', type=str, default='tree', help='Search strategy to use (default: tree)')
    args = parser.parse_args()
    k = args.k
    n_steps = args.n_steps
    strategy = args.strategy

    # Set multiprocessing start method based on OS
    system = platform.system()
    if system == 'Darwin':
        mp.set_start_method("fork", force=True)
    elif system == 'Linux':
        mp.set_start_method("spawn", force=True)
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    # Load input data
    with open('../data_and_models/cleaned_data.pkl', 'rb') as f:
        df_averaged = pickle.load(f)

    # Initialize the shared search object
    search_obj = AverageCellPerturbationSearch(df_averaged)

    # Prepare all search tasks
    search_args = []
    search_id = 0
    for start_cl in search_obj.cell_lines:
        for end_cl in search_obj.cell_lines:
            if start_cl == end_cl:
                continue
            search_args.append((search_id, start_cl, end_cl, k, n_steps, strategy))
            search_id += 1

    # Time the full execution
    start_time = time.time()

    # Run in parallel with tqdm progress tracking
    with mp.Pool(processes=8, initializer=init_worker, initargs=(search_obj,)) as pool:
        results = list(tqdm(pool.imap_unordered(run_search, search_args), total=len(search_args)))

    # Save results
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds")

    search_df = pd.concat([search_to_df(r) for r in results], ignore_index=True)
    output_path = f'../data_and_models/search_results_{strategy}_k{k}_n{n_steps}.pkl'
    search_df.to_pickle(output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
