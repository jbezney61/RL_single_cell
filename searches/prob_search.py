import pandas as pd
import pickle
import numpy as np
import time
import multiprocessing as mp
import argparse
import platform
import random
from tqdm import tqdm
from search_utils import AverageCellPerturbationSearch, search_to_df

# Local worker reference
worker_search = None

def init_worker(search_obj):
    global worker_search
    worker_search = search_obj
    random.seed()  # ensure different RNG per worker

def run_probabilistic_search(args):
    sid, start, end, n_paths, n_steps, beta, blend, threshold = args
    return worker_search.probabilistic_search(
        search_id=sid,
        starting_cl=start,
        ending_cl=end,
        n_paths=n_paths,
        n_steps=n_steps,
        beta=beta,
        blend=blend,
        threshold=threshold
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_paths', type=int, required=True, help='Number of Monte Carlo paths per pair (default: 100)')
    parser.add_argument('--n_steps', type=int, required=True, help='Max number of steps per path (default: 5)')
    parser.add_argument('--beta', type=float, default=1.0, help='Inverse temperature for Gibbs sampling (default: 1.0)')
    parser.add_argument('--blend', type=int, default=2, help='Number of WT centroids to blend (default: 2)')
    parser.add_argument('--threshold', type=float, default=0.4, help='Threshold coverage (default: 0.4)')
    args = parser.parse_args()

    n_paths = args.n_paths
    n_steps = args.n_steps
    beta = args.beta
    blend = args.blend
    threshold = args.threshold

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

    search_obj = AverageCellPerturbationSearch(df_averaged)

    # Prepare all start-end search tasks
    search_args = []
    search_id = 0
    for start_cl in search_obj.cell_lines:
        for end_cl in search_obj.cell_lines:
            if start_cl == end_cl:
                continue
            search_args.append((search_id, start_cl, end_cl, n_paths, n_steps, beta, blend, threshold))
            search_id += 1

    start_time = time.time()

    with mp.Pool(processes=8, initializer=init_worker, initargs=(search_obj,)) as pool:
        results = list(tqdm(pool.imap_unordered(run_probabilistic_search, search_args), total=len(search_args)))

    # Save results
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds")

    search_df = pd.concat([search_to_df(r) for r in results], ignore_index=True)
    output_path = f'../data_and_models/probabilistic_search_results_paths{n_paths}_steps{n_steps}_b{beta}_blend{blend}_thr{threshold}.pkl'
    search_df.to_pickle(output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
