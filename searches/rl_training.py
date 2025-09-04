import argparse
import pickle
import multiprocessing as mp
from rl_utils import DQNTrainer
from search_utils import AverageCellPerturbationSearch
import random
import torch
import numpy as np
import pathlib


def set_seeds(seed_value=43):
    """Sets random seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def main():
    set_seeds()
    parser = argparse.ArgumentParser(description="Run DQN training for a specific target cell line.")
    parser.add_argument('--target', type=str, required=True, help='Target cell line symbol (e.g., "CVCL_0292").')
    parser.add_argument('--episodes', type=int, default=10000, help='Total number of training episodes.')
    parser.add_argument('--max_steps', type=int, default=5, help='Maximum steps allowed per episode.')
    args = parser.parse_args()

    # --- Setup paths ---
    script_dir = pathlib.Path(__file__).parent
    data_dir = script_dir.parent / 'data_and_models'
    cleaned_data_path = data_dir / 'cleaned_data.pkl'

    # --- Load data ---
    print(f"Loading data from: {cleaned_data_path}")
    with open(cleaned_data_path, 'rb') as f:
        df = pickle.load(f)
    print("Successfully loaded cleaned_data.pkl.")
    
    # --- Initialize environment ---
    # The environment will contain all cell lines from the dataframe.
    env = AverageCellPerturbationSearch(df)

    # --- Define hyperparameters ---
    hparams = {
        'LEARNING_RATE': 1e-5,
        'REPLAY_BUFFER_CAPACITY': 500000,
        'BATCH_SIZE': 64,
        'GAMMA': 0.99,
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.05,
        'EPSILON_DECAY_EPISODES': int(0.9 * args.episodes),
        'TOTAL_TRAINING_EPISODES': args.episodes,
        'TARGET_UPDATE_FREQUENCY': 100,
        'MAX_PATH_STEPS': args.max_steps,
    }

    # --- Initialize and run trainer ---
    trainer = DQNTrainer(env, hparams)
    
    # By not passing a second argument, the `train_for_target` method
    # will default to using all available cell lines (except the target)
    # as starting points for training episodes.
    trainer.train_for_target(args.target)

if __name__ == '__main__':
    # Set start method for multiprocessing to prevent potential issues on some platforms
    mp.set_start_method('spawn', force=True)
    main()