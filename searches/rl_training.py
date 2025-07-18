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
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def main():
    set_seeds()
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help='Target cell line')
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--max_steps', type=int, default=5)
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent
    data_dir = script_dir.parent / 'data_and_models'
    
    cleaned_data_path = data_dir / 'cleaned_data.pkl'
    conversion_dict_path = data_dir / 'conversion_dict.pkl'
    # --- PATH CORRECTION END ---

    # Load data using the corrected paths
    print(f"Loading data from: {cleaned_data_path}")
    with open(cleaned_data_path, 'rb') as f:
        df = pickle.load(f)
    print("Successfully loaded cleaned_data.pkl.")

    print(f"Loading conversion dict from: {conversion_dict_path}")
    with open(conversion_dict_path, 'rb') as f:
        conversion_dict = pickle.load(f)
    print("Successfully loaded conversion_dict.pkl.")
    
    starting_cls = conversion_dict[args.target]

    env = AverageCellPerturbationSearch(df)

    # Define hyperparameters
    hparams = {
        'LEARNING_RATE': 1e-4,
        'REPLAY_BUFFER_CAPACITY': 200000,
        'BATCH_SIZE': 256,
        'GAMMA': 0.99,
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.05,
        'EPSILON_DECAY_EPISODES': int(0.9 * args.episodes),
        'TOTAL_TRAINING_EPISODES': args.episodes,
        'TARGET_UPDATE_FREQUENCY': 20,
        'MAX_PATH_STEPS': args.max_steps,
    }

    trainer = DQNTrainer(env, hparams)
    trainer.train_for_target(args.target, starting_cls)

if __name__ == '__main__':
    main()
