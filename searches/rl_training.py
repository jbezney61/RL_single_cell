import argparse
import pickle
import multiprocessing as mp
from rl_utils import DQNTrainer
from search_utils import AverageCellPerturbationSearch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help='Target cell line')
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--max_steps', type=int, default=6)
    args = parser.parse_args()

    # Load data
    with open('../data_and_models/cleaned_data.pkl', 'rb') as f:
        df = pickle.load(f)

    with open('conversion_dict.pkl', 'rb') as f:
        conversion_dict = pickle.load(f)
    
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
        'EPSILON_DECAY_EPISODES': 7000,
        'TOTAL_TRAINING_EPISODES': 10000,
        'TARGET_UPDATE_FREQUENCY': 20,
        'MAX_PATH_STEPS': 8,
    }

    trainer = DQNTrainer(env, hparams)
    trainer.train_for_target(args.target, starting_cls)

if __name__ == '__main__':
    main()
