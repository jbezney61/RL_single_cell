# --- 1. Imports and Setup ---
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
import pickle
import random
import argparse # Used for command-line arguments
import pathlib

# Import your custom classes from the .py files
# Make sure these files are in the same directory or your Python path
from search_utils import AverageCellPerturbationSearch
from rl_utils2 import DQNTrainer, ReplayBuffer # Assuming ReplayBuffer is also in rl_utils

def set_seeds(seed_value=43):
    """Sets seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"Seeds set to {seed_value} for reproducibility.")

def main():
    """
    Main function to run the RL training pipeline.
    """
    # --- 2. Define Configuration & Hyperparameters ---
    # Arguments are now parsed from the command line
    parser = argparse.ArgumentParser(description="Run DQN training with K-Fold Cross-Validation.")
    parser.add_argument('--target', type=str, required=True, help='The target cell line to train for (e.g., CVCL_0292).')
    parser.add_argument('--episodes', type=int, default=10000, help='Total number of training episodes.')
    parser.add_argument('--max_steps', type=int, default=5, help='Maximum steps per episode.')
    args = parser.parse_args()

    set_seeds()
    
    # It's best practice to keep all hyperparameters in one place
    H_PARAMS = {
        'LEARNING_RATE': 1e-5,
        'REPLAY_BUFFER_CAPACITY': args.episodes * args.max_steps,
        'BATCH_SIZE': 64,
        'GAMMA': 0.99,
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.05,
        'EPSILON_DECAY_EPISODES': int(0.9 * args.episodes),
        'TARGET_UPDATE_FREQUENCY': 100, # episodes
        'TOTAL_TRAINING_EPISODES': args.episodes,
        'MAX_PATH_STEPS': args.max_steps,
    }

    print("Hyperparameters defined:")
    print(H_PARAMS)

    # --- 3. Initialize Environment and Trainer ---
    try:

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

        # Initialize the search environment from the data
        search_env = AverageCellPerturbationSearch(df)
        print("Search environment initialized.")

    except FileNotFoundError as e:
        print(f"ERROR: A required data file was not found: {e}")
        return # Exit the script if data is missing

    # Initialize the DQNTrainer
    dqn_trainer = DQNTrainer(
        search_environment=search_env,
        h_params=H_PARAMS
    )
    print("DQNTrainer initialized.")

    # --- 4. K-Fold Cross-Validation Setup ---
    TARGET_CL = args.target
    N_FOLDS = 5

    print(f"\n--- Setting up {N_FOLDS}-Fold Cross-Validation for Target: {TARGET_CL} ---")

    try:
        possible_starts = np.array(conversion_dict[TARGET_CL])
        print(f"Found {len(possible_starts)} specific starting cells for target {TARGET_CL}.")
    except KeyError:
        print(f"ERROR: Target '{TARGET_CL}' not found in conversion_dict.pkl.")
        return

    start_coords = np.array([search_env.centroids[cl] for cl in possible_starts])
    kmeans = KMeans(n_clusters=N_FOLDS, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(start_coords)
    print(f"Clustered starting cells into {N_FOLDS} spatial folds for validation.")

    cv_results = []

    # --- 5. Run the Cross-Validation Loop ---
    for fold_to_hold_out in range(N_FOLDS):
        print(f"\n===== Starting Cross-Validation Fold {fold_to_hold_out + 1}/{N_FOLDS} =====")
        
        validation_indices = np.where(cluster_labels == fold_to_hold_out)[0]
        training_indices = np.where(cluster_labels != fold_to_hold_out)[0]
        
        validation_cls = possible_starts[validation_indices].tolist()
        training_cls = possible_starts[training_indices].tolist()

        print("Re-initializing model weights and replay buffer for new fold...")
        def reset_weights(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        dqn_trainer.policy_net.apply(reset_weights)
        dqn_trainer.target_net.load_state_dict(dqn_trainer.policy_net.state_dict())
        dqn_trainer.replay_buffer = ReplayBuffer(dqn_trainer.h_params['REPLAY_BUFFER_CAPACITY'])
        
        best_score_for_fold = dqn_trainer.train_for_target(
            target_cl=TARGET_CL,
            training_starts=training_cls,
            validation_starts=validation_cls,
            fold_num=fold_to_hold_out + 1
        )
        
        cv_results.append({
            'fold': fold_to_hold_out + 1,
            'best_validation_score': best_score_for_fold
        })

    # --- 6. Analyze Cross-Validation Results ---
    results_df = pd.DataFrame(cv_results)
    print("\n\n===== Cross-Validation Summary =====")
    print(results_df)

    final_avg_score = results_df['best_validation_score'].mean()
    final_std_score = results_df['best_validation_score'].std()
    print(f"\nAverage Validation Score across {N_FOLDS} folds: {final_avg_score:.4f} (+/- {final_std_score:.4f})")
    print("\nThis score gives a reliable estimate of the performance of your chosen hyperparameters.")


if __name__ == '__main__':
    main()