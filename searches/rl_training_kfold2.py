# --- 1. Imports and Setup ---
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
import pickle
import random
import argparse
import pathlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Import your custom classes
from search_utils import AverageCellPerturbationSearch
from rl_utils_kfold2 import DQNTrainer, ReplayBuffer

def set_seeds(seed_value=43):
    """Sets seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"Seeds set to {seed_value} for reproducibility.")

def main():
    # --- 2. Define Configuration & Hyperparameters ---
    parser = argparse.ArgumentParser(description="Run DQN training with K-Fold Cross-Validation.")
    parser.add_argument('--target', type=str, required=True, help='The target cell line.')
    parser.add_argument('--episodes', type=int, default=10000, help='Total number of training episodes.')
    parser.add_argument('--max_steps', type=int, default=5, help='Maximum steps per episode.')
    args = parser.parse_args()

    set_seeds()
    
    H_PARAMS = {
        'LEARNING_RATE': 1e-5,
        'REPLAY_BUFFER_CAPACITY': args.episodes * args.max_steps,
        'BATCH_SIZE': 64,
        'GAMMA': 0.99,
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.05,
        'EPSILON_DECAY_EPISODES': int(0.9 * args.episodes),
        'TARGET_UPDATE_FREQUENCY': 100,
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

        with open(cleaned_data_path, 'rb') as f: df = pickle.load(f)
        search_env = AverageCellPerturbationSearch(df)
        print("Search environment initialized.")
    except FileNotFoundError as e:
        print(f"ERROR: A required data file was not found: {e}")
        return

    dqn_trainer = DQNTrainer(search_environment=search_env, h_params=H_PARAMS)

    # --- 4. K-Fold Cross-Validation Setup ---
    TARGET_CL = args.target
    N_FOLDS = 5
    print(f"\n--- Setting up {N_FOLDS}-Fold Cross-Validation for Target: {TARGET_CL} ---")

    # --- MODIFIED: Use all cell lines (except the target) for cross-validation ---
    possible_starts = np.array([cl for cl in search_env.cell_lines if cl != TARGET_CL])
    print(f"Using all {len(possible_starts)} available cell lines as starting points for K-Fold setup.")

    start_coords = np.array([search_env.centroids[cl] for cl in possible_starts])
    kmeans = KMeans(n_clusters=N_FOLDS, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(start_coords)
    print(f"Clustered starting cells into {N_FOLDS} spatial folds for validation.")

    cv_results = []

    # --- 5. Run the Cross-Validation Loop ---
    for fold_to_hold_out in range(N_FOLDS):
        fold_num = fold_to_hold_out + 1
        print(f"\n===== Starting Cross-Validation Fold {fold_num}/{N_FOLDS} =====")
        
        validation_indices = np.where(cluster_labels == fold_to_hold_out)[0]
        training_indices = np.where(cluster_labels != fold_to_hold_out)[0]
        validation_cls = possible_starts[validation_indices].tolist()
        training_cls = possible_starts[training_indices].tolist()

        def reset_weights(m):
            if hasattr(m, 'reset_parameters'): m.reset_parameters()
        dqn_trainer.policy_net.apply(reset_weights)
        dqn_trainer.target_net.load_state_dict(dqn_trainer.policy_net.state_dict())
        dqn_trainer.replay_buffer = ReplayBuffer(dqn_trainer.h_params['REPLAY_BUFFER_CAPACITY'])
        
        fold_results = dqn_trainer.train_for_target(
            target_cl=TARGET_CL, training_starts=training_cls,
            validation_starts=validation_cls, fold_num=fold_num
        )
        cv_results.append(fold_results)

        # --- Generate and save a plot for the completed fold ---
        print(f"\n--- Generating Performance Plot for Fold {fold_num} ---")
        history_df = pd.DataFrame(fold_results['history'])
        plot_data = history_df.set_index('episode')

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.plot(plot_data.index, plot_data['train_greedy'], label='Train Greedy (k=1)', color='royalblue', marker='o', markersize=5)
        ax.plot(plot_data.index, plot_data['valid_greedy'], label='Validation Greedy (k=1)', color='skyblue', marker='o', markersize=5, linestyle='--')
        ax.plot(plot_data.index, plot_data['train_beam'], label='Train Beam (k=190)', color='crimson', marker='x', markersize=5)
        ax.plot(plot_data.index, plot_data['valid_beam'], label='Validation Beam (k=190)', color='salmon', marker='x', markersize=5, linestyle='--')

        ax.set_title(f'Overfitting Analysis for Target: {args.target} (Fold {fold_num})', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Episodes', fontsize=12)
        ax.set_ylabel('Average Progress', fontsize=12)
        ax.legend(title='Metric', fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        plt.tight_layout()

        plot_filename = data_dir / f'overfitting_plot_{args.target}_fold_{fold_num}.png'
        plt.savefig(plot_filename, dpi=300)
        print(f"Performance plot for Fold {fold_num} saved to: {plot_filename}")
        plt.close(fig) # Close the figure to free up memory

    # --- 6. Analyze Cross-Validation Results (Final Summary) ---
    results_df = pd.DataFrame(cv_results)
    print("\n\n===== Cross-Validation Summary =====")
    print(results_df.drop(columns=['history']).to_string())
    
    avg_train_greedy = results_df['train_greedy'].mean()
    avg_train_beam = results_df['train_beam'].mean()
    avg_valid_greedy = results_df['valid_greedy'].mean()
    avg_valid_beam = results_df['valid_beam'].mean()

    print("\n--- Average Performance Across All Folds ---")
    print(f"  Training Set   -> Greedy Progress: {avg_train_greedy:.2%}, Beam (k=190) Progress: {avg_train_beam:.2%}")
    print(f"  Validation Set -> Greedy Progress: {avg_valid_greedy:.2%}, Beam (k=190) Progress: {avg_valid_beam:.2%}")

if __name__ == '__main__':
    main()
