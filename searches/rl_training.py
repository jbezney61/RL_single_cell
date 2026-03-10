import pandas as pd
import numpy as np
import pickle
import torch
import random
import pathlib
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from search_utils import AverageCellPerturbationSearch, search_to_df
from searches.rl_utils import DebugTrainer
from tqdm import tqdm

# --- 1. PARALLEL WORKER BLOCK (MUST BE TOP-LEVEL) ---
# These variables live on each individual CPU core
_worker_trainer = None
_worker_target_cl = None

def _init_worker(env, hp, model_state, target):
    """Sets up the agent on a specific CPU core."""
    global _worker_trainer, _worker_target_cl
    # We re-initialize the trainer in 'eval' mode on the CPU for workers
    trainer = DebugTrainer(env, hp)
    trainer.device = torch.device('cpu') 
    trainer.policy_net.to(trainer.device)
    trainer.policy_net.load_state_dict(model_state)
    trainer.policy_net.eval()
    _worker_trainer = trainer
    _worker_target_cl = target

def _worker_eval_task(start_cl):
    # 1. Greedy (k=1)
    g_dist = _worker_trainer.env.eval_path_dqn(
        start_cl, _worker_target_cl, _worker_trainer.policy_net, 
        _worker_trainer.device, strategy='tree', k=1
    )
    # 2. Beam (k=20)
    b_dist = _worker_trainer.env.eval_path_dqn(
        start_cl, _worker_target_cl, _worker_trainer.policy_net, 
        _worker_trainer.device, strategy='beam', k=20
    )
    
    init_dist = np.linalg.norm(_worker_trainer.env.centroids[start_cl] - _worker_trainer.env.centroids[_worker_target_cl])
    return 1.0 - (g_dist / (init_dist + 1e-6)), 1.0 - (b_dist / (init_dist + 1e-6))

# --- 2. THE EVALUATION HUB ---
def parallel_evaluate(trainer, env, cell_list, target_cl):
    """Distributes the cell list across CPU cores with a progress bar."""
    # Convert model to CPU to send to workers
    state_dict = {k: v.cpu() for k, v in trainer.policy_net.state_dict().items()}
    
    # Auto-detect cores or use your terminal check result
    num_processes = min(mp.cpu_count(), 14) 

    results = []
    with mp.Pool(processes=num_processes, initializer=_init_worker, 
                 initargs=(env, trainer.hp, state_dict, target_cl)) as pool:
        
        # Use imap to wrap with tqdm for a real-time progress bar
        # total=len(cell_list) is required for tqdm to show the percentage
        for res in tqdm(pool.imap(_worker_eval_task, cell_list), 
                        total=len(cell_list), 
                        desc="Periodic Evaluation", 
                        leave=False):
            results.append(res)
        
    greedy_results, beam_results = zip(*results)
    return np.mean(greedy_results), np.mean(beam_results)


# --- 3. MAIN TRAINING LOOP ---
def main():
    HP = {
        'LEARNING_RATE': 1e-5, 'BATCH_SIZE': 64, 'GAMMA': 0.99,
        'EPSILON_START': 1.0, 'EPSILON_END': 0.10, 'EPSILON_DECAY_EPISODES': 150000,
        'TOTAL_EPISODES': 200000, 'MAX_PATH_STEPS': 5,
        'REPLAY_BUFFER_CAPACITY': 500000, 'JITTER_STD': 0.005, 'TARGET_UPDATE_FREQ': 200
    }
    none_moves = 0 

    random.seed(43); np.random.seed(43); torch.manual_seed(43)
    
    data_path = pathlib.Path(__file__).parent.parent / 'data_and_models' / 'cleaned_data.pkl'
    with open(data_path, 'rb') as f: df = pickle.load(f)
    env = AverageCellPerturbationSearch(df)
    target = "CVCL_0292"

    all_starts = [cl for cl in env.cell_lines if cl != target]
    train_cl, temp_cl = train_test_split(all_starts, test_size=0.20, random_state=42)

    val_cl, test_cl = train_test_split(temp_cl, test_size=0.50, random_state=42)

    print(f"Dataset Split: Train={len(train_cl)}, Val={len(val_cl)}, Test={len(test_cl)}")
    
    trainer = DebugTrainer(env, HP)
    history = []

    print(f"Starting Debugging Run for {target}...")

    for ep in tqdm(range(HP['TOTAL_EPISODES'])):
        start_cl = np.random.choice(train_cl)
        curr_pos = env.centroids[start_cl]
        target_pos = env.centroids[target]
        radius = env.radiuses[target]
        init_dist = np.linalg.norm(curr_pos - target_pos)

        eps = max(HP['EPSILON_END'], HP['EPSILON_START'] - (ep/HP['EPSILON_DECAY_EPISODES']) * (HP['EPSILON_START']-HP['EPSILON_END']))

        for t in range(HP['MAX_PATH_STEPS']):
            steps_rem = HP['MAX_PATH_STEPS'] - t
            action = trainer.select_action(curr_pos, steps_rem, eps)
            
            # Step Logic (Blended Displacement)
            dists, idxs = env.centroid_tree.query(curr_pos, k=2)
            weights = 1.0/(dists + 1e-6); weights /= weights.sum()
            disp, _ = env.blended_disp([(env.centroid_keys[i], w) for i,w in zip(idxs, weights)], env.drugs[action])
            if disp is None:
                #none_moves += 1
                #if none_moves % 1000 == 0: # Warning every 1000 stalls to avoid clutter
                    #print(f"WARNING: Stall detected. Drug '{env.drugs[action]}' has no data for neighbors of cell.")
                disp = np.zeros_like(curr_pos)
            
            next_pos = curr_pos + disp
            d_before, d_after = np.linalg.norm(curr_pos - target_pos), np.linalg.norm(next_pos - target_pos)
            
            # Decision:
            reward = (d_before - d_after) / (init_dist + 1e-6)*3
            done = (d_after < radius) or (t + 1 == HP['MAX_PATH_STEPS'])
            trainer.replay_buffer.append((curr_pos, steps_rem, action, reward, next_pos, steps_rem-1, done))
            curr_pos = next_pos
            trainer.learn()
            if done: break
        
        if ep % HP['TARGET_UPDATE_FREQ'] == 0:
            trainer.target_net.load_state_dict(trainer.policy_net.state_dict())

        # Decision: Use the Parallel Evaluation Hub every 20000 episodes
        if (ep + 1) % 20000 == 0:
            tr_greedy, tr_beam = parallel_evaluate(trainer, env, train_cl, target)
            val_greedy, val_beam = parallel_evaluate(trainer, env, val_cl, target)

            history.append({'ep': ep+1, 'tr_g': tr_greedy, 'tr_b': tr_beam, 'val_g': val_greedy, 'val_b': val_beam})
            print(f"\nEp {ep+1} | Train Greedy: {tr_greedy:.1%} | Train Beam: {tr_beam:.1%} | Val Greedy: {val_greedy:.1%} | Val Beam: {val_beam:.1%}")

    print("Addestramento completato. Generazione risultati RL per Sanity Check...")

    trainer.policy_net.eval()

    model_save_path = "dqn_policy_net.pth"
    
    # Save the state_dict (standard PyTorch practice)
    torch.save(trainer.policy_net.state_dict(), model_save_path)
    print(f"Modello salvato con successo in '{model_save_path}'.")
    
    rl_results = []
    for cl in tqdm(test_cl, desc="Final RL Test Evaluation"):
        # Utilizziamo 'tree' con k=1 per avere la scelta "Greedy" pura del modello
        res = env.search_path_dqn(
            f"rl_test_{cl}", cl, target, 
            trainer.policy_net, trainer.device, 
            strategy='tree', n_steps=5, k=1
        )
        rl_results.append(search_to_df(res))

    # Salvataggio del dataset per il confronto con Tree Search e Probabilistic Search
    rl_eval_df = pd.concat(rl_results, ignore_index=True)
    output_file = "rl_greedy_eval.pkl"
    rl_eval_df.to_pickle(output_file)
    print(f"File '{output_file}' salvato con successo (formato Pickle).")
    
    # Generate the Final Graph
    plot_debug(history, target)

def plot_debug(history, target):
    df = pd.DataFrame(history).set_index('ep')
    plt.figure(figsize=(10, 5))
    plt.plot(df['tr_g'], label='Train Greedy', color='blue')
    plt.plot(df['val_g'], label='Val Greedy', color='skyblue', linestyle='--')
    plt.plot(df['val_b'], label='Val Beam (k=20)', color='red')
    plt.axhline(0, color='black', alpha=0.3)
    plt.title(f"Clean-Slate Debugging: {target}")
    plt.legend()
    plt.savefig(f"debug_results_{target}.png")

if __name__ == '__main__':
    main()