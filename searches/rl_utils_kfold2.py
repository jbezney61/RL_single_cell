import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from collections import namedtuple
from tqdm import trange, tqdm # Import tqdm for the new progress bars
from search_utils import AverageCellPerturbationSearch
import pathlib
import multiprocessing as mp # Import for parallel evaluation

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(64, n_actions)
        )
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    def add(self, state: tuple, action: int, reward:float, next_state: tuple, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# --- Helper functions for parallel evaluation ---
_worker_trainer = None
_worker_target_cl = None

def _init_worker(env, h_params, model_state_dict, target_cl):
    """
    Initializes each worker process with a lightweight, CPU-only trainer.
    """
    global _worker_trainer, _worker_target_cl
    
    # Create a new trainer instance within the worker process
    trainer = DQNTrainer(env, h_params)
    
    # --- CRITICAL: Ensure this worker's trainer is entirely on the CPU ---
    trainer.device = torch.device('cpu')
    trainer.policy_net.to(trainer.device)
    trainer.policy_net.load_state_dict(model_state_dict)
    trainer.policy_net.eval()
    
    _worker_trainer = trainer
    _worker_target_cl = target_cl

def _run_eval_on_cell(start_cl: str) -> tuple:
    """Function run by each worker. Evaluates a single starting cell."""
    # Run greedy test (k=1)
    dist, _ = _worker_trainer._run_greedy_test(start_cl, _worker_target_cl)
    greedy_metric = 1.0 - dist

    # Run wide beam search (k=190)
    beam_metric = 0.0
    search_result = _worker_trainer.env_provider.search_path_dqn(
        search_id='eval_worker', starting_cl=start_cl, ending_cl=_worker_target_cl,
        q_network=_worker_trainer.policy_net, device=_worker_trainer.device,
        n_steps=_worker_trainer.h_params['MAX_PATH_STEPS'], k=190, threshold=-100
    )
    df = search_result['progress_table']
    if not df.empty:
        final_step_paths = df[df['step'] == _worker_trainer.h_params['MAX_PATH_STEPS']]
        if not final_step_paths.empty:
            best_dist = final_step_paths['path'].apply(lambda p: p[4]).min()
            init_dist = np.linalg.norm(_worker_trainer.env_provider.centroids[start_cl] - _worker_trainer.env_provider.centroids[_worker_target_cl])
            beam_metric = 1.0 - (best_dist / (init_dist + 1e-6))
            
    return greedy_metric, beam_metric


class DQNTrainer:
    def __init__(self, search_environment: AverageCellPerturbationSearch, h_params: dict):
        self.env_provider = search_environment
        self.h_params = h_params
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.state_dims = len(self.env_provider.embedding_cols) + self.h_params['MAX_PATH_STEPS']
        self.n_actions = len(self.env_provider.drugs)
        self.policy_net = QNetwork(self.state_dims, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.state_dims, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=h_params['LEARNING_RATE'])
        self.replay_buffer = ReplayBuffer(h_params['REPLAY_BUFFER_CAPACITY'])
        self.loss_fn = nn.SmoothL1Loss()

    def _one_hot_encode_steps(self, steps_remaining: int) -> np.ndarray:
        one_hot = np.zeros(self.h_params['MAX_PATH_STEPS'])
        if steps_remaining > 0:
            one_hot[steps_remaining - 1] = 1
        return one_hot

    def _select_action(self, state: tuple, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                state_pos, steps_rem = state
                steps_one_hot = self._one_hot_encode_steps(steps_rem)
                state_vec = np.hstack([state_pos, steps_one_hot]).astype(np.float32)
                state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def _unpack_and_prep_batch(self, experiences: list) -> tuple:
        batch = Experience(*zip(*experiences))
        positions = np.vstack([s[0] for s in batch.state])
        steps_one_hot = np.vstack([self._one_hot_encode_steps(s[1]) for s in batch.state])
        state_vectors = np.hstack([positions, steps_one_hot]).astype(np.float32)
        states_tensor = torch.from_numpy(state_vectors).to(self.device)
        actions_tensor = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if len(non_final_next_states_list) > 0:
            next_positions = np.vstack([s[0] for s in non_final_next_states_list])
            next_steps_one_hot = np.vstack([self._one_hot_encode_steps(s[1]) for s in non_final_next_states_list])
            non_final_next_state_vectors = np.hstack([next_positions, next_steps_one_hot]).astype(np.float32)
            non_final_next_states_tensor = torch.from_numpy(non_final_next_state_vectors).to(self.device)
        else:
            non_final_next_states_tensor = torch.empty(0, self.state_dims, device=self.device)
        return states_tensor, actions_tensor, rewards_tensor, non_final_next_states_tensor, non_final_mask

    def _learn(self):
        if len(self.replay_buffer) < self.h_params['BATCH_SIZE']:
            return None
        self.policy_net.train()
        experiences = self.replay_buffer.sample(self.h_params['BATCH_SIZE'])
        states, actions, rewards, next_states, non_final_mask = self._unpack_and_prep_batch(experiences)
        current_q_values = self.policy_net(states).gather(1, actions)
        next_state_values = torch.zeros(self.h_params['BATCH_SIZE'], 1, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (self.h_params['GAMMA'] * next_state_values)
        loss = self.loss_fn(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def _calculate_epsilon(self, episode: int) -> float:
        decay_period = self.h_params['EPSILON_DECAY_EPISODES']
        if episode >= decay_period:
            return self.h_params['EPSILON_END']
        start_val = self.h_params['EPSILON_START']
        end_val = self.h_params['EPSILON_END']
        frac = episode / decay_period
        return start_val - frac * (start_val - end_val)

    def _calculate_reward(self, dist_before, dist_after, is_success, initial_dist):
        if is_success: return 1.0
        progress_reward = (dist_before - dist_after) / (initial_dist + 1e-6)
        step_penalty = -0.02
        return progress_reward + step_penalty

    def _run_greedy_test(self, starting_cl, ending_cl):
        self.policy_net.eval()
        start_pos = self.env_provider.centroids[starting_cl]
        end_pos = self.env_provider.centroids[ending_cl]
        initial_dist = np.linalg.norm(start_pos - end_pos)
        if initial_dist < 1e-6:
            return 0.0, []
        current_pos = start_pos
        path = []
        for t in range(self.h_params['MAX_PATH_STEPS']):
            steps_remaining = self.h_params['MAX_PATH_STEPS'] - t
            state = (current_pos, steps_remaining)
            action_idx = self._select_action(state, epsilon=0)
            drug_name = self.env_provider.drugs[action_idx]
            path.append(drug_name)
            dists, idxs = self.env_provider.centroid_tree.query(current_pos, k=2)
            weights = 1.0 / (dists + 1e-6)
            weights /= weights.sum()
            centroid_weight_list = [(self.env_provider.centroid_keys[idx], weight) for idx, weight in zip(idxs, weights)]
            disp, _ = self.env_provider.blended_disp(centroid_weight_list, drug_name)
            if disp is None: break
            current_pos = current_pos + disp
        final_dist = np.linalg.norm(current_pos - end_pos)
        return final_dist / initial_dist, path

    def evaluate_performance(self, cell_list: list, target_cl: str) -> dict:
        """
        Runs a comprehensive evaluation in parallel on the CPU.
        """
        # --- FIX: Get the model's state dict on the CPU to make it serializable ---
        cpu_state_dict = {k: v.cpu() for k, v in self.policy_net.state_dict().items()}
        
        num_workers = min(mp.cpu_count(), 8)
        
        # --- FIX: Pass the safe, serializable objects to the workers ---
        init_args = (self.env_provider, self.h_params, cpu_state_dict, target_cl)
        
        with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=init_args) as pool:
            results = list(tqdm(pool.imap(_run_eval_on_cell, cell_list), 
                                total=len(cell_list), 
                                desc="  - Evaluating Cells", 
                                leave=False, ncols=80))

        greedy_metrics, beam_metrics = zip(*results)

        return {
            'greedy_progress': np.mean(greedy_metrics) if greedy_metrics else 0,
            'beam_progress': np.mean(beam_metrics) if beam_metrics else 0
        }

    def train_for_target(self, target_cl: str, training_starts: list, validation_starts: list, fold_num: int):
        EVAL_FREQ = 1000
        all_losses = []
        periodic_eval_log = []

        pbar = trange(self.h_params['TOTAL_TRAINING_EPISODES'], desc=f"Training Fold {fold_num}")
        for episode in pbar:
            # --- Training Step ---
            starting_cl = random.choice(training_starts)
            start_pos = self.env_provider.centroids[starting_cl]
            end_pos = self.env_provider.centroids[target_cl]
            end_radius = self.env_provider.radiuses[target_cl]
            initial_dist = np.linalg.norm(start_pos - end_pos)
            current_pos = start_pos
            for t in range(self.h_params['MAX_PATH_STEPS']):
                steps_remaining = self.h_params['MAX_PATH_STEPS'] - t
                state = (current_pos, steps_remaining)
                epsilon = self._calculate_epsilon(episode)
                action_idx = self._select_action(state, epsilon)
                drug_name = self.env_provider.drugs[action_idx]
                dists, idxs = self.env_provider.centroid_tree.query(current_pos, k=2)
                weights = 1.0 / (dists + 1e-6)
                weights /= weights.sum()
                centroid_weight_list = [(self.env_provider.centroid_keys[idx], weight) for idx, weight in zip(idxs, weights)]
                disp, _ = self.env_provider.blended_disp(centroid_weight_list, drug_name)
                if disp is None: disp = np.zeros_like(current_pos)
                next_pos = current_pos + disp
                dist_to_target = np.linalg.norm(next_pos - end_pos)
                is_success = dist_to_target < end_radius
                done = is_success or (t + 1) >= self.h_params['MAX_PATH_STEPS']
                reward = self._calculate_reward(np.linalg.norm(current_pos - end_pos), dist_to_target, is_success, initial_dist)
                next_state = (next_pos, steps_remaining - 1) if not done else None
                self.replay_buffer.add(state, action_idx, reward, next_state, done)
                current_pos = next_pos
                loss = self._learn()
                if loss is not None: all_losses.append(loss)
                if done: break

            if episode % self.h_params['TARGET_UPDATE_FREQUENCY'] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # --- Comprehensive Evaluation for Diagnostics ---
            if (episode + 1) % EVAL_FREQ == 0 or episode == self.h_params['TOTAL_TRAINING_EPISODES'] - 1:
                print(f"\n--- Running Comprehensive Evaluation at Episode {episode + 1} ---")
                print("Evaluating on Training Set...")
                train_eval = self.evaluate_performance(training_starts, target_cl)
                print("Evaluating on Validation Set...")
                valid_eval = self.evaluate_performance(validation_starts, target_cl)
                
                periodic_eval_log.append({
                    'episode': episode + 1,
                    'train_greedy': train_eval['greedy_progress'],
                    'train_beam': train_eval['beam_progress'],
                    'valid_greedy': valid_eval['greedy_progress'],
                    'valid_beam': valid_eval['beam_progress']
                })
                
                temp_df = pd.DataFrame(periodic_eval_log).set_index('episode')
                print("\n--- Live Overfitting Analysis (Progress %) ---")
                print(temp_df.to_string(formatters={col: '{:,.2%}'.format for col in temp_df.columns}))
                
                avg_loss = np.mean(all_losses[-1000:]) if all_losses else float('nan')
                current_valid_greedy = valid_eval['greedy_progress']
                trange_desc = (f"Fold {fold_num} | Loss: {avg_loss:.5f} | "
                               f"Val Greedy Prog: {current_valid_greedy:.2%}")
                pbar.set_description(trange_desc)

        print(f"\n--- Training Finished for Fold {fold_num} ---")
        
        final_train_eval = self.evaluate_performance(training_starts, target_cl)
        final_valid_eval = self.evaluate_performance(validation_starts, target_cl)
        
        return {
            'fold': fold_num,
            'train_greedy': final_train_eval['greedy_progress'],
            'train_beam': final_train_eval['beam_progress'],
            'valid_greedy': final_valid_eval['greedy_progress'],
            'valid_beam': final_valid_eval['beam_progress'],
            'history': periodic_eval_log
        }
