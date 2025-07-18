import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from collections import namedtuple
from tqdm import trange
from search_utils import AverageCellPerturbationSearch

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class QNetwork(nn.Module):
    
    def __init__(self, state_dim: int, n_actions: int):
        super(QNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, n_actions)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)
    

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state: tuple, action: int, reward:float, next_state: tuple, done: bool):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
class DQNTrainer:
    """
    Orchestrates the DQN training process for a specific A-to-B conversion.
    """
    def __init__(self, search_environment: AverageCellPerturbationSearch, h_params: dict):
        """
        Initializes the trainer and all necessary components for the RL agent.
        """
        self.env_provider = search_environment
        self.h_params = h_params
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        else:
            self.device = torch.device("cpu")
            print("No GPU available, using CPU")

        self.state_dims = len(self.env_provider.embedding_cols) + self.h_params['MAX_PATH_STEPS']
        
        # --- MODIFIED: Added 1 to the action count for the "no-op" action. ---
        # This new action allows the agent to terminate the episode.
        self.n_actions = len(self.env_provider.drugs) + 1

        self.policy_net = QNetwork(self.state_dims, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.state_dims, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=h_params['LEARNING_RATE'])
        self.replay_buffer = ReplayBuffer(h_params['REPLAY_BUFFER_CAPACITY'])
        self.loss_fn = nn.SmoothL1Loss()
        
    def _one_hot_encode_steps(self, steps_remaining: int) -> np.ndarray:
        """Helper to create a one-hot vector for steps remaining."""
        one_hot = np.zeros(self.h_params['MAX_PATH_STEPS'])
        if steps_remaining > 0:
            one_hot[steps_remaining - 1] = 1
        return one_hot
        
    def _select_action(self, state: tuple, epsilon: float) -> int:
        """Selects an action using an epsilon-greedy policy."""
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
        """Converts a batch of experiences into tensors ready for the network."""
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
        """Performs one step of learning from a batch of experiences."""
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

    def train_for_target(self, target_cl: str, training_starts: list, validation_starts: list, fold_num: int):
        """
        The main training loop. Trains a Q-network using a dedicated
        training set and evaluates it on a hold-out validation set.
        """
        end_pos = self.env_provider.centroids[target_cl]
        end_radius = self.env_provider.radiuses[target_cl]

        if not training_starts: raise ValueError("The training_starts list cannot be empty.")
        if not validation_starts: raise ValueError("The validation_starts list cannot be empty.")

        print(f"--- Starting Training for Target: {target_cl} (Fold {fold_num}) ---")
        print(f"Training with {len(training_starts)} cells, Validating with {len(validation_starts)} cells.")

        all_losses = []
        best_avg_greedy_metric = float('inf')

        pbar = trange(self.h_params['TOTAL_TRAINING_EPISODES'], desc=f"Training Fold {fold_num}")
        for episode in pbar:
            starting_cl = random.choice(training_starts)
            start_pos = self.env_provider.centroids[starting_cl]
            initial_dist = np.linalg.norm(start_pos - end_pos)
            current_pos = start_pos
            
            for t in range(self.h_params['MAX_PATH_STEPS']):
                steps_remaining = self.h_params['MAX_PATH_STEPS'] - t
                state = (current_pos, steps_remaining)
                epsilon = self._calculate_epsilon(episode)
                action_idx = self._select_action(state, epsilon)
                
                # --- MODIFIED BLOCK: Handle the "no-op" action ---
                # The last action index is now the no-op action.
                if action_idx == len(self.env_provider.drugs):
                    next_pos = current_pos  # Position does not change.
                    dist_to_target = np.linalg.norm(next_pos - end_pos)
                    is_success = dist_to_target < end_radius
                    # Episode terminates immediately upon choosing "no-op".
                    done = True
                else:
                    # Original logic for applying a drug.
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
                # --- END OF MODIFIED BLOCK ---
                
                reward = self._calculate_reward(np.linalg.norm(current_pos - end_pos), dist_to_target, is_success, initial_dist)
                next_state = (next_pos, steps_remaining - 1) if not done else None
                
                self.replay_buffer.add(state, action_idx, reward, next_state, done)
                current_pos = next_pos
                loss = self._learn()
                if loss is not None: all_losses.append(loss)
                if done: break

            if episode % self.h_params['TARGET_UPDATE_FREQUENCY'] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if (episode + 1) % 100 == 0 or episode == self.h_params['TOTAL_TRAINING_EPISODES'] - 1:
                avg_loss = np.mean(all_losses[-1000:]) if all_losses else float('nan')
                
                test_metrics = []
                for validation_start_cl in validation_starts:
                    dist, _ = self._run_greedy_test(validation_start_cl, target_cl)
                    test_metrics.append(dist)
                
                avg_test_metric = np.mean(test_metrics)
                
                trange_desc = (f"Fold {fold_num} | Loss: {avg_loss:.5f} | "
                               f"Eps: {epsilon:.3f} | Val Dist: {avg_test_metric:.4f}")
                
                if avg_test_metric < best_avg_greedy_metric:
                    best_avg_greedy_metric = avg_test_metric
                    model_path = f'../data_and_models/dqn_model_{target_cl}_fold{fold_num}.pth'
                    torch.save(self.policy_net.state_dict(), model_path)
                    trange_desc += " (New best model saved!)"
                
                pbar.set_description(trange_desc)

        print(f"\n--- Training Finished for Fold {fold_num} ---")
        return best_avg_greedy_metric

    def _calculate_epsilon(self, episode: int) -> float:
        """Calculates epsilon for the epsilon-greedy policy with linear decay."""
        decay_period = self.h_params['EPSILON_DECAY_EPISODES']
        if episode >= decay_period:
            return self.h_params['EPSILON_END']
        start_val = self.h_params['EPSILON_START']
        end_val = self.h_params['EPSILON_END']
        frac = episode / decay_period
        return start_val - frac * (start_val - end_val)

    def _calculate_reward(self, dist_before, dist_after, is_success, initial_dist):
        """Calculates a normalized, potential-based reward."""
        if is_success: return 1.0
        progress_reward = (dist_before - dist_after) / (initial_dist + 1e-6)
        step_penalty = -0.02
        return progress_reward + step_penalty
    
    def _run_greedy_test(self, starting_cl, ending_cl):
        """Runs one episode with a purely greedy policy."""
        self.policy_net.eval()
        
        start_pos = self.env_provider.centroids[starting_cl]
        end_pos = self.env_provider.centroids[ending_cl]
        initial_dist = np.linalg.norm(start_pos - end_pos)
        
        if initial_dist < 1e-6: 
            self.policy_net.train()
            return 0.0, []

        current_pos = start_pos
        path = []

        for t in range(self.h_params['MAX_PATH_STEPS']):
            steps_remaining = self.h_params['MAX_PATH_STEPS'] - t
            state = (current_pos, steps_remaining)
            action_idx = self._select_action(state, epsilon=0)

            # --- MODIFIED BLOCK: Handle the "no-op" action in testing ---
            # Check if the chosen action is the "no-op" action.
            if action_idx == len(self.env_provider.drugs):
                path.append("NO-OP")  # Log that the agent stopped.
                break  # Terminate the path.
            else:
                # Original logic for applying a drug.
                drug_name = self.env_provider.drugs[action_idx]
                path.append(drug_name)

                dists, idxs = self.env_provider.centroid_tree.query(current_pos, k=2)
                weights = 1.0 / (dists + 1e-6)
                weights /= weights.sum()
                centroid_weight_list = [(self.env_provider.centroid_keys[idx], weight) for idx, weight in zip(idxs, weights)]
                disp, _ = self.env_provider.blended_disp(centroid_weight_list, drug_name)
                if disp is None: break
                current_pos = current_pos + disp
            # --- END OF MODIFIED BLOCK ---
        
        final_dist = np.linalg.norm(current_pos - end_pos)
        
        self.policy_net.train()
        
        return final_dist / initial_dist, path