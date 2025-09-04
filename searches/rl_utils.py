import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from collections import namedtuple
from tqdm import trange
from search_utils import AverageCellPerturbationSearch # Assuming this is in a local file
import pathlib

# Define the structure for an experience
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

# --- CHANGE 1: USE THE QNETWORK WITH DROPOUT ---
# This is the network architecture from your tuning script, which includes
# dropout for regularization to prevent overfitting.
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

        # --- CHANGE 2: USE ONE-HOT ENCODING (PART 1 - STATE DIMENSION) ---
        # The state dimension is updated to reflect the length of the one-hot vector
        # for steps_remaining, not just a single integer.
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
                # --- CHANGE 2: USE ONE-HOT ENCODING (PART 2 - STATE PREPARATION) ---
                steps_one_hot = self._one_hot_encode_steps(steps_rem)
                state_vec = np.hstack([state_pos, steps_one_hot]).astype(np.float32) 
                state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)
                
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def _unpack_and_prep_batch(self, experiences: list) -> tuple:
        """Converts a batch of experiences into tensors ready for the network."""
        batch = Experience(*zip(*experiences))

        # --- CHANGE 2: USE ONE-HOT ENCODING (PART 3 - BATCH PREPARATION) ---
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
        
        # --- CHANGE 3: MODE SWITCHING (PART 1 - SET TO TRAIN) ---
        # This enables dropout layers during the learning step.
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

    def train_for_target(self, target_cl: str, starting_cls_list: list = None):
        """
        The main training loop. Trains a Q-network to find paths to a fixed target 
        cell line from a pool of starting cell lines.
        """
        end_pos = self.env_provider.centroids[target_cl]
        end_radius = self.env_provider.radiuses[target_cl]

        if starting_cls_list is None:
            possible_starting_cls = [cl for cl in self.env_provider.cell_lines if cl != target_cl]
        else:
            possible_starting_cls = [cl for cl in starting_cls_list if cl != target_cl]

        if not possible_starting_cls:
            raise ValueError("The list of possible starting cells is empty.")

        print(f"--- Starting Target-Centric Training for Target: {target_cl} ---")
        all_losses = []
        best_avg_greedy_metric = float('inf')

        pbar = trange(self.h_params['TOTAL_TRAINING_EPISODES'], desc=f"Training to {target_cl}")
        for episode in pbar:
            starting_cl = random.choice(possible_starting_cls)
            start_pos = self.env_provider.centroids[starting_cl]
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

            if (episode + 1) % 500 == 0 or episode == self.h_params['TOTAL_TRAINING_EPISODES'] - 1:
                avg_loss = np.mean(all_losses[-1000:]) if all_losses else float('nan')
                
                test_metrics = []
                for validation_start_cl in possible_starting_cls:
                    dist, _ = self._run_greedy_test(validation_start_cl, target_cl)
                    test_metrics.append(dist)
                
                avg_test_metric = np.mean(test_metrics)
                
                trange_desc = (f"Target: {target_cl} | Loss: {avg_loss:.5f} | "
                               f"Eps: {epsilon:.3f} | Val Dist: {avg_test_metric:.4f}")
                
                if avg_test_metric < best_avg_greedy_metric:
                    best_avg_greedy_metric = avg_test_metric
                    
                    script_dir = pathlib.Path(__file__).parent
                    save_dir = script_dir.parent / 'data_and_models'
                    save_dir.mkdir(parents=True, exist_ok=True)
                    model_path = save_dir / f'dqn_model_{target_cl}.pth'
                    
                    torch.save(self.policy_net.state_dict(), model_path)
                    trange_desc += " (New best model saved!)"
                
                pbar.set_description(trange_desc)

        print("\n--- Training Finished ---")

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
        # --- CHANGE 3: MODE SWITCHING (PART 2 - SET TO EVAL) ---
        # This disables dropout layers for deterministic evaluation.
        self.policy_net.eval()
        
        start_pos = self.env_provider.centroids[starting_cl]
        end_pos = self.env_provider.centroids[ending_cl]
        initial_dist = np.linalg.norm(start_pos - end_pos)
        
        if initial_dist < 1e-6: 
            self.policy_net.train() # Ensure we switch back before returning
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
        
        # --- CHANGE 3: MODE SWITCHING (PART 3 - SET BACK TO TRAIN) ---
        # It's good practice to set the model back to train mode after evaluation.
        self.policy_net.train()
        
        return final_dist / initial_dist, path
