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
import pathlib

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class QNetwork(nn.Module):
    
    def __init__(self, state_dim: int, n_actions: int):
        super(QNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
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

        Args:
            search_environment: An instance of your AverageCellPerturbationSearch class.
            h_params: A dictionary of hyperparameters.
        """
        self.env_provider = search_environment
        self.h_params = h_params
        
        # Determine the device (GPU or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        else:
            self.device = torch.device("cpu")
            print("No GPU available, using CPU")


        # Get state and action dimensions from the environment provider
        self.state_dims = len(self.env_provider.embedding_cols) + 1  # PCA dims + 1 for steps_remaining
        self.n_actions = len(self.env_provider.drugs)

        # Initialize Networks
        self.policy_net = QNetwork(self.state_dims, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.state_dims, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not in training mode

        # Initialize other components
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=h_params['LEARNING_RATE'])
        self.replay_buffer = ReplayBuffer(h_params['REPLAY_BUFFER_CAPACITY'])
        self.loss_fn = nn.SmoothL1Loss()
        
    def _select_action(self, state: tuple, epsilon: float) -> int:
        """Selects an action using an epsilon-greedy policy."""
        if random.random() < epsilon:
            return random.randrange(self.n_actions) # Explore
        else: # Exploit
            with torch.no_grad():
                # Convert state into a tensor for the network
                state_pos, steps_rem = state
                steps_rem_arr = np.array([steps_rem])
                state_vec = np.hstack([state_pos, steps_rem_arr]).astype(np.float32) 
                state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)
                
                # Get Q-values from the policy network and choose the best action
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def _unpack_and_prep_batch(self, experiences: list) -> tuple:
        """Converts a batch of experiences into tensors ready for the network."""
        batch = Experience(*zip(*experiences))

        # --- State Preparation ---
        positions = np.vstack([s[0] for s in batch.state])
        steps_rem = np.array([s[1] for s in batch.state]).reshape(-1, 1)
        state_vectors = np.hstack([positions, steps_rem]).astype(np.float32)
        states_tensor = torch.from_numpy(state_vectors).to(self.device)

        # --- Action, Reward, and Done Preparation ---
        actions_tensor = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        
        # --- Next State Preparation with Defensive Check ---
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        
        # Check if there are any non-terminal states in the batch
        if len(non_final_next_states_list) > 0:
            next_positions = np.vstack([s[0] for s in non_final_next_states_list])
            next_steps_rem = np.array([s[1] for s in non_final_next_states_list]).reshape(-1, 1)
            non_final_next_state_vectors = np.hstack([next_positions, next_steps_rem]).astype(np.float32)
            non_final_next_states_tensor = torch.from_numpy(non_final_next_state_vectors).to(self.device)
        else:
            # If all states were terminal, create an empty tensor with the correct dimensions
            # The subsequent logic in _learn() will handle this correctly.
            non_final_next_states_tensor = torch.empty(0, self.state_dims, device=self.device)

        return states_tensor, actions_tensor, rewards_tensor, non_final_next_states_tensor, non_final_mask

    def _learn(self):
        """Performs one step of learning from a batch of experiences."""
        if len(self.replay_buffer) < self.h_params['BATCH_SIZE']:
            return None # Not enough experiences to learn yet

        experiences = self.replay_buffer.sample(self.h_params['BATCH_SIZE'])
        states, actions, rewards, next_states, non_final_mask = self._unpack_and_prep_batch(experiences)
        
        # 1. Compute Q(s, a) for the actions we actually took
        current_q_values = self.policy_net(states).gather(1, actions) # shape (b,1)
        
        # 2. Compute V(s') for all next states.
        # For terminal states, this value is 0.
        next_state_values = torch.zeros(self.h_params['BATCH_SIZE'], 1, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(next_states).max(1)[0].unsqueeze(1) # before max shape (non_final,n_actions), after max(1) (values, idx) where values shape (non_final,), after [0].unsqueeze(1) shape (non_final, 1)
            
        # 3. Compute the expected Q values (the "y" target)
        expected_q_values = rewards + (self.h_params['GAMMA'] * next_state_values)

        # 4. Compute loss
        loss = self.loss_fn(current_q_values, expected_q_values)
        
        # 5. Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # Prevents exploding gradients
        self.optimizer.step()
        
        return loss.item()



    def train_for_target(self, target_cl: str, starting_cls_list: list = None):
        """
        The main training loop. Trains a Q-network to find paths to a fixed target 
        cell line from a pool of starting cell lines.

        Args:
            target_cl (str): The specific cell line to target.
            starting_cls_list (list, optional): A specific list of cell lines to use as 
                                                starting points for episode sampling. If None,
                                                all other cell lines are used. Defaults to None.
        """
        # --- 1. SETUP FOR THE TRAINING RUN ---
        end_pos = self.env_provider.centroids[target_cl]
        end_radius = self.env_provider.radiuses[target_cl]

        if starting_cls_list is None:
            print("INFO: No starting list provided. Using all other cell lines for sampling.")
            all_cell_lines = self.env_provider.cell_lines
            possible_starting_cls = [cl for cl in all_cell_lines if cl != target_cl]
        else:
            print(f"INFO: Using a specific list of {len(starting_cls_list)} starting cell lines for sampling.")
            possible_starting_cls = [cl for cl in starting_cls_list if cl != target_cl]

        if not possible_starting_cls:
            raise ValueError("The list of possible starting cells is empty. Check your inputs.")

        print(f"--- Starting Target-Centric Training for Target: {target_cl} ---")

        # Initialize tracking variables
        all_losses = []
        best_avg_greedy_metric = float('inf')

        # --- 2. MAIN EPISODIC TRAINING LOOP ---
        for episode in trange(self.h_params['TOTAL_TRAINING_EPISODES'], desc=f"Training to {target_cl}"):

            # --- Start of Episode ---
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
                
                if disp is None:
                    disp = np.zeros_like(current_pos)
                
                next_pos = current_pos + disp
                dist_to_target = np.linalg.norm(next_pos - end_pos)

                is_success = dist_to_target < end_radius
                done = is_success or (t + 1) >= self.h_params['MAX_PATH_STEPS']
                reward = self._calculate_reward(np.linalg.norm(current_pos - end_pos), dist_to_target, is_success, initial_dist)

                next_state = (next_pos, steps_remaining - 1) if not done else None
                self.replay_buffer.add(state, action_idx, reward, next_state, done)
                
                current_pos = next_pos

                loss = self._learn()
                if loss is not None:
                    all_losses.append(loss)
                
                if done:
                    break

            # --- 3. END OF EPISODE MAINTENANCE ---

            if episode % self.h_params['TARGET_UPDATE_FREQUENCY'] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # ====================================================================
            # MODIFIED MONITORING BLOCK (every 100 episodes for less frequent, more stable logging)
            # ====================================================================
            if (episode + 1) % 100 == 0 or episode == self.h_params['TOTAL_TRAINING_EPISODES'] - 1:
                avg_loss = np.mean(all_losses[-1000:]) if all_losses else float('nan')
                
                # --- Robust Evaluation ---
                # Define the set of starting cells to test against. This is the same set used for training.
                validation_starts = possible_starting_cls
                test_metrics = []
                
                print(f"\n--- Running Evaluation at Episode {episode+1} ---")
                for validation_start_cl in validation_starts:
                    dist, _ = self._run_greedy_test(validation_start_cl, target_cl)
                    test_metrics.append(dist)
                
                # Calculate the average performance across all test runs
                avg_test_metric = np.mean(test_metrics)
                
                print(f"Episode {episode+1}/{self.h_params['TOTAL_TRAINING_EPISODES']} | "
                    f"Avg Loss: {avg_loss:.8f} | "
                    f"Epsilon: {epsilon:.3f} | "
                    f"Avg Greedy Test Dist: {avg_test_metric:.4f}")
                
                # Save the model only if the AVERAGE performance has improved
                if avg_test_metric < best_avg_greedy_metric:
                    best_avg_greedy_metric = avg_test_metric

                    # --- PATH CORRECTION FOR SAVING ---
                    # 1. Get the directory where this script (e.g., rl_utils.py) is located
                    script_dir = pathlib.Path(__file__).parent
                    
                    # 2. Build the correct, absolute path to the 'data_and_models' directory
                    save_dir = script_dir.parent / 'data_and_models'
                    
                    # 3. Create the directory if it doesn't exist (this is crucial)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 4. Define the full path for the model file
                    model_path = save_dir / f'dqn_model_{target_cl}.pth'
                    # --- END CORRECTION ---

                    # Save the model that achieved this new best average performance
                    torch.save(self.policy_net.state_dict(), model_path)
                    print(f"  ** New best model saved to '{model_path}' with avg metric: {best_avg_greedy_metric:.4f} **\n")


        print("\n--- Training Finished ---")

    def _calculate_epsilon(self, episode: int) -> float:
        """
        Calculates epsilon for the epsilon-greedy policy with linear decay.
        After the decay period, epsilon remains fixed at its final value.
        """
        decay_period = self.h_params['EPSILON_DECAY_EPISODES']
        
        # If we are past the decay period, return the final epsilon value
        if episode >= decay_period:
            return self.h_params['EPSILON_END']
        
        # Otherwise, perform the linear interpolation
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
        """
        Runs one episode with a purely greedy policy and returns a
        normalized performance metric.
        """
        start_pos = self.env_provider.centroids[starting_cl]
        end_pos = self.env_provider.centroids[ending_cl]
        initial_dist = np.linalg.norm(start_pos - end_pos)
        
        # Handle the edge case where start and end are the same
        if initial_dist < 1e-6:
            return 0.0, [] # 0 distance remaining is a perfect score

        current_pos = start_pos
        path = []

        for t in range(self.h_params['MAX_PATH_STEPS']):
            steps_remaining = self.h_params['MAX_PATH_STEPS'] - t
            state = (current_pos, steps_remaining)
            action_idx = self._select_action(state, epsilon=0) # Epsilon = 0 for greedy
            drug_name = self.env_provider.drugs[action_idx]
            path.append(drug_name)

            dists, idxs = self.env_provider.centroid_tree.query(current_pos, k=2)
            weights = 1.0 / (dists + 1e-6)
            weights /= weights.sum()
            centroid_weight_list = [(self.env_provider.centroid_keys[idx], weight) for idx, weight in zip(idxs, weights)]
            disp, _ = self.env_provider.blended_disp(centroid_weight_list, drug_name)
            if disp is None: break
            current_pos = current_pos + disp
        
        # Calculate the final distance and the normalized metric
        final_dist = np.linalg.norm(current_pos - end_pos)
        
        return final_dist / initial_dist, path