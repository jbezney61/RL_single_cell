import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super(QNetwork, self).__init__()
        # 256-128 Bottleneck for global navigation capacity
        self.layers = nn.Sequential(            
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, n_actions)
        )
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)

class DebugTrainer:
    def __init__(self, env, h_params):
        self.env = env
        self.hp = h_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dims = len(env.embedding_cols) + h_params['MAX_PATH_STEPS']
        self.n_actions = len(env.drugs)
        
        self.policy_net = QNetwork(self.state_dims, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.state_dims, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # KEY: Target net must stay in eval mode to provide stable labels
        self.target_net.eval() 
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=h_params['LEARNING_RATE'], weight_decay=1e-4)
        self.replay_buffer = deque(maxlen=h_params['REPLAY_BUFFER_CAPACITY'])
        self.loss_fn = nn.SmoothL1Loss()

    def get_state_tensor(self, pos, steps_rem):
        one_hot = np.zeros(self.hp['MAX_PATH_STEPS'])
        if steps_rem > 0: one_hot[steps_rem - 1] = 1
        vec = np.hstack([pos, one_hot]).astype(np.float32)
        return torch.from_numpy(vec).unsqueeze(0).to(self.device)

    def select_action(self, pos, steps_rem, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        
        self.policy_net.eval() # Ensure dropout is off for evaluation/selection
        with torch.no_grad():
            state_t = self.get_state_tensor(pos, steps_rem)
            return self.policy_net(state_t).argmax().item()

    def learn(self):
        if len(self.replay_buffer) < self.hp['BATCH_SIZE']:
            return None
            
        self.policy_net.train() # Ensure dropout is on for learning
        batch = random.sample(self.replay_buffer, self.hp['BATCH_SIZE'])
        states_p, states_s, actions, rewards, next_p, next_s, dones = zip(*batch)

        pos_t = np.vstack(states_p)
        steps_t = np.vstack([self._one_hot_raw(s) for s in states_s])
        s_t = torch.from_numpy(np.hstack([pos_t, steps_t]).astype(np.float32)).to(self.device)
        
        # Jitter applied to current states only for robustness
        emb_dim = len(self.env.embedding_cols)
        s_t[:, :emb_dim] += torch.randn_like(s_t[:, :emb_dim]) * self.hp['JITTER_STD']

        a_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        
        current_q = self.policy_net(s_t).gather(1, a_t)
        next_q = torch.zeros(self.hp['BATCH_SIZE'], 1, device=self.device)
        
        mask = torch.tensor([not d for d in dones], dtype=torch.bool, device=self.device)
        if mask.any():
            # KEY: Wrapped in no_grad for stability and speed
            with torch.no_grad():
                nf_p = np.vstack([p for p, d in zip(next_p, dones) if not d])
                nf_s = np.vstack([self._one_hot_raw(s) for s, d in zip(next_s, dones) if not d])
                nf_t = torch.from_numpy(np.hstack([nf_p, nf_s]).astype(np.float32)).to(self.device)
                next_q[mask] = self.target_net(nf_t).max(1)[0].unsqueeze(1)
            
        expected_q = r_t + (self.hp['GAMMA'] * next_q)
        loss = self.loss_fn(current_q, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Explicit clipping to 100 to prevent 'messy' divergent weight updates
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()

    def _one_hot_raw(self, steps_rem):
        vec = np.zeros(self.hp['MAX_PATH_STEPS'])
        if steps_rem > 0: vec[steps_rem - 1] = 1
        return vec