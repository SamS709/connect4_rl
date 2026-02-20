import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Force TensorFlow to use CPU (RTX 5070 Ti compute capability 12.0 not yet fully supported)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import random
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from collections import deque
from scripts.env import Env
from scripts.Connect4 import *
from scripts.logger import Logger
from scripts.ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer

# Custom layer for Dueling DQN aggregation


class DuelingDDQNModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 6 * 7 = 42 length and 3 channels (0, 1, 2) -> empty, p1, p2
        num_actions = 7
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, stride = 1, padding = 1), # output_length = 42
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels= 64, kernel_size=3, stride = 1, padding = 1), # output_length = 42
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.dense = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64 * 42, 256),
            nn.ReLU(),
        )
        self.a = nn.Linear(256, num_actions)
        self.v_s = nn.Linear(256,1)
    
    def forward(self, X):
        X = F.one_hot(X, num_classes=3).float().permute([0, 3, 1, 2])
        X = self.layers(X)
        X = self.dense(X.flatten(1))
        return self.v_s(X) + self.a(X)

class DuelingDDQN(Connect4):

    def __init__(self, model_name, softmax_=False, n_layers=2, n_neurons=32, learning_rate=1e-2, gamma=1e-1, eps=0.9, P1="1", reset=False, use_prioritized=True, alpha=0.6, beta_start=0.4, beta_frames=100000):

        super().__init__()
        self.game_rewards = []
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        
        # Initialize replay buffer (prioritized or standard)
        if use_prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(
                maxlen=2000,
                alpha=alpha,
                beta_start=beta_start,
                beta_frames=beta_frames
            )
            self.use_prioritized = True
        else:
            self.replay_buffer = ReplayBuffer(2000)
            self.use_prioritized = False
        
        self.epsilon = eps
        self.gamma = gamma
        self.P1 = P1
        self.batch_size = 32
        self.loss_fn = nn.MSELoss(reduction='none')  # Per-sample loss for prioritized replay
        
        self.model_name = model_name
        self.path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", self.model_name)
        self.model_path = os.path.join(self.path, self.model_name+self.P1+".pt")

        try:
            print("SUCCESSFULLY LOADED MODEL ", self.model_name)
            
            self.load_model()
        except:
            print("SUCCESSFULLY CREATED MODEL ", self.model_name)
            self.create_model()
            self.save()
        self.__str__()
        self.logger = Logger(model_name,P1)
        self.logger.set_algo_name(self.get_algo_name())
        self.optimizer = torch.optim.Adam(params = self.model.parameters(),lr = 0.001)
        

    def __str__(self):
        print(self.model)
        return str(self.model)
    
    def get_algo_name(self):
        return "DDQN"

    def save(self):
        self.target.load_state_dict(self.model.state_dict())
        torch.save(self.model, self.model_path)
        

    def create_model(self):
        self.model = DDQNModel()
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(self.model, self.model_path)
        self.load_model()
        
    def load_model(self):
        self.model = torch.load(self.model_path, weights_only= False)
        self.target = torch.load(self.model_path, weights_only= False)
        self.target.load_state_dict(self.model.state_dict())


    def epsilon_greedy(self, state: torch.tensor, eps = None):
        epsilon = self.epsilon if eps==None else eps
        if np.random.rand()<epsilon: 
            return np.random.randint(0,7)
        else:
            Q_values = self.model(state.unsqueeze(0))[0]
            #np.newaxis augment la dimension de state
            #verbose = 0 => don't show the progress bar of evaluating
            return torch.argmax(Q_values)
    
    def select_action(self, state: torch.tensor, eps = None):
        return self.epsilon_greedy(state,eps)
    
    def get_sorted_actions(self, state: torch.tensor):
        """Returns actions (indices) sorted by Q-values in descending order"""
        Q_values = self.model(state.unsqueeze(0))[0]
        # argsort gives indices in ascending order, so we reverse with [::-1]
        return torch.argsort(Q_values, descending=True)
    
    def get_action_probabilities(self, state: torch.tensor):
        """Returns Q-values (probabilities) and corresponding sorted actions"""
        Q_values = self.model(state.unsqueeze(0))[0]
        sorted_actions = torch.argsort(Q_values, descending=True)
        sorted_probabilities = Q_values[sorted_actions]
        return sorted_actions, sorted_probabilities

    
    def sample_experiences(self):
        if self.use_prioritized:
            # Prioritized sampling returns: experiences, indices, weights
            experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
            return experiences, indices, weights
        else:
            # Standard uniform sampling
            experiences = self.replay_buffer.sample(self.batch_size)
            return experiences, None, None  # No indices/weights for uniform
    
    def training_step(self):
        
        self.model.train()
        experiences, indices, weights = self.sample_experiences()
        states, actions, rewards, next_states, runs = experiences
        
        # Compute Q-values and targets
        actual_batch_size = states.shape[0]
        next_actions = torch.argmax(self.model(next_states), axis=1)
        next_Q_values = self.target(next_states)  # using target network
        max_next_Q_values = next_Q_values[torch.arange(actual_batch_size), next_actions]
        target_Q_values = rewards + runs * self.gamma * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1,1)
        mask = F.one_hot(actions, 7)

        all_Q_values = self.model(states)
        Q_values = torch.sum(all_Q_values * mask, axis=1, keepdims=True)
        td_errors = target_Q_values - Q_values
        
        if self.use_prioritized:
            # Compute per-sample losses and weight them
            sample_losses = self.loss_fn(Q_values, target_Q_values).squeeze()
            weighted_loss = torch.mean(weights.to(Q_values.device) * sample_losses)
            weighted_loss.backward()
        else:
            loss = torch.mean(self.loss_fn(Q_values, target_Q_values))
            loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update priorities in replay buffer
        if self.use_prioritized:
            td_errors_np = torch.abs(td_errors.detach().flatten())
            self.replay_buffer.update_priorities(indices, td_errors_np)


if __name__=="__main__":
    dqn = DuelingDDQN("torch_ddqn",eps=0.5)
    state = torch.tensor([0 for i in range(42)])
    dqn.select_action(state.unsqueeze(0),0.3)

