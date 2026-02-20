import numpy as np
import torch
from collections import deque


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    Prioritizes experiences based on TD error for more efficient learning
    """
    
    def __init__(self, maxlen=2000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            maxlen: Maximum buffer size
            alpha: How much prioritization to use (0 = uniform, 1 = full priority)
            beta_start: Initial importance sampling weight (compensates for bias)
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon = 1e-6  # Small constant to avoid zero priority
        
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        """
        Add experience to buffer with maximum priority
        experience: (state, action, reward, next_state, done)
        """
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(experience)
        self.priorities.append(max_priority)
    
    def sample(self, batch_size):
        """
        Sample batch of experiences based on priorities
        
        Returns:
            List of numpy arrays: [states, actions, rewards, next_states, dones]
            indices: Indices of sampled experiences (for updating priorities)
            weights: Importance sampling weights
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        beta = self.beta_start + (1.0 - self.beta_start) * min(1.0, self.frame / self.beta_frames)
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights
        
        self.frame += 1
        
        # Return in the format: [states, actions, rewards, next_states, dones]
        experiences = [
            torch.stack([torch.as_tensor(experience[field_index]) for experience in batch])
            for field_index in range(5)
        ]
        
        return experiences, indices, torch.from_numpy(weights).float()
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities for sampled experiences based on TD errors
        
        Args:
            indices: Indices of experiences to update
            td_errors: TD errors for corresponding experiences
        """
        for idx, td_error in zip(indices, td_errors):
            # Store raw priority (TD error + epsilon), alpha applied during sampling
            priority = abs(td_error) + self.epsilon
            self.priorities[idx] = priority


class ReplayBuffer:
    """
    Standard uniform replay buffer (for comparison)
    """
    
    def __init__(self, maxlen=2000):
        self.buffer = deque(maxlen=maxlen)
    
    def __len__(self):
        return len(self.buffer)
    
    def get_exp(self):
        return [
            torch.tensor([experience[field_index] for experience in self.buffer])
            for field_index in range(5)
        ]
    
    def empty_buffer(self):
        self.buffer.clear()
    
    def append(self, experience):
        """
        Add experience to buffer
        experience: (state, action, reward, next_state, done)
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample batch of experiences uniformly
        
        Returns:
            List of torch tensors: [states, actions, rewards, next_states, dones]
        """
        indices = np.random.randint(len(self.buffer), size=batch_size)
        batch = [self.buffer[index] for index in indices]
        
        return [
            torch.stack([torch.as_tensor(experience[field_index]) for experience in batch])
            for field_index in range(5)
        ]


if __name__ == "__main__":
    # Test the prioritized replay buffer
    print("Testing PrioritizedReplayBuffer...")
    
    buffer = PrioritizedReplayBuffer(maxlen=100)
    
    # Add some dummy experiences
    for i in range(50):
        state = np.random.rand(42)
        action = np.random.randint(0, 7)
        reward = np.random.rand()
        next_state = np.random.rand(42)
        done = np.random.rand() > 0.9
        
        buffer.append((state, action, reward, next_state, done))
    
    print(f"Buffer size: {len(buffer)}")
    
    # Sample a batch
    experiences, indices, weights = buffer.sample(batch_size=8)
    states, actions, rewards, next_states, dones = experiences
    
    print(f"Batch shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Next states: {next_states.shape}")
    print(f"  Dones: {dones.shape}")
    print(f"  Weights: {weights.shape}")
    print(f"  Sampled indices: {indices}")
    
    # Update priorities with dummy TD errors
    td_errors = np.random.rand(8) * 2  # Random TD errors
    buffer.update_priorities(indices, td_errors)
    print("\nPriorities updated successfully!")
