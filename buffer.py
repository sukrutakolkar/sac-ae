import numpy as np
import torch, os, pickle

class ReplayBuffer:
    def __init__(self, obs_shape, action_dim, capacity, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # pre-allocate memory 
        self.state = np.ascontiguousarray(np.zeros((capacity, *obs_shape), dtype=np.uint8))
        self.nxt_state = np.ascontiguousarray(np.zeros((capacity, *obs_shape), dtype=np.uint8))
        self.action = np.ascontiguousarray(np.zeros((capacity, action_dim), dtype=np.float32))
        self.reward = np.ascontiguousarray(np.zeros((capacity, 1), dtype=np.float32))
        self.done = np.ascontiguousarray(np.zeros((capacity, 1), dtype=np.float32))

    def add(self, obs, action, reward, next_obs, done):
        self.state[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.nxt_state[self.ptr] = next_obs
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.as_tensor(self.state[ind], device=self.device).float(),
            torch.as_tensor(self.action[ind], device=self.device).float(),
            torch.as_tensor(self.reward[ind], device=self.device).float(),
            torch.as_tensor(self.nxt_state[ind], device=self.device).float(),
            1.0 - torch.as_tensor(self.done[ind], device=self.device).float(),
        )

    def save(self, path): # TODO: use .npz instead of pickle 
        data = {
            'obs': self.state[:self.size],
            'next_obs': self.nxt_state[:self.size],
            'action': self.action[:self.size],
            'reward': self.reward[:self.size],
            'not_done': self.done[:self.size],
            'ptr': self.ptr
        }
        with open(os.path.join(path, 'buffer.pkl'), 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, path):
        path = os.path.join(path, 'buffer.pkl')
        assert os.path.exists(path)

        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        load_size = len(data['obs'])
        self.state[:load_size] = data['obs']
        self.nxt_state[:load_size] = data['next_obs']
        self.action[:load_size] = data['action']
        self.reward[:load_size] = data['reward']
        self.done[:load_size] = data['not_done']
        self.ptr = data['ptr']
        self.size = load_size

class SmoothedReplayBuffer(ReplayBuffer):
    # EMA over rewards stabilizes policy, DreamSmooth - https://arxiv.org/abs/2311.01450
    def __init__(self, obs_shape, action_dim, capacity, device, alpha=0.1):
        super().__init__(obs_shape, action_dim, capacity, device)
        self.alpha = alpha
        self.ema_reward = 0.0
        self.reset_ema = True 

    def add(self, obs, action, reward, next_obs, done):
        if self.reset_ema:
            self.ema_reward = reward
            self.reset_ema = False
        else:
            # R'_t = alpha * R_t + (1 - alpha) * R'_{t-1}
            self.ema_reward = self.alpha * reward + (1 - self.alpha) * self.ema_reward
        
        super().add(obs, action, self.ema_reward, next_obs, done)

        if done:
            self.reset_ema = True

    def reset(self):
        self.reset_ema = True