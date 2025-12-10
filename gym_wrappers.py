import gymnasium as gym
import numpy as np
from collections import deque
import cv2

class BasePixelStateWrapper(gym.Wrapper):
    def __init__(self, env, num_stack=4, frame_shape=(84, 84), grayscale=True, num_repeat=3, initial_skip=0, initial_action=None) :
        super().__init__(env)
        self.env = env
        self.frame_shape = frame_shape
        self.grayscale = grayscale
        self.num_stack = num_stack
        self.num_repeat = num_repeat
        self.initial_skip = initial_skip

        if initial_action is None:
            self.initial_action = np.zeros(env.action_space.shape) if hasattr(env.action_space, 'shape') else 0
        else:
            self.initial_action = initial_action

        self.frames = deque([], maxlen=num_stack)

        channels = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(num_stack * channels, frame_shape[0], frame_shape[1]),
            dtype=np.uint8
        )

    def _get_observation(self):
        # stack frames along the channel dimension
        assert len(self.frames) == self.num_stack
        return np.transpose(np.concatenate(self.frames, axis=2), (2, 0, 1))
    
    def preprocess_frame(self, frame: np.ndarray):
        #frame = np.array(frame)
        if frame.shape[:2] != self.frame_shape:
            frame = cv2.resize(frame, dsize=(self.frame_shape[1], self.frame_shape[0]), interpolation=cv2.INTER_AREA)

        if self.grayscale and len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.expand_dims(frame, axis=-1)  # Add channel dimension 
        #frame = frame[:84, 6:90, :]            # Crop bottom status bar

        return frame 
    
    def reset(self, **kwargs):
        self.frames.clear()
        obs, info = self.env.reset(**kwargs)

        for i in range(self.initial_skip + self.num_stack):
            obs, _, _, _, _ = self.env.step(self.initial_action)
            if i >= self.initial_skip: self.frames.append(self.preprocess_frame(obs))

        return self._get_observation(), info
    
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for _ in range(self.num_repeat):
            obs, reward, term, trunc, i = self.env.step(action)
            total_reward += reward
            info.update(i)
            
            if term or trunc:
                terminated = term
                truncated = trunc
                break
            
        self.frames.append(self.preprocess_frame(obs))
        
        return self._get_observation(), total_reward, terminated, truncated, info

class CarRacingWrapper(BasePixelStateWrapper):
    def __init__(self, env, early_truncation=True, reward_threshold=-30, check_interval=100, **kwargs):
        super().__init__(env, initial_action=np.array([0.0, 0.1, 0.0]), **kwargs)
        
        self.reward_threshold = reward_threshold
        self.check_interval = check_interval     
        self.reward_buffer = deque([], maxlen=check_interval)
        self.early_truncation = early_truncation
        self._max_episode_steps = env.spec.max_episode_steps

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32
        )
        self.true_action_delta = self.env.action_space.high - self.env.action_space.low
        self.norm_action_delta = self.action_space.high - self.action_space.low

    def reset(self, **kwargs):
        self.reward_buffer.clear()
        return super().reset(**kwargs)

    def step(self, action):
        action = self._standarize_action(action)
        
        obs, total_reward, terminated, truncated, info = super().step(action)

        self.reward_buffer.append(total_reward)

        # Early truncation - check if total reward over the last 250 steps is less than the allowed threshold
        if self.early_truncation and len(self.reward_buffer) == self.reward_buffer.maxlen and sum(self.reward_buffer) < self.reward_threshold:
            total_reward -= 10                   # Apply additional penalty
            terminated = True                    # End the episode
            info['early_termination'] = True    # additional info for logging
   
        return (obs, total_reward, terminated, truncated, info)

    def preprocess_frame(self, frame):
        frame = np.array(frame)
        frame = frame[:84, 6:90, :]            # Crop bottom status bar
        return super().preprocess_frame(frame) 
    
    def _standarize_action(self, action: np.ndarray):
        action = action.astype(np.float64).clip(-1.0, 1.0)
        action = (action - self.action_space.low) / self.norm_action_delta
        action = action * self.true_action_delta + self.env.action_space.low
        return action.astype(np.float32)
    
    
class VisualPendulumWrapper(BasePixelStateWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env, initial_action=np.array([0.0]), **kwargs)
        
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32
        )
        
        self.true_action_delta = self.env.action_space.high - self.env.action_space.low
        self.norm_action_delta = self.action_space.high - self.action_space.low

    def reset(self, **kwargs):
        self.frames.clear()
        _, info = self.env.reset(**kwargs)

        start_frame = self.env.render()
        processed_frame = self.preprocess_frame(start_frame)

        for _ in range(self.num_stack):
            self.frames.append(processed_frame)

        return self._get_observation(), info

    def step(self, action):
        action = self._standarize_action(action)
        
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for _ in range(self.num_repeat):
            _, reward, term, trunc, i = self.env.step(action)
            total_reward += reward
            info.update(i)
            
            if term or trunc:
                terminated = term
                truncated = trunc
                break
        
        frame = self.env.render()
        self.frames.append(self.preprocess_frame(frame))
        
        return self._get_observation(), total_reward, terminated, truncated, info

    def _standarize_action(self, action: np.ndarray):
        action = action.astype(np.float64).clip(-1.0, 1.0)
        action = (action - self.action_space.low) / self.norm_action_delta
        action = action * self.true_action_delta + self.env.action_space.low
        return action.astype(np.float32)