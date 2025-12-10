import gymnasium as gym
import numpy as np
from gym_wrappers import CarRacingWrapper
import time, pygame, os, datetime
import pickle as pkl

save_dir = "recorded_runs"
os.makedirs(save_dir, exist_ok=True)

def register_input():
    global quit_game, restart_episode, a
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                a[0] = -1.0  
            if event.key == pygame.K_RIGHT:
                a[0] = +1.0  
            if event.key == pygame.K_UP:
                a[1] = +1.0  
            if event.key == pygame.K_SPACE:
                a[2] = +1.0  
            if event.key == pygame.K_RETURN:
                restart_episode = True
            if event.key == pygame.K_ESCAPE:
                quit_game = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                a[0] = 0.0   
            if event.key == pygame.K_RIGHT:
                a[0] = 0.0   
            if event.key == pygame.K_UP:
                a[1] = -1.0  
            if event.key == pygame.K_SPACE:
                a[2] = -1.0  

        if event.type == pygame.QUIT:
            quit_game = True

env = CarRacingWrapper(gym.make("CarRacing-v3", domain_randomize=False, render_mode="human", lap_complete_percent=0.95, continuous=True))
quit_game = False
a = np.array([0.0, -1.0, -1.0])
transitions = []
total_reward = 0
restart_episode = False

state, info = env.reset()

terminated, truncated = False, False
start_time = time.time()

while not terminated and not truncated and not restart_episode and not quit_game:
    register_input()
    
    next_state, reward, terminated, truncated, info = env.step(a)
    
    total_reward += reward
    
    transitions.append((state, a.copy(), reward, next_state, terminated, truncated, info))
    
    state = next_state
    
total_time = time.time() - start_time

if len(transitions) > 10:
    print(f"Episode Finished. Reward: {total_reward:.2f} | Time: {total_time:.1f}s | Transitions: {len(transitions)}")
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_rw-{int(total_reward)}_len-{len(transitions)}.pkl"
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, "wb") as f:
        pkl.dump(transitions, f)
    print(f"Saved: {filepath}")
else:
    print("Run too short, discarded.")


env.close()
print("Recording complete.")