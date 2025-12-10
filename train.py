import numpy as np
import torch, os, argparse, time, cv2, glob, pickle
import gymnasium as gym
from datetime import datetime

from gym_wrappers import CarRacingWrapper, VisualPendulumWrapper
from sac import PixelSAC
from buffer import ReplayBuffer, SmoothedReplayBuffer
from logger import Logger, ILogger

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Training Settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_steps', type=int, default=500000)
    parser.add_argument('--exploration_steps', type=int, default=5000) 
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--buffer_capacity', default=100000, type=int)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_freq', type=int, default=2500)
    parser.add_argument('--eval_episodes', type=int, default=3)
    parser.add_argument('--enable_logging', default=True, action='store_true')
    parser.add_argument('--log_freq', type=int, default=2500)
    parser.add_argument('--record_video', default=True, action='store_true')
    parser.add_argument('--reward_smoothing', default=True, action='store_true')
    parser.add_argument('--reward_smoothing_alpha', default=0.1, type=float)

    # Env Settings
    parser.add_argument('--env', default='CarRacing', type=str, help="CarRacing / VisualPendulum")
    parser.add_argument('--env_num_stack', type=int, default=4)
    parser.add_argument('--env_num_repeat', type=int, default=3)

    # Checkpointing
    parser.add_argument('--checkpoint_freq', type=int, default=5000)
    parser.add_argument('--save_dir', default='runs', type=str)
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_buffer', default=True, action='store_true')
    parser.add_argument('--save_video', default=True, action='store_true')

    # Offpolicy 
    parser.add_argument('--offpolicy_pretraining', default=True, action='store_true')
    parser.add_argument('--recorded_runs_path', default='.\\recorded_runs', type=str)
    parser.add_argument('--offpolicy_steps', type=int, default=5000)
    
    # Actor Settings
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--log_std_min', default=-20, type=float)
    parser.add_argument('--log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    parser.add_argument('--actor_hidden_dim', default=1024, type=int)

    # Critic Settings
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    parser.add_argument('--critic_hidden_dim', default=1024, type=int)

    # AE
    parser.add_argument('--encoder_latent_dim', default=64, type=int)
    parser.add_argument('--encoder_lr', default=5e-4, type=float)
    parser.add_argument('--ae_update_freq', type=int, default=1)
    parser.add_argument('--ae_num_layers', default=4, type=int)
    parser.add_argument('--ae_num_filters', default=32, type=int)
    parser.add_argument('--ae_mask_ratio', type=float, default=0.0) 
    parser.add_argument('--random_shift_pad', default=4, type=int)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--ae_log_freq', type=int, default=2500)

    # SAC params
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--init_alpha', default=0.5, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--tau', default=0.0075, type=float)

    args = parser.parse_args()

    config = {
        # General
        'seed': args.seed,
        'num_steps': args.num_steps,
        'exploration_steps': args.exploration_steps,
        'batch_size': args.batch_size,
        'buffer_capacity': args.buffer_capacity,
        'device': args.device,
        'eval_freq': args.eval_freq,
        'eval_episodes': args.eval_episodes,
        'enable_logging': args.enable_logging,
        'log_freq': args.log_freq,
        'record_video': args.record_video,
        'reward_smoothing': args.reward_smoothing,
        'reward_smoothing_alpha': args.reward_smoothing_alpha,

        # Env
        'env': args.env,
        'env_num_stack': args.env_num_stack,
        'env_num_repeat': args.env_num_repeat,
        
        # Checkpointing
        'checkpoint_freq': args.checkpoint_freq,
        'save_dir': args.save_dir,
        'save_model': args.save_model,
        'save_buffer': args.save_buffer,
        'save_video': args.save_video,

        # Offpolicy
        'offpolicy_pretraining': args.offpolicy_pretraining,
        'recorded_runs_path': args.recorded_runs_path,
        'offpolicy_steps': args.offpolicy_steps,

        'gamma': args.gamma,
        'tau': args.tau,

        'ae_params': {
            'latent_dim': args.encoder_latent_dim,
            'enc_lr': args.encoder_lr,
            'dec_lr': args.decoder_lr,
            'num_layers': args.ae_num_layers,
            'num_filters': args.ae_num_filters,
            'mask_ratio': args.ae_mask_ratio,
            'update_freq': args.ae_update_freq,
            'log_freq': args.ae_log_freq,
            'random_shift_pad': args.random_shift_pad
        },

        'actor_params': {
            'lr': args.actor_lr,
            'hidden_dim': args.actor_hidden_dim,
            'log_std_min': args.log_std_min,
            'log_std_max': args.log_std_max,
            'update_freq': args.actor_update_freq
        },

        'critic_params': {
            'lr': args.critic_lr,
            'hidden_dim': args.critic_hidden_dim,
            'target_update_freq': args.critic_target_update_freq
        },

        'alpha_params': {
            'lr': args.alpha_lr,
            'val': args.init_alpha
        }
    }

    return config

def load_recorded_data(buffer: ReplayBuffer, data_path: str, max_files=None):
    files = glob.glob(os.path.join(data_path, "*.pkl"))
    
    if not files:
        print(f"[Pretraining] No recorded data found in {data_path}!")
        return 0
        
    print(f"[Pretraining] Found {len(files)} recorded files. Loading...")
    
    loaded_transitions = 0
    
    for i, fpath in enumerate(files):
        if max_files and i >= max_files: break
        
        with open(fpath, "rb") as f:
            transitions = pickle.load(f)
            
        for t in transitions:
            state, action, reward, next_state, term, trunc, info = t
            
            buffer.add(state, action, reward, next_state, float(term))
            loaded_transitions += 1
            if (term or trunc) and isinstance(buffer, SmoothedReplayBuffer):
                buffer.reset()

    print(f"[Pretraining] Loaded {loaded_transitions} transitions into buffer.")
    return loaded_transitions

def train(agent: PixelSAC, buffer: ReplayBuffer, logger: Logger, config: dict, step: int):
    states, actions, rewards, nxt_states, not_dones = buffer.sample(config['batch_size'])

    agent.critic_update(states, actions, rewards, nxt_states, not_dones, logger, step)
    if step % config['log_freq'] == 0:
        agent.log_critic(logger, step)

    if step % config['critic_params']['target_update_freq'] == 0:
        agent.soft_update(agent.critic, agent.target_critic)
        agent.soft_update(agent.encoder, agent.target_critic_encoder)

    if step % config['actor_params']['update_freq'] == 0:
        agent.actor_update(states, logger, step)
        if step % config['log_freq'] == 0:
            agent.log_actor(logger, step)
            agent.log_alpha_val(logger, step)

    if step % config['ae_params']['update_freq'] == 0:
        agent.update_ae(states, logger, step)
        if step % config['ae_params']['log_freq'] == 0:
            agent.log_ae(logger, step)


def evaluate(env, agent: PixelSAC, n_episodes: int, record_video: bool, logger: ILogger, step: int):
    total_reward = 0
    for i in range(n_episodes):
        frames = []
        print(f"\n[Step: {step} | Ep: {i + 1} / {n_episodes}] Running Evaluation...")
        
        obs, _ = env.reset()
        term, trunc, epi_reward = False, False, 0
        
        while not term and not trunc:
            with torch.no_grad():
                action = agent.act(obs, sample_action=False)
            obs, reward, term, trunc, _ = env.step(action)
            epi_reward += reward

            if i == 0 and record_video:
                frame = env.render()
                frame = cv2.resize(frame, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                frames.append(frame)

        if i == 0 and record_video and len(frames) > 0:
            video_np = np.array(frames).transpose(0, 3, 1, 2)
            logger.log_video("eval/video", video_np, step, 30)
        
        total_reward += epi_reward
        print(f"\n[Step: {step} | Ep: {i + 1} / {n_episodes}] Episode Reward: {epi_reward} | Truncated: {trunc} | Terminated / Done: {term}")

    avg_reward = total_reward / n_episodes
    logger.log_metrics({"eval/avg_reward": avg_reward}, step)
    print(f"\n[Step: {step} | Eval Avg. Reward: {avg_reward}")

def main():
    config = parse_args()
    
    if config['env'].lower() == "carracing": 
        env = CarRacingWrapper(gym.make("CarRacing-v3", domain_randomize=False, render_mode="rgb_array"), num_stack=config['env_num_stack'])
    elif config['env'].lower() == "visualpendulum":
        env = VisualPendulumWrapper(gym.make("Pendulum-v1", render_mode="rgb_array"), num_stack=config['env_num_stack'], num_repeat=config['env_num_repeat'])
    else:
        raise ValueError("Env not yet supported.")
    
    # Set seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    np.random.seed(config['seed'])
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"run_{timestamp}"
    
    if config['enable_logging']:
        logger = Logger(config, "sac-ae-cr-v3", name=run_name)
    else: 
        logger  = ILogger()
    
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    model_dir = os.path.join(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    buffer_dir = os.path.join(run_dir, "buffer")
    os.makedirs(buffer_dir, exist_ok=True)
    video_dir = os.path.join(run_dir, "video")
    os.makedirs(video_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    
    print(env.spec)

    sac_agent = PixelSAC(env.observation_space.shape, env.action_space.shape[0], config['ae_params'], config['actor_params'], \
                         config['critic_params'], config['alpha_params'], config['gamma'], config['tau'], device)
    
    if config['reward_smoothing']:
        buffer = SmoothedReplayBuffer(env.observation_space.shape, env.action_space.shape[0], config['buffer_capacity'], device, config['reward_smoothing_alpha'])    
    else:
        buffer = ReplayBuffer(env.observation_space.shape, env.action_space.shape[0], config['buffer_capacity'], device)    

    pt_steps = 1
    if config['offpolicy_pretraining']:
        count = load_recorded_data(buffer, config['recorded_runs_path'])
        
        if count > config['batch_size']:
            offpolicy_steps = min(config['offpolicy_steps'], count)
            print(f"Pretraining for {offpolicy_steps} steps.")
            start = time.time()
            
            for step in range(1, offpolicy_steps + 1):
                train(sac_agent, buffer, logger, config, step=step)
                pt_steps = step + 1
                if step % 1000 == 0:
                    print(f"Pretraining Step {step}/{offpolicy_steps}")
            
            print(f"Pretraining Complete. Time: {time.time() - start:.1f}s")
        else:
            print("Not enough data. Skipping pretraining.")
            config['offpolicy_pretraining'] = False
    
    state, _ = env.reset()
    episode_reward = 0
    episode_step = 0
    episode = 1
    last_eval = 0
    
    start = epi_start = time.time()
    for step in range(pt_steps, config['num_steps']):
        if step < config['exploration_steps']:
            action = env.action_space.sample()
        else:
            action = sac_agent.act(state, sample_action=True)

        nxt_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        buffer.add(state, action, reward, nxt_state, term)
        if done and isinstance(buffer, SmoothedReplayBuffer):
            buffer.reset()
        
        state = nxt_state
        episode_reward += reward
        episode_step += 1

        if done:
            logger.log_metrics({
                "train/episode_reward": episode_reward,
                "train/episode_len": episode_step,
                "train/episode": episode
            }, step)

            print(f"[Step: {step} | Episode: {episode}] Reward: {episode_reward} | Truncated: {trunc} | Terminated / Done: {term} | Time: {time.time() - epi_start}s")

            if (step - last_eval) >= config['eval_freq']:
                evaluate(env, sac_agent, config['eval_episodes'], config['record_video'], logger, step)
                last_eval = step
            
            state, _ = env.reset()
            episode_reward = 0
            episode_step = 0
            episode += 1
            epi_start = time.time()

        if step >= config['exploration_steps'] or config['offpolicy_pretraining']:
            train(sac_agent, buffer, logger, config, step)

        if step % config['checkpoint_freq'] == 0:
            sac_agent.save_checkpoint(model_dir, step)
            if config['save_buffer']:
                buffer.save(buffer_dir)

    print(f"Completed. Time taken: {time.time() - start}")

if __name__ == '__main__':
    main()
    