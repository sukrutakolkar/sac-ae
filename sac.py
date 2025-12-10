import torch, os, copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vision import CNNEncoder, CNNDecoder, RandomShiftsAug, dequantize
from logger import Logger
from itertools import chain

def _delta_orthogonal_init(layer):
    # taken from Yarats' SAE+AE  - 1910.01741, which in turn cites - https://arxiv.org/abs/1806.05393
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight.data)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)

    elif isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        assert layer.weight.size(2) == layer.weight.size(3)
        
        layer.weight.data.fill_(0.0)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)
            
        mid = layer.weight.size(2) // 2
        
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(layer.weight.data[:, :, mid, mid], gain)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=1024, log_std_min=-10, log_std_max=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.outputs = {}

    def forward(self, obs, sample_action=True):
        x = self.net(obs)    
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max) 
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std
        
        if sample_action:
            dist = torch.distributions.Normal(mu, std)
            
            action = dist.rsample() 
            pi = torch.tanh(action)
            
            log_prob = dist.log_prob(action)
            log_prob -= torch.log(1.0 - pi.pow(2).clamp(0, 1) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

            return pi, log_prob
        else:
            pi = torch.tanh(mu)
            return pi, None
        
    def log(self, logger: Logger, step: int):
        logger.log_histogram('actor/mu',  self.outputs['mu'],  step)
        logger.log_histogram('actor/std', self.outputs['std'], step)

        
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()

        self.fcn = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        q = self.fcn(data)
        return q
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()

        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)

        self.outputs = {}

    def forward(self, state, action):
        data = torch.cat([state, action], dim=1)
        
        q1 = self.q1(data)
        q2 = self.q2(data)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger: Logger, step: int):
        logger.log_histogram('critic/q1', self.outputs['q1'], step)
        logger.log_histogram('critic/q2', self.outputs['q2'], step)

class PixelSAC:
    def __init__(self, obs_shape, action_dim, ae_params, actor_params, critic_params, alpha_params, gamma, tau, device):        
        self.encoder = CNNEncoder(obs_shape, ae_params['latent_dim'], ae_params['num_layers'], ae_params['num_filters'], ae_params['mask_ratio']).to(device)
        self.decoder = CNNDecoder(obs_shape, ae_params['latent_dim'], self.encoder.spatial_out_dim, ae_params['num_layers'], ae_params['num_filters']).to(device)
        self.shifts_aug = RandomShiftsAug(ae_params['random_shift_pad'])

        self.actor = Actor(ae_params['latent_dim'], action_dim, actor_params['hidden_dim']).to(device)
        self.critic = Critic(ae_params['latent_dim'], action_dim, critic_params['hidden_dim']).to(device)
        self.target_critic = Critic(ae_params['latent_dim'], action_dim, critic_params['hidden_dim']).to(device)

        self.encoder.apply(_delta_orthogonal_init)
        self.decoder.apply(_delta_orthogonal_init)
        self.actor.apply(_delta_orthogonal_init)
        self.critic.apply(_delta_orthogonal_init)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic_encoder = copy.deepcopy(self.encoder)
        self.actor_encoder = copy.deepcopy(self.encoder)
        self.actor_encoder.convs = self.encoder.convs

        self.log_alpha = torch.tensor(np.log(alpha_params['val']), device=device, requires_grad=True)
        self.target_entropy = -action_dim  # Negative of action dimension
        self.gamma = torch.tensor(gamma, device=device)
        self.tau = torch.tensor(tau, device=device)

        self.enc_opt    = torch.optim.Adam(self.encoder.parameters(), lr=ae_params['enc_lr'])
        self.dec_opt    = torch.optim.Adam(self.decoder.parameters(), lr=ae_params['dec_lr'])
        self.critic_opt = torch.optim.Adam(chain(self.critic.parameters(), self.encoder.parameters()), lr=critic_params['lr'])
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_params['lr'])
        self.actor_opt  = torch.optim.Adam(chain(self.actor.parameters(), 
                                                 self.actor_encoder.fcn.parameters(), 
                                                 self.actor_encoder.ln.parameters()), lr=actor_params['lr'])

        self.device = device

    def actor_update(self, states, logger: Logger, step: int):
        with torch.no_grad():
            states = self.shifts_aug(states)
        
        actor_latent = self.actor_encoder(states, detach=True)
        critic_latent = self.encoder(states, detach=True)

        actions, log_prob = self.actor(actor_latent)
        q1, q2 = self.critic(critic_latent, actions)
        q = torch.min(q1, q2)

        alpha = self.log_alpha.exp()
        # Note: it is important to detach the alpha component in actor loss and the actor component in alpha loss
        # in actor loss specifically or the alpha value will drop to 0 very quickly.
        loss = (alpha.detach() * log_prob - q).mean() # (alpha.detach() * log_prob - q).mean()
        logger.log_metrics({'actor/loss': loss.item()}, step)

        self.actor_opt.zero_grad(True)
        loss.backward()
        self.actor_opt.step()
        
        loss_alpha = (alpha * (-log_prob - self.target_entropy).detach()).mean() # alpha * (-log_prob - self.target_entropy)
        logger.log_metrics({
            'alpha/loss': loss_alpha.item()
        }, step)

        self.log_alpha_opt.zero_grad(True)
        loss_alpha.backward()
        self.log_alpha_opt.step()

    def critic_update(self, states, actions, rewards, next_states, not_dones, logger: Logger, step: int):
        self.alpha = self.log_alpha.exp()
        with torch.no_grad():
            next_states = self.shifts_aug(next_states)
            
            nxt_latent_actor = self.actor_encoder(next_states)
            nxt_acts, next_log_probs = self.actor(nxt_latent_actor)

            nxt_latent_tc = self.target_critic_encoder(next_states)
            nxt_q1, nxt_q2 = self.target_critic(nxt_latent_tc, nxt_acts)

            nxt_q = torch.min(nxt_q1, nxt_q2) - self.alpha.detach() * next_log_probs
            target_q = rewards + (not_dones * self.gamma * nxt_q)

        states = self.shifts_aug(states)
        latent = self.encoder(states)
        q1, q2 = self.critic(latent, actions)
        loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        logger.log_metrics({"critic/loss": loss.item()}, step)

        self.critic_opt.zero_grad(True)
        #self.enc_opt.zero_grad(True)
        loss.backward()
        self.critic_opt.step()
        #self.enc_opt.step()

        #self.soft_update(self.critic, self.target_critic)

    def update_ae(self, obs, logger: Logger, step: int):
        latent = self.encoder(obs, apply_mask=True, return_mask=False)
        latent_penalty = (0.5 * latent.pow(2).sum(1)).mean() * 1e-6
        
        recon = self.decoder(latent)

        obs = dequantize(obs)

        loss = F.mse_loss(obs, recon) + latent_penalty
        logger.log_metrics({"ae/loss": loss.item()}, step)

        self.enc_opt.zero_grad(True)
        self.dec_opt.zero_grad(True)
        loss.backward()
        self.enc_opt.step()
        self.dec_opt.step()

    def log_actor(self, logger: Logger, step: int):
        self.actor.log(logger, step)

    def log_critic(self, logger: Logger, step: int):
        self.critic.log(logger, step)

    def log_alpha_val(self, logger: Logger, step: int):
        logger.log_metrics({'alpha/val': self.log_alpha.exp().item()}, step)

    def log_ae(self, logger: Logger, step: int):
        self.encoder.log(logger, step)
        self.decoder.log(logger, step)

    def soft_update(self, src:nn.Module, dst: nn.Module):
        for src_param, dst_param in zip(src.parameters(), dst.parameters()):
            dst_param.data.copy_((src_param.data * self.tau) + ((1.0 - self.tau) * dst_param.data), True)

    def act(self, obs, sample_action=False):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device).float()
            obs = obs.unsqueeze(0)
            latent = self.actor_encoder(obs)
            action, _, = self.actor(latent, sample_action=sample_action)
            return action.cpu().numpy().flatten()

    def save_checkpoint(self, path, step):
        if not os.path.exists(path):
            os.makedirs(path)
            
        save_path = os.path.join(path, f"checkpoint_{step}.pt")
        
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'log_alpha': self.log_alpha.detach(),
            'target_critic_encoder': self.target_critic_encoder.state_dict(),
            'actor_encoder': self.actor_encoder.state_dict(), 
            
            'enc_opt': self.enc_opt.state_dict(),
            'dec_opt': self.dec_opt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'log_alpha_opt': self.log_alpha_opt.state_dict(),
            
        }, save_path)

    def load_checkpoint(self, path, step):
        load_path = os.path.join(path, f"checkpoint_{step}.pt")
        
        assert os.path.exists(load_path)

        print(f"Loading checkpoint from {load_path}")
        
        ckpt = torch.load(load_path, map_location=self.device)

        self.encoder.load_state_dict(ckpt['encoder'])
        self.decoder.load_state_dict(ckpt['decoder'])
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_critic.load_state_dict(ckpt['target_critic'])
        self.target_critic_encoder.load_state_dict(ckpt['target_critic_encoder'])
        self.actor_encoder.load_state_dict(ckpt['actor_encoder'])
        self.actor_encoder.convs = self.encoder.convs

        with torch.no_grad():
            self.log_alpha.copy_(ckpt['log_alpha'])

        self.enc_opt.load_state_dict(ckpt['enc_opt'])
        self.dec_opt.load_state_dict(ckpt['dec_opt'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.log_alpha_opt.load_state_dict(ckpt['log_alpha_opt'])
        
        print("Checkpoint loaded successfully.")






        

        