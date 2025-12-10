import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import Logger

class CNNEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, mask_ratio=0.0):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers  = num_layers
        self.num_filters = num_filters
        self.mask_ratio  = mask_ratio

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2), 
             nn.LeakyReLU(inplace=True)]
        )

        for _ in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
            self.convs.append(nn.LeakyReLU(inplace=True))
            
        # out dim calculation
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            for layer in self.convs:
                dummy = layer(dummy)
            
            self.spatial_out_dim = dummy.shape[-1] 
            self.flat_dim = dummy.shape[1] * dummy.shape[2] * dummy.shape[3]
            
            print(f"Encoder Spatial Output Dim.: {self.spatial_out_dim}x{self.spatial_out_dim}, Flattened Dim: {self.flat_dim}")

        self.fcn  = nn.Linear(self.flat_dim, self.feature_dim)
        self.ln   = nn.LayerNorm(self.feature_dim)
        self.tanh = nn.Tanh()

        self.outputs = {}

    def forward_conv(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs / 255.
        
        for i, layer in enumerate(self.convs):
            x = layer(x)
            if i % 2 == 1: self.outputs[f'conv{(i//2) + 1}'] = x
            
        x = x.flatten(start_dim=1)
        return x
    
    def forward(self, obs, detach=False, apply_mask=False, return_mask=False):
        self.outputs['obs'] = obs
        mask = None

        if apply_mask and self.mask_ratio > 0:
            obs, mask = self.mask_input(obs)
        
        x = self.forward_conv(obs)

        if detach:
            x = x.detach()

        x = self.fcn(x)
        self.outputs['fcn'] = x

        x = self.ln(x)
        self.outputs['ln'] = x

        x = self.tanh(x) # [-1, 1]
        self.outputs['tanh']  = x

        if apply_mask and self.mask_ratio > 0 and return_mask:
            return x, mask
        elif return_mask:
            return x, None
        
        return x
    
    def mask_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mask_ratio <= 0:
            return x, None
            
        N, C, H, W = x.shape
        x_masked = x.clone()  

        mask = torch.rand(N, 1, H, W, device=x.device) < self.mask_ratio
        x_masked[mask.expand_as(x)] = 0
        
        return x_masked, mask
    
    def log(self, logger: Logger, step: int):
        for k, v in self.outputs.items():
            if k == 'obs':
                logger.log_video("encoder/obs", v[0:1], step=step, fps=1)
            elif "conv" in k:
                logger.log_image(f"encoder/{k}_map", v[0, 0], step=step)
                logger.log_histogram(f"encoder/{k}_activations", v, step=step)
            else:
                logger.log_histogram(f"encoder/{k}_activations", v, step=step)
    
class CNNDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, encoder_spatial_dim, num_layers, num_filters):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.spatial_dim = encoder_spatial_dim 
        
        # feature_dim -> filters * H * W
        self.fcn = nn.Sequential(
                nn.Linear(feature_dim, num_filters * self.spatial_dim * self.spatial_dim),
                nn.LeakyReLU(inplace=True)
            )

        self.deconvs = nn.ModuleList()

        for _ in range(self.num_layers - 1):
            self.deconvs.append(nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1))
            self.deconvs.append(nn.LeakyReLU(inplace=True))
            
        self.deconvs.append(nn.ConvTranspose2d(num_filters, obs_shape[0], 3, stride=2, output_padding=1))
        #self.deconvs.append(nn.LeakyReLU(inplace=True)) 

        self.outputs = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fcn(x)
        
        x = x.view(-1, self.num_filters, self.spatial_dim, self.spatial_dim)

        for i, layer in enumerate(self.deconvs):
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose2d): self.outputs[f'deconv{(i//2) + 1}'] = x
            
        self.outputs['obs'] = x
        return x
    
    def log(self, logger: Logger, step: int):
        for k, v in self.outputs.items():
            if k == 'obs':
                logger.log_video("decoder/obs", torch.clamp(v[0:1], 0, 1), step=step, fps=1)
            elif "deconv" in k:
                logger.log_image(f"decoder/{k}_map", v[0, 0], step=step)
                logger.log_histogram(f"decoder/{k}_activations", v, step=step)
            else:
                logger.log_histogram(f"decoder/{k}_activations", v, step=step)

class RandomShiftsAug(nn.Module):
    # Taken from DrQ-v2 - https://arxiv.org/abs/2107.09645
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

def dequantize(obs, bits=5):
    # Preprocessing block taken from Yarats' SAE+AE - 1910.01741, which in-turn cites - https://arxiv.org/abs/1807.03039.
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs
