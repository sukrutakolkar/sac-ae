import wandb
import numpy as np
import os
import torch

class ILogger:
    def __init__(self):
        pass

    def log_metrics(self):
        pass

    def log_image(self, key, image, step):
        pass

    def log_histogram(self, key, data, step):
        pass

    def log_video(self, key, frames, step, fps):
        pass

class Logger(ILogger):
    def __init__(self, config, project_name="carracing-v3", name=None):
        super().__init__()

        self.run = wandb.init(
            project=project_name,
            name=name,
            config=config,
        )

        self.save_dir = os.path.join("checkpoints", self.run.name)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Logging to W&B Project: {project_name}, Run: {self.run.name}")
        print(f"Model checkpoints path: {self.save_dir}")

    def log_metrics(self, metrics: dict, step: int, verbose=False):
        wandb.log(metrics, step=step)
        
        if verbose:
            log_str = f"Step: {step} | "
            for k, v in metrics.items():
                if isinstance(v, (float, np.float32)):
                    log_str += f"{k}: {v:.4f} | "
            print(log_str)

    def log_image(self, key, image, step):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        wandb.log({key: wandb.Image(image)}, step=step)

    def log_histogram(self, key, data, step):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
            
        wandb.log({key: wandb.Histogram(data)}, step=step)

    def log_video(self, key, frames, step, fps=30):
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()
        
        if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], torch.Tensor):
            frames = np.array([f.detach().cpu().numpy() for f in frames])
        frames = np.array(frames)
        
        if frames.ndim == 4 and frames.shape[-1] in [1, 3]:
            frames = frames.transpose(0, 3, 1, 2)

        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
            
        wandb.log({key: wandb.Video(frames, fps=fps, format="mp4")}, step=step)
        
    def finish(self):
        self.run.finish()