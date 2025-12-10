# Visual Soft Actor-Critic (SAC-AE)

My implementation of a **Visual Soft Actor-Critic**, originally created as a course project for **CSCI-B 659 Reinforcement Learning (Spring 2024)**.

My goal with this project was to understand Deep-RL/Actor-Critic models more closely, specifically focusing on understanding **vision encoding** in Deep-RL models. While it started as a coursework, I have continued to work on it, incorporating advances from **SAC+AE** (Yarats et al.) and **DrQ** (Kostrikov et al.).

## Environments & Performance
### CarRacing-v3
The primary goal was to learn vision encoding in Actor-Critic algorithms on a medium grade but also fun environment and CarRacing fit the bill. You can read more about the environment here - <link>.
The environment is considered solved when the model consistently reaches a score of **~900** per episode. The current implementation converges in **~800,000** environment steps. With Off-policy pretraining this can be reduced to **< 100k steps** (~4-5 hours), depending on the amount and quality of pretraining data. This can be further reduced, with an agressive learning rate but the policy will be unstable.

### VisualPendulum
A far simpler environment used for sanity checking, solved in ~10k steps. (30–40 mins).

## Code Structure
* **`sac.py`**: Core logic for PixelSAC, Actor, and Critic.
* **`vision.py`**: Definitions for the CNN-Encoder and CNN-Decoder.
* **`train.py`**: Main training loop and evaluation.
* **`buffer.py`**: Replay buffer implementations.
* **`gym_wrappers.py`**: Custom wrappers for `CarRacing-v3` and `Pendulum-v1`.
* **`game_recorder.py`**: Utility for capturing human-played episodes.
* **`logger.py`**: Wrapper for Weights & Biases logging.

## Installation
1.  Clone the repository and cd into it.
2.  Install dependencies.
   ```bash
pip install -r requirements.txt
```
Note that gymnasium[box2d] requires swig to be installed first.

## Usage
### Training - from scratch
**CarRacing:**
```bash
python train.py --env CarRacing --seed 42
```

**Visual Pendulum (Pendulum-v1):**
```bash
python train.py --env VisualPendulum 
```

### Off-Policy Pretraining
1.  **Recording Human Demonstrations** You can record human gameplay, this data is saved to a buffer and can be used to pretrain the agent. Controls: Arrow Keys - Steer/Gas, Space - Brake.
    ```bash
    python game_recorder.py
    ```
2.  **Train with Off-Policy Pretraining:**
    ```bash
    python train.py --env CarRacing --offpolicy_pretraining --offpolicy_steps 5000
    ```
## Learnings
This project served as a deep dive into the instability of training Convolutional Neural Networks (CNNs) simultaneously with Actor-Critic losses. Below is a summary of my learnings and architectural evolution of this implementation.

### Failed Architectures 
I experimented with the following architectures, **none of which worked**:
1.  **Shared CNN (Joint Training):** A shared CNN between Actor and Critic, jointly trained on their respective losses.
2.  **Frozen Pretrained CNN-AE:** A CNN AutoEncoder pretrained purely on reconstruction loss, frozen, and then used as a shared feature extractor.
3.  **Shared CNN-AE (Joint + Reconstruction):** A shared encoder jointly trained on Actor/Critic losses while also being trained on reconstruction loss via a decoder.

### Issues
#### Actor vs. Critic Losses
The nature, magnitude, and variance of Actor vs. Critic losses are very different. The **Critic** tries to extract features for value estimation, while the **Actor** tries to maximize probability log-likelihoods. Joint training of the encoder therefore corrupts the CNN feature extractor, further collapsing both Actor and Critic training.

#### Soft Updates and Inefficient Feature Extraction
In using a shared CNN Backbone, there is also the issue of the target critic soft update over the critic's weights. This process is hampered by a constantly updating CNN encoder. This necessitates each component handling its own feature encoder.

However, separate encoders did not work for me either. While the critic has a relatively stable loss and tries to find features that allow it to maximally infer the value of each state, the actor loss has a high variance and doesn't lead to a stable representation.

This makes a case for a shared encoder updated *only* on the critic's Bellman loss (where state representation is handled by the critic and informs the actor's policy). But again, this didn't work for me; the feature extraction proved very inefficient.

### Solution 
Since using an AutoEncoder gave a good, relatively stable pixel state representation, I decided to keep the **reconstruction loss** and share the encoder among the Actor and Critic, using the critic's Bellman loss as an auxillary loss for the encoder and keeping the encoder detached during the actor's update. Further, following Yarats' work. I let the Actor update its own projection of the CNN features, to improve it's understanding of the learned features to select the best policy. This approach solves Visual Pendulum in 20k steps and CarRacing in ~10^6

Following Yarats' work and progress in further Soft Q-learning models like **DrQ** and **DrQ-v2**, I added the following enhancements, which brought Pendulum convergence to ~10k steps and CarRacing to ~8e5:
* Dequantization in the AE update (Kingma et. al.).
* L2 Penalty on the encoder's latent vector.
* Random Shift Augmentation.
* Delta-Orthogonal Initialization (Xiao et. al.).

Note: I've also added reward smoothing (Lee et. al.), but it doesn't offer much improvement in these envs. In Pendulum, it severely slows down training. In CarRacing, while it slows down training, the convergence seems to be relatively stable compared to training without it.*

## References
* **SAC+AE:** Yarats, D., et al. "Improving Sample Efficiency in Model-Free Reinforcement Learning from Images." [arXiv:1910.01741]
* **DrQ:** Kostrikov, I., et al. "Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels." [arXiv:2004.13649]
* **DrQ-v2:** Yarats, D., et al. "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning" [arXiv:2107.09645]
* **Soft Actor-Critic:** Haarnoja, T., et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." [arXiv:1801.01290]
* **Soft Actor-Critic v2:** Haarnoja, T., et al. "Soft Actor-Critic Algorithms and Applications." [arXiv:1801.01290]
* **Dequantization Normalization:** Kingma, D., Dhariwal, P., "Glow: Generative Flow with Invertible 1 1 Convolutions" [arXiv:1807.03039]
* **Delta-Orthogonal Initialization:** Xiao, L., et. al., "Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks" [arXiv:1806.05393]



