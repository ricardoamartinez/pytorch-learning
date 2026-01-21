# =============================================================================
# LESSON 15: Complete World Model Implementation
# =============================================================================
# Putting it all together: A complete world model for learning and planning.
# Based on the Dreamer architecture (Hafner et al.)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
import numpy as np
from collections import deque

# -----------------------------------------------------------------------------
# THE CONCEPT: World Model Architecture
# -----------------------------------------------------------------------------
"""
WORLD MODEL = Encoder + Dynamics Model + Decoder + Reward/Value Heads

    ┌──────────────────────────────────────────────────────────────┐
    │                        WORLD MODEL                           │
    │                                                              │
    │   Observation ──► [Encoder] ──► Latent z                    │
    │        │              │            │                         │
    │        │              │            ▼                         │
    │        │              │      [RSSM Dynamics] ◄── Action     │
    │        │              │            │                         │
    │        │              │            ▼                         │
    │        │              │      Next Latent z'                 │
    │        │              │            │                         │
    │        │              │            ├──► [Reward Head] ──► r  │
    │        │              │            │                         │
    │        │              │            └──► [Decoder] ──► o'    │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

TRAINING LOOP:
1. Collect real experience (observations, actions, rewards)
2. Train world model on collected data
3. Imagine trajectories using world model
4. Train policy/value on imagined data
5. Repeat

This is the "Dreamer" approach!
"""

# -----------------------------------------------------------------------------
# STEP 1: Convolutional Encoder/Decoder for Images
# -----------------------------------------------------------------------------
print("=" * 60)
print("IMAGE ENCODER/DECODER")
print("=" * 60)

class ConvEncoder(nn.Module):
    """Encode images to feature vectors."""
    def __init__(self, input_channels=3, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),   # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8 -> 4
            nn.ReLU(),
            nn.Flatten(),
        )
        # 256 * 4 * 4 = 4096 for 64x64 input
        self.fc = nn.Linear(256 * 4 * 4, feature_dim)

    def forward(self, x):
        return self.fc(self.conv(x))


class ConvDecoder(nn.Module):
    """Decode feature vectors to images."""
    def __init__(self, feature_dim=256, output_channels=3):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1),  # 32 -> 64
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)
        return self.deconv(x)

# Test
encoder = ConvEncoder(input_channels=3, feature_dim=256)
decoder = ConvDecoder(feature_dim=256, output_channels=3)

img = torch.randn(4, 3, 64, 64)
features = encoder(img)
recon = decoder(features)
print(f"Input image:  {img.shape}")
print(f"Features:     {features.shape}")
print(f"Reconstructed: {recon.shape}")

# -----------------------------------------------------------------------------
# STEP 2: RSSM (Recurrent State-Space Model)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RSSM DYNAMICS MODEL")
print("=" * 60)

class RSSM(nn.Module):
    """
    Recurrent State-Space Model for latent dynamics.

    State = (h, z) where:
    - h: Deterministic recurrent state (memory)
    - z: Stochastic state (sampled)
    """
    def __init__(self, embed_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Recurrent model
        self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # Prior network: p(z | h)
        self.prior_fc = nn.Linear(hidden_dim, hidden_dim)
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_std = nn.Linear(hidden_dim, latent_dim)

        # Posterior network: q(z | h, embed)
        self.posterior_fc = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        self.posterior_mean = nn.Linear(hidden_dim, latent_dim)
        self.posterior_std = nn.Linear(hidden_dim, latent_dim)

    def init_state(self, batch_size, device):
        """Initialize hidden and latent states."""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return h, z

    def get_prior(self, h):
        """Compute prior distribution p(z | h)."""
        x = F.relu(self.prior_fc(h))
        mean = self.prior_mean(x)
        std = F.softplus(self.prior_std(x)) + 0.1
        return mean, std

    def get_posterior(self, h, embed):
        """Compute posterior distribution q(z | h, embed)."""
        x = torch.cat([h, embed], dim=-1)
        x = F.relu(self.posterior_fc(x))
        mean = self.posterior_mean(x)
        std = F.softplus(self.posterior_std(x)) + 0.1
        return mean, std

    def sample(self, mean, std):
        """Sample using reparameterization trick."""
        return mean + std * torch.randn_like(std)

    def observe(self, prev_h, prev_z, action, embed):
        """Step with observation (training)."""
        # Update recurrent state
        x = torch.cat([prev_z, action], dim=-1)
        h = self.rnn(x, prev_h)

        # Get distributions
        prior_mean, prior_std = self.get_prior(h)
        post_mean, post_std = self.get_posterior(h, embed)

        # Sample from posterior
        z = self.sample(post_mean, post_std)

        return h, z, prior_mean, prior_std, post_mean, post_std

    def imagine(self, prev_h, prev_z, action):
        """Step without observation (imagination)."""
        # Update recurrent state
        x = torch.cat([prev_z, action], dim=-1)
        h = self.rnn(x, prev_h)

        # Sample from prior
        prior_mean, prior_std = self.get_prior(h)
        z = self.sample(prior_mean, prior_std)

        return h, z, prior_mean, prior_std


# -----------------------------------------------------------------------------
# STEP 3: Complete World Model
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPLETE WORLD MODEL")
print("=" * 60)

class WorldModel(nn.Module):
    """
    Complete World Model combining:
    - Image encoder/decoder
    - RSSM dynamics
    - Reward predictor
    - Continue predictor
    """
    def __init__(self, obs_channels=3, action_dim=4,
                 embed_dim=256, hidden_dim=256, latent_dim=32):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.feature_dim = hidden_dim + latent_dim  # h + z

        # Encoder/Decoder
        self.encoder = ConvEncoder(obs_channels, embed_dim)
        self.decoder = ConvDecoder(self.feature_dim, obs_channels)

        # Dynamics
        self.rssm = RSSM(embed_dim, action_dim, hidden_dim, latent_dim)

        # Prediction heads
        self.reward_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.continue_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def get_features(self, h, z):
        """Concatenate h and z into feature vector."""
        return torch.cat([h, z], dim=-1)

    def encode(self, obs):
        """Encode observation to embedding."""
        return self.encoder(obs)

    def decode(self, features):
        """Decode features to observation."""
        return self.decoder(features)

    def predict_reward(self, features):
        """Predict reward from features."""
        return self.reward_head(features).squeeze(-1)

    def predict_continue(self, features):
        """Predict continue probability."""
        return torch.sigmoid(self.continue_head(features)).squeeze(-1)

    def observe_sequence(self, observations, actions):
        """
        Process a sequence of observations.

        Args:
            observations: (B, T, C, H, W)
            actions: (B, T, action_dim)

        Returns:
            Dictionary of outputs
        """
        B, T = observations.shape[:2]
        device = observations.device

        # Initialize states
        h, z = self.rssm.init_state(B, device)

        # Storage
        outputs = {
            'h': [], 'z': [], 'features': [],
            'prior_mean': [], 'prior_std': [],
            'post_mean': [], 'post_std': [],
            'recon': [], 'reward_pred': [], 'continue_pred': []
        }

        for t in range(T):
            # Encode observation
            embed = self.encode(observations[:, t])

            # RSSM step
            action = actions[:, t]
            h, z, prior_mean, prior_std, post_mean, post_std = \
                self.rssm.observe(h, z, action, embed)

            # Get features and predictions
            features = self.get_features(h, z)
            recon = self.decode(features)
            reward_pred = self.predict_reward(features)
            continue_pred = self.predict_continue(features)

            # Store
            outputs['h'].append(h)
            outputs['z'].append(z)
            outputs['features'].append(features)
            outputs['prior_mean'].append(prior_mean)
            outputs['prior_std'].append(prior_std)
            outputs['post_mean'].append(post_mean)
            outputs['post_std'].append(post_std)
            outputs['recon'].append(recon)
            outputs['reward_pred'].append(reward_pred)
            outputs['continue_pred'].append(continue_pred)

        # Stack outputs
        for key in outputs:
            outputs[key] = torch.stack(outputs[key], dim=1)

        return outputs

    def imagine_trajectory(self, initial_h, initial_z, policy, horizon):
        """
        Imagine future trajectory using prior (no observations).

        Args:
            initial_h: Starting hidden state (B, hidden_dim)
            initial_z: Starting latent state (B, latent_dim)
            policy: Policy network that maps features -> actions
            horizon: Number of steps to imagine

        Returns:
            Imagined trajectory
        """
        h, z = initial_h, initial_z
        trajectory = {
            'h': [], 'z': [], 'features': [],
            'action': [], 'reward_pred': [], 'continue_pred': []
        }

        for _ in range(horizon):
            features = self.get_features(h, z)

            # Get action from policy
            action = policy(features)

            # Predict rewards/continue
            reward_pred = self.predict_reward(features)
            continue_pred = self.predict_continue(features)

            # Store
            trajectory['h'].append(h)
            trajectory['z'].append(z)
            trajectory['features'].append(features)
            trajectory['action'].append(action)
            trajectory['reward_pred'].append(reward_pred)
            trajectory['continue_pred'].append(continue_pred)

            # Imagine next state (prior only, no observation)
            h, z, _, _ = self.rssm.imagine(h, z, action)

        # Stack
        for key in trajectory:
            trajectory[key] = torch.stack(trajectory[key], dim=1)

        return trajectory


# Create world model
world_model = WorldModel(
    obs_channels=3,
    action_dim=4,
    embed_dim=256,
    hidden_dim=256,
    latent_dim=32
)

print("World Model created!")
print(f"  Feature dim: {world_model.feature_dim}")
print(f"  Parameters:  {sum(p.numel() for p in world_model.parameters()):,}")

# Test observe sequence
obs = torch.randn(4, 10, 3, 64, 64)  # B=4, T=10, 3x64x64 images
actions = torch.randn(4, 10, 4)

outputs = world_model.observe_sequence(obs, actions)
print(f"\nObserve sequence test:")
print(f"  Input obs:     {obs.shape}")
print(f"  Input actions: {actions.shape}")
print(f"  Output h:      {outputs['h'].shape}")
print(f"  Output z:      {outputs['z'].shape}")
print(f"  Recon:         {outputs['recon'].shape}")
print(f"  Reward pred:   {outputs['reward_pred'].shape}")

# -----------------------------------------------------------------------------
# STEP 4: World Model Training Loss
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("WORLD MODEL LOSS")
print("=" * 60)

def world_model_loss(outputs, observations, rewards, continues,
                     kl_weight=1.0, reward_weight=1.0, continue_weight=1.0):
    """
    Compute world model training loss.

    Components:
    1. Reconstruction loss (image prediction)
    2. KL divergence (prior-posterior matching)
    3. Reward prediction loss
    4. Continue prediction loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(outputs['recon'], observations)

    # KL divergence
    prior_dist = Normal(outputs['prior_mean'], outputs['prior_std'])
    post_dist = Normal(outputs['post_mean'], outputs['post_std'])

    kl_loss = torch.distributions.kl_divergence(post_dist, prior_dist)
    kl_loss = kl_loss.sum(dim=-1).mean()  # Sum over latent dim, mean over batch/time

    # Reward loss
    reward_loss = F.mse_loss(outputs['reward_pred'], rewards)

    # Continue loss (binary cross-entropy)
    continue_loss = F.binary_cross_entropy(
        outputs['continue_pred'],
        continues.float()
    )

    # Total loss
    total_loss = (
        recon_loss
        + kl_weight * kl_loss
        + reward_weight * reward_loss
        + continue_weight * continue_loss
    )

    return {
        'total': total_loss,
        'recon': recon_loss,
        'kl': kl_loss,
        'reward': reward_loss,
        'continue': continue_loss
    }

# Test loss computation
rewards = torch.randn(4, 10)
continues = torch.ones(4, 10)

losses = world_model_loss(outputs, obs, rewards, continues)
print(f"Total loss:    {losses['total'].item():.4f}")
print(f"Recon loss:    {losses['recon'].item():.4f}")
print(f"KL loss:       {losses['kl'].item():.4f}")
print(f"Reward loss:   {losses['reward'].item():.4f}")
print(f"Continue loss: {losses['continue'].item():.4f}")

# -----------------------------------------------------------------------------
# STEP 5: Actor-Critic for Imagination
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ACTOR-CRITIC IN IMAGINATION")
print("=" * 60)

class Actor(nn.Module):
    """Policy network for action selection."""
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, features):
        x = self.net(features)
        mean = self.mean_head(x)
        std = F.softplus(self.std_head(x)) + 0.1
        return mean, std

    def sample(self, features):
        mean, std = self(features)
        dist = Normal(mean, std)
        action = dist.rsample()  # Reparameterized sample
        return torch.tanh(action), dist  # Bound actions to [-1, 1]


class Critic(nn.Module):
    """Value network for state evaluation."""
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


actor = Actor(world_model.feature_dim, action_dim=4)
critic = Critic(world_model.feature_dim)

print(f"Actor parameters:  {sum(p.numel() for p in actor.parameters()):,}")
print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")

# -----------------------------------------------------------------------------
# STEP 6: Dreamer-Style Imagination Training
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DREAMER IMAGINATION TRAINING")
print("=" * 60)

def imagine_and_train(world_model, actor, critic, start_h, start_z,
                      horizon=15, gamma=0.99, lambda_=0.95):
    """
    Imagine trajectories and compute actor-critic losses.

    This is the key insight of Dreamer:
    - Generate imagined experience using world model
    - Train policy on imagined experience
    - No real environment interaction needed!
    """
    # Imagine trajectory
    h, z = start_h, start_z
    features_list = []
    actions_list = []
    rewards_list = []
    continues_list = []

    for _ in range(horizon):
        features = world_model.get_features(h, z)

        # Sample action from policy
        action, dist = actor.sample(features)

        # Predict reward and continue
        with torch.no_grad():
            reward = world_model.predict_reward(features)
            cont = world_model.predict_continue(features)

        features_list.append(features)
        actions_list.append(action)
        rewards_list.append(reward)
        continues_list.append(cont)

        # Imagine next state
        h, z, _, _ = world_model.rssm.imagine(h, z, action)

    # Stack imagined trajectory
    features = torch.stack(features_list, dim=1)  # (B, T, feature_dim)
    rewards = torch.stack(rewards_list, dim=1)    # (B, T)
    continues = torch.stack(continues_list, dim=1)  # (B, T)

    # Compute values
    values = critic(features)  # (B, T)

    # Compute lambda-returns (GAE-style)
    returns = compute_lambda_returns(rewards, values, continues, gamma, lambda_)

    # Actor loss: maximize returns
    # (Simplified - full implementation uses policy gradient)
    actor_loss = -returns.mean()

    # Critic loss: predict returns
    critic_loss = F.mse_loss(values[:, :-1], returns[:, :-1].detach())

    return actor_loss, critic_loss, returns.mean()


def compute_lambda_returns(rewards, values, continues, gamma, lambda_):
    """Compute lambda-returns for advantage estimation."""
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)

    # Bootstrap from last value
    returns[:, -1] = values[:, -1]

    for t in reversed(range(T - 1)):
        returns[:, t] = (
            rewards[:, t]
            + gamma * continues[:, t] * (
                (1 - lambda_) * values[:, t + 1]
                + lambda_ * returns[:, t + 1]
            )
        )

    return returns

# Test imagination training
B = 4
h, z = world_model.rssm.init_state(B, 'cpu')
actor_loss, critic_loss, mean_return = imagine_and_train(
    world_model, actor, critic, h, z, horizon=15
)
print(f"Actor loss:  {actor_loss.item():.4f}")
print(f"Critic loss: {critic_loss.item():.4f}")
print(f"Mean return: {mean_return.item():.4f}")

# -----------------------------------------------------------------------------
# STEP 7: Complete Training Loop (Pseudocode)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPLETE DREAMER TRAINING LOOP")
print("=" * 60)

"""
DREAMER TRAINING LOOP:

def train_dreamer(env, world_model, actor, critic, num_steps):

    replay_buffer = ReplayBuffer()

    for step in range(num_steps):

        # ===== 1. ENVIRONMENT INTERACTION =====
        # Collect experience using current policy
        obs = env.reset()
        done = False
        episode_data = []

        while not done:
            # Encode observation and get state
            with torch.no_grad():
                embed = world_model.encode(obs)
                features = world_model.get_features(h, z)
                action, _ = actor.sample(features)

            # Step environment
            next_obs, reward, done = env.step(action)
            episode_data.append((obs, action, reward, next_obs, done))

            # Update RSSM state
            h, z, ... = world_model.rssm.observe(h, z, action, embed)
            obs = next_obs

        replay_buffer.add(episode_data)


        # ===== 2. WORLD MODEL TRAINING =====
        # Sample batch from replay buffer
        batch = replay_buffer.sample(batch_size=50, seq_len=50)

        # Train world model
        outputs = world_model.observe_sequence(batch.obs, batch.actions)
        wm_loss = world_model_loss(outputs, batch.obs, batch.rewards, batch.continues)

        wm_optimizer.zero_grad()
        wm_loss['total'].backward()
        wm_optimizer.step()


        # ===== 3. BEHAVIOR LEARNING (IMAGINATION) =====
        # Get starting states from replay
        with torch.no_grad():
            outputs = world_model.observe_sequence(batch.obs, batch.actions)
            start_h = outputs['h'][:, -1]  # Last hidden state
            start_z = outputs['z'][:, -1]  # Last latent state

        # Train actor-critic on imagined trajectories
        actor_loss, critic_loss, _ = imagine_and_train(
            world_model, actor, critic, start_h, start_z, horizon=15
        )

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()


        # ===== 4. LOGGING =====
        if step % 100 == 0:
            print(f"Step {step}: WM Loss={wm_loss['total']:.4f}, "
                  f"Actor Loss={actor_loss:.4f}")

    return world_model, actor, critic
"""

print("See the pseudocode above for the complete Dreamer training loop!")
print("\nKey insight: Most training happens in IMAGINATION!")
print("Real environment interactions are only for collecting diverse data.")

# -----------------------------------------------------------------------------
# SUMMARY: World Model Components
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("WORLD MODEL SUMMARY")
print("=" * 60)

print("""
WORLD MODEL ARCHITECTURE:
========================

1. ENCODER (Lesson 9-10)
   - Input: Observation (image, state, etc.)
   - Output: Embedding / Latent code
   - Types: CNN, VAE, VQ-VAE

2. DYNAMICS MODEL (Lesson 14)
   - Input: Current state (h, z) + Action
   - Output: Next state (h', z')
   - Types: RNN, LSTM, GRU, Transformer
   - Key: RSSM with prior/posterior

3. DECODER (Lesson 9-10)
   - Input: Latent state
   - Output: Predicted observation
   - Used for: Visualization, reconstruction loss

4. REWARD PREDICTOR (Lesson 13-14)
   - Input: Latent state
   - Output: Predicted reward
   - Used for: Planning, imagination training

5. ACTOR-CRITIC (Lesson 13)
   - Actor: π(a|s) - policy
   - Critic: V(s) - value function
   - Trained on imagined trajectories!

TRAINING:
=========
- World Model: Reconstruction + KL + Reward losses
- Actor: Maximize imagined returns
- Critic: Predict imagined returns

FAMOUS WORLD MODELS:
===================
- Dreamer (v1, v2, v3): RSSM + Actor-Critic
- MuZero: Learned model + MCTS planning
- IRIS: Transformer dynamics + discrete tokens
- Genie: Generative world model from video

CONGRATULATIONS! You now understand the core concepts
of world models and can implement them in PyTorch!
""")

# -----------------------------------------------------------------------------
# EXERCISE:
# 1. Train on a simple environment (CartPole, MountainCar)
# 2. Visualize imagination rollouts as video
# 3. Compare with model-free RL (same number of environment steps)
# 4. Try transformer-based dynamics instead of RSSM
# 5. Implement VQ-VAE encoder for discrete latents (like IRIS)
# 6. Add curiosity-based exploration bonus
# -----------------------------------------------------------------------------
