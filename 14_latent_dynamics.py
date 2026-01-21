# =============================================================================
# LESSON 14: Latent Dynamics Models
# =============================================================================
# The core of world models: predicting how latent states evolve over time.
# This combines VAE (for encoding) with sequence models (for dynamics).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# -----------------------------------------------------------------------------
# THE CONCEPT: Latent Dynamics
# -----------------------------------------------------------------------------
"""
WORLD MODEL ARCHITECTURE:

    Observation o_t ─────┐
                         │
                    [Encoder/VAE]
                         │
                         ▼
                    Latent z_t ───┐
                                  │
    Action a_t ──────────────────►│
                                  │
                            [Dynamics Model]
                                  │
                                  ▼
                            Latent z_{t+1}
                                  │
                            [Decoder] (optional)
                                  │
                                  ▼
                        Predicted o_{t+1}

KEY INSIGHT:
- Don't predict pixels directly (high-dimensional, noisy)
- Predict in LATENT SPACE (compact, meaningful)
- Decode to pixels only when needed

TYPES OF DYNAMICS MODELS:
1. Deterministic: z' = f(z, a)
2. Stochastic: z' ~ p(z' | z, a) - captures uncertainty
3. Recurrent: Uses hidden state h for memory
4. Transformer-based: Attention over history
"""

# -----------------------------------------------------------------------------
# STEP 1: Simple Deterministic Latent Dynamics
# -----------------------------------------------------------------------------
print("=" * 60)
print("DETERMINISTIC LATENT DYNAMICS")
print("=" * 60)

class DeterministicDynamics(nn.Module):
    """
    Simplest dynamics model: z' = f(z, a)
    No stochasticity, no memory.
    """
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z, a):
        """
        Args:
            z: Current latent state (batch, latent_dim)
            a: Action (batch, action_dim)
        Returns:
            Next latent state (batch, latent_dim)
        """
        x = torch.cat([z, a], dim=-1)
        z_next = self.network(x)
        return z_next

det_dynamics = DeterministicDynamics(latent_dim=32, action_dim=4)
print(det_dynamics)

# Test
z = torch.randn(8, 32)
a = torch.randn(8, 4)
z_next = det_dynamics(z, a)
print(f"\nCurrent z: {z.shape}")
print(f"Action:    {a.shape}")
print(f"Next z:    {z_next.shape}")

# -----------------------------------------------------------------------------
# STEP 2: Stochastic Latent Dynamics
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STOCHASTIC LATENT DYNAMICS")
print("=" * 60)

"""
Real environments are often stochastic!
- Same state + action can lead to different outcomes
- Model this with a distribution: z' ~ N(μ(z,a), σ(z,a))

Benefits:
- Captures environment uncertainty
- Can sample diverse futures
- Better for planning under uncertainty
"""

class StochasticDynamics(nn.Module):
    """
    Stochastic dynamics: z' ~ N(μ, σ) where μ, σ = f(z, a)
    """
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Separate heads for mean and std
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logstd_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z, a, deterministic=False):
        """
        Args:
            z: Current latent (batch, latent_dim)
            a: Action (batch, action_dim)
            deterministic: If True, return mean (no sampling)
        Returns:
            z_next: Sampled next latent
            mean: Distribution mean
            std: Distribution std
        """
        x = torch.cat([z, a], dim=-1)
        h = self.network(x)

        mean = self.mean_head(h)
        logstd = self.logstd_head(h)
        std = F.softplus(logstd) + 1e-4  # Ensure positive

        if deterministic:
            return mean, mean, std

        # Reparameterization trick
        eps = torch.randn_like(std)
        z_next = mean + std * eps

        return z_next, mean, std

stoch_dynamics = StochasticDynamics(latent_dim=32, action_dim=4)
print(stoch_dynamics)

# Test
z_next, mean, std = stoch_dynamics(z, a)
print(f"\nSampled z_next: {z_next.shape}")
print(f"Mean:           {mean.shape}")
print(f"Std:            {std.shape}")
print(f"Std range:      [{std.min().item():.4f}, {std.max().item():.4f}]")

# -----------------------------------------------------------------------------
# STEP 3: Recurrent State-Space Model (RSSM)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RECURRENT STATE-SPACE MODEL (RSSM)")
print("=" * 60)

"""
RSSM (from Dreamer):
- Combines deterministic recurrence with stochastic state
- h_t = deterministic recurrent state (memory)
- z_t = stochastic state (sampled)

Two pathways:
1. Prior (imagination): p(z_t | h_t)
   - Used when we don't have observations
   - For imagining/planning

2. Posterior (observation): q(z_t | h_t, o_t)
   - Used when we have observations
   - For training, better estimate

RSSM equations:
    h_t = f(h_{t-1}, z_{t-1}, a_{t-1})           # Deterministic transition
    prior:     z_t ~ p(z_t | h_t)               # Predict from recurrent state
    posterior: z_t ~ q(z_t | h_t, o_t)          # Refine with observation
"""

class RSSM(nn.Module):
    """
    Recurrent State-Space Model from Dreamer.
    Combines deterministic recurrence with stochastic state.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=200,
                 latent_dim=30, embed_dim=200):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ReLU(),
        )

        # Recurrent model: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # Prior: p(z_t | h_t) - for imagination
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )
        self.prior_mean = nn.Linear(embed_dim, latent_dim)
        self.prior_logstd = nn.Linear(embed_dim, latent_dim)

        # Posterior: q(z_t | h_t, o_t) - with observation
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.posterior_mean = nn.Linear(embed_dim, latent_dim)
        self.posterior_logstd = nn.Linear(embed_dim, latent_dim)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim)

    def prior(self, h):
        """Compute prior p(z | h)."""
        x = self.prior_net(h)
        mean = self.prior_mean(x)
        logstd = self.prior_logstd(x)
        std = F.softplus(logstd) + 0.1
        return mean, std

    def posterior(self, h, obs_embed):
        """Compute posterior q(z | h, o)."""
        x = torch.cat([h, obs_embed], dim=-1)
        x = self.posterior_net(x)
        mean = self.posterior_mean(x)
        logstd = self.posterior_logstd(x)
        std = F.softplus(logstd) + 0.1
        return mean, std

    def sample(self, mean, std):
        """Sample z using reparameterization trick."""
        eps = torch.randn_like(std)
        return mean + std * eps

    def observe_step(self, prev_h, prev_z, prev_action, obs):
        """
        Single step with observation (for training).
        Returns both prior and posterior for KL loss.
        """
        # Recurrent update
        rnn_input = torch.cat([prev_z, prev_action], dim=-1)
        h = self.rnn(rnn_input, prev_h)

        # Prior (what we'd predict without observation)
        prior_mean, prior_std = self.prior(h)

        # Encode observation
        obs_embed = self.obs_encoder(obs)

        # Posterior (refined with observation)
        post_mean, post_std = self.posterior(h, obs_embed)
        z = self.sample(post_mean, post_std)

        return h, z, prior_mean, prior_std, post_mean, post_std

    def imagine_step(self, prev_h, prev_z, prev_action):
        """
        Single step without observation (for imagination/planning).
        Uses prior only.
        """
        # Recurrent update
        rnn_input = torch.cat([prev_z, prev_action], dim=-1)
        h = self.rnn(rnn_input, prev_h)

        # Sample from prior
        prior_mean, prior_std = self.prior(h)
        z = self.sample(prior_mean, prior_std)

        return h, z, prior_mean, prior_std

rssm = RSSM(obs_dim=64, action_dim=4, hidden_dim=200, latent_dim=30)
print(rssm)

# Test observe step
batch_size = 8
h = rssm.init_hidden(batch_size)
z = torch.randn(batch_size, 30)
action = torch.randn(batch_size, 4)
obs = torch.randn(batch_size, 64)

h_new, z_new, prior_mean, prior_std, post_mean, post_std = rssm.observe_step(h, z, action, obs)

print(f"\nObserve step:")
print(f"  Hidden:    {h_new.shape}")
print(f"  Latent z:  {z_new.shape}")
print(f"  Prior:     mean={prior_mean.shape}, std={prior_std.shape}")
print(f"  Posterior: mean={post_mean.shape}, std={post_std.shape}")

# Test imagine step
h_imag, z_imag, prior_mean, prior_std = rssm.imagine_step(h, z, action)
print(f"\nImagine step (no observation):")
print(f"  Hidden: {h_imag.shape}")
print(f"  Latent: {z_imag.shape}")

# -----------------------------------------------------------------------------
# STEP 4: Training RSSM - KL Divergence
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING RSSM")
print("=" * 60)

"""
RSSM TRAINING LOSS:

1. Reconstruction loss: How well can we reconstruct observations?
   L_recon = ||decode(z) - obs||^2

2. KL divergence: Prior should match posterior
   L_kl = KL(posterior || prior)

   This is crucial! It forces the prior (imagination) to be
   accurate even without observations.

3. (Optional) Reward prediction loss
   L_reward = ||predict_reward(h, z) - actual_reward||^2

Total: L = L_recon + β * L_kl + L_reward
"""

def kl_divergence_gaussian(mean1, std1, mean2, std2):
    """
    KL divergence between two Gaussians: KL(N1 || N2)
    """
    var1 = std1 ** 2
    var2 = std2 ** 2
    kl = 0.5 * (
        torch.log(var2 / var1)
        + var1 / var2
        + (mean1 - mean2) ** 2 / var2
        - 1
    )
    return kl.sum(dim=-1).mean()

# Example KL computation
kl = kl_divergence_gaussian(post_mean, post_std, prior_mean, prior_std)
print(f"KL divergence (posterior || prior): {kl.item():.4f}")

# -----------------------------------------------------------------------------
# STEP 5: Imagination Rollouts
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("IMAGINATION ROLLOUTS")
print("=" * 60)

"""
KEY CAPABILITY: Imagine future trajectories!

Given:
- Initial state (h_0, z_0)
- Policy π(a | h, z)

Generate:
- Imagined trajectory: (h_0, z_0, a_0) -> (h_1, z_1, a_1) -> ...

Use cases:
- Planning: Evaluate action sequences
- Training: Generate data for policy training (Dreamer)
- Visualization: See what the model expects
"""

def imagine_rollout(rssm, policy, init_h, init_z, horizon):
    """
    Generate imaginary trajectory using prior only.

    Args:
        rssm: RSSM dynamics model
        policy: Policy network π(a | h, z)
        init_h: Initial hidden state
        init_z: Initial latent state
        horizon: Number of steps to imagine

    Returns:
        Trajectory of (h, z, a) tuples
    """
    h, z = init_h, init_z
    trajectory = []

    for t in range(horizon):
        # Get action from policy
        state = torch.cat([h, z], dim=-1)
        action = policy(state)

        # Save current state
        trajectory.append({'h': h.clone(), 'z': z.clone(), 'a': action.clone()})

        # Imagine next state (using prior, no observation!)
        h, z, _, _ = rssm.imagine_step(h, z, action)

    return trajectory

# Simple policy for testing
class SimplePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Linear(state_dim, action_dim)

    def forward(self, state):
        return torch.tanh(self.net(state))

policy = SimplePolicy(state_dim=230, action_dim=4)

# Imagine 15 steps into the future
trajectory = imagine_rollout(rssm, policy, h, z, horizon=15)
print(f"Imagined {len(trajectory)} steps into the future!")
print(f"Each step has: h={trajectory[0]['h'].shape}, z={trajectory[0]['z'].shape}, a={trajectory[0]['a'].shape}")

# -----------------------------------------------------------------------------
# STEP 6: Complete Latent Dynamics Module
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPLETE LATENT DYNAMICS MODULE")
print("=" * 60)

class LatentDynamics(nn.Module):
    """
    Complete latent dynamics module with:
    - Encoder (observation -> latent)
    - RSSM (latent dynamics)
    - Decoder (latent -> observation)
    - Reward predictor
    """
    def __init__(self, obs_shape, action_dim, latent_dim=32, hidden_dim=256):
        super().__init__()

        self.obs_shape = obs_shape
        obs_dim = obs_shape[0] * obs_shape[1] if len(obs_shape) > 1 else obs_shape[0]

        # Encoder: observation -> embedding
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # RSSM dynamics
        self.rssm = RSSM(
            obs_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            embed_dim=hidden_dim
        )

        # Decoder: latent -> observation
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Continue predictor (episode termination)
        self.continue_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def encode_obs(self, obs):
        """Encode observation to embedding."""
        obs_flat = obs.view(obs.shape[0], -1)
        return self.encoder(obs_flat)

    def decode(self, h, z):
        """Decode hidden + latent to observation."""
        features = torch.cat([h, z], dim=-1)
        return self.decoder(features)

    def predict_reward(self, h, z):
        """Predict reward from state."""
        features = torch.cat([h, z], dim=-1)
        return self.reward_head(features)

    def predict_continue(self, h, z):
        """Predict probability of episode continuing."""
        features = torch.cat([h, z], dim=-1)
        return self.continue_head(features)

    def forward(self, observations, actions, initial_h=None, initial_z=None):
        """
        Process a sequence of observations and actions.

        Args:
            observations: (batch, seq_len, *obs_shape)
            actions: (batch, seq_len, action_dim)

        Returns:
            Dictionary with reconstructions, predictions, distributions
        """
        batch_size, seq_len = observations.shape[:2]

        # Initialize hidden states
        if initial_h is None:
            h = self.rssm.init_hidden(batch_size).to(observations.device)
        else:
            h = initial_h

        if initial_z is None:
            z = torch.zeros(batch_size, self.rssm.latent_dim).to(observations.device)
        else:
            z = initial_z

        # Storage
        all_h, all_z = [], []
        all_prior_mean, all_prior_std = [], []
        all_post_mean, all_post_std = [], []
        all_recon, all_reward, all_cont = [], [], []

        for t in range(seq_len):
            obs = observations[:, t]
            action = actions[:, t] if t < seq_len else actions[:, -1]

            # Encode observation
            obs_embed = self.encode_obs(obs)

            # RSSM step with observation
            h, z, prior_mean, prior_std, post_mean, post_std = \
                self.rssm.observe_step(h, z, action, obs_embed)

            # Predictions
            recon = self.decode(h, z)
            reward = self.predict_reward(h, z)
            cont = self.predict_continue(h, z)

            # Store
            all_h.append(h)
            all_z.append(z)
            all_prior_mean.append(prior_mean)
            all_prior_std.append(prior_std)
            all_post_mean.append(post_mean)
            all_post_std.append(post_std)
            all_recon.append(recon)
            all_reward.append(reward)
            all_cont.append(cont)

        return {
            'h': torch.stack(all_h, dim=1),
            'z': torch.stack(all_z, dim=1),
            'prior_mean': torch.stack(all_prior_mean, dim=1),
            'prior_std': torch.stack(all_prior_std, dim=1),
            'post_mean': torch.stack(all_post_mean, dim=1),
            'post_std': torch.stack(all_post_std, dim=1),
            'recon': torch.stack(all_recon, dim=1),
            'reward_pred': torch.stack(all_reward, dim=1),
            'continue_pred': torch.stack(all_cont, dim=1),
        }

latent_dynamics = LatentDynamics(obs_shape=(64,), action_dim=4, latent_dim=32, hidden_dim=256)
print(latent_dynamics)

# Test full forward pass
obs_seq = torch.randn(8, 20, 64)  # batch=8, seq=20, obs_dim=64
act_seq = torch.randn(8, 20, 4)

outputs = latent_dynamics(obs_seq, act_seq)
print(f"\nInput observations: {obs_seq.shape}")
print(f"Input actions:      {act_seq.shape}")
print(f"Output hidden h:    {outputs['h'].shape}")
print(f"Output latent z:    {outputs['z'].shape}")
print(f"Reconstructions:    {outputs['recon'].shape}")
print(f"Reward predictions: {outputs['reward_pred'].shape}")

# -----------------------------------------------------------------------------
# KEY CONCEPTS FOR WORLD MODELS
# -----------------------------------------------------------------------------
"""
LATENT DYNAMICS IS THE HEART OF WORLD MODELS!

1. WHY LATENT SPACE?
   - Observations (images) are high-dimensional, noisy
   - Latent space is compact, captures essential dynamics
   - Much easier to predict z' than pixels!

2. WHY STOCHASTIC?
   - Real environments have uncertainty
   - Model can represent multiple possible futures
   - Important for planning under uncertainty

3. WHY RSSM?
   - Deterministic path (h) provides stable memory
   - Stochastic path (z) captures uncertainty
   - Prior/posterior separation enables imagination

4. TRAINING DYNAMICS:
   - Reconstruction loss: Learn good representations
   - KL divergence: Match prior to posterior
   - Reward loss: Predict task-relevant signals

5. IMAGINATION:
   - Use prior (no observations) to imagine futures
   - Train policy on imagined data
   - Massive sample efficiency!

NEXT: We put it all together into a complete world model!
"""

# -----------------------------------------------------------------------------
# EXERCISE:
# 1. Add convolutional encoder/decoder for image observations
# 2. Implement sequence training with proper batching
# 3. Add discount factor to reward predictions
# 4. Visualize imagination rollouts
# 5. Compare deterministic vs stochastic dynamics on a simple task
# -----------------------------------------------------------------------------
