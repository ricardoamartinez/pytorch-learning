# =============================================================================
# LESSON 14: Latent Dynamics Models
# =============================================================================
# The core of world models: predicting how latent states evolve over time.
# This combines VAE (for encoding) with sequence models (for dynamics).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import sys

from utils.lesson_utils import LessonRunner, ExerciseTest, ask_question, show_progress

lesson = LessonRunner(lesson_number=14, title="Latent Dynamics Models", total_points=80)

# =============================================================================
# SECTION 1: Latent Dynamics Concept
# =============================================================================
@lesson.section("Latent Dynamics Concept")
def latent_dynamics_concept():
    """
    WORLD MODEL ARCHITECTURE:

        Observation o_t --> [Encoder/VAE] --> Latent z_t
                                                  |
        Action a_t --------------------------> [Dynamics Model]
                                                  |
                                                  v
                                             Latent z_{t+1}
                                                  |
                                            [Decoder] (optional)
                                                  |
                                                  v
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
    print("World models predict in latent space, not pixel space!")
    print()
    print("Why latent space?")
    print("  - Observations: 64x64x3 = 12,288 dimensions")
    print("  - Latent state: 32-64 dimensions")
    print("  - Much easier to predict dynamics!")

# =============================================================================
# QUIZ 1: Latent Dynamics
# =============================================================================
@lesson.exercise("Quiz: Why Latent Space?", points=10)
def quiz_latent_space():
    print("Q: Why do world models predict in latent space instead of pixel space?")
    answer = ask_question([
        "A) Pixels are harder to display",
        "B) Latent space is compact and captures essential dynamics",
        "C) Latent space uses less GPU memory",
        "D) Pixels cannot be predicted by neural networks"
    ])

    if answer == "B":
        print("Correct! Latent space is low-dimensional and semantically meaningful.")
        return True
    else:
        print("The key reason: latent space is compact and captures essential dynamics.")
        return False

# =============================================================================
# SECTION 2: Deterministic Dynamics
# =============================================================================
@lesson.section("Deterministic Dynamics")
def deterministic_dynamics():
    """
    DETERMINISTIC DYNAMICS:

    Simplest model: z' = f(z, a)

    - No stochasticity, no memory
    - Works well for simple, predictable environments
    - Cannot represent uncertainty
    """
    class DeterministicDynamics(nn.Module):
        """z' = f(z, a)"""
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
            x = torch.cat([z, a], dim=-1)
            return self.network(x)

    model = DeterministicDynamics(latent_dim=32, action_dim=4)
    print(model)

    z = torch.randn(8, 32)
    a = torch.randn(8, 4)
    z_next = model(z, a)
    print(f"\nCurrent z: {z.shape}")
    print(f"Action:    {a.shape}")
    print(f"Next z:    {z_next.shape}")

    return DeterministicDynamics

# =============================================================================
# EXERCISE 1: Deterministic Dynamics
# =============================================================================
@lesson.exercise("Build Deterministic Dynamics", points=10)
def exercise_deterministic():
    """Build a deterministic dynamics model."""
    print("Create a DeterministicModel that:")
    print("  - Takes z (dim=16) and a (dim=2)")
    print("  - Returns next z (dim=16)")
    print("  - Uses a 64-unit hidden layer with ReLU")

    class DeterministicModel(nn.Module):
        def __init__(self, latent_dim=16, action_dim=2, hidden=64):
            super().__init__()
            # TODO: Implement f(z, a) -> z'
            self.net = nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, latent_dim)
            )

        def forward(self, z, a):
            x = torch.cat([z, a], dim=-1)
            return self.net(x)

    return DeterministicModel

class TestDeterministic(ExerciseTest):
    def run_tests(self, DeterministicModel):
        model = DeterministicModel()
        z = torch.randn(4, 16)
        a = torch.randn(4, 2)
        z_next = model(z, a)

        self.test_equal(z_next.shape, torch.Size([4, 16]), "Output shape")
        self.test_true(z_next.requires_grad, "Has gradient")

# =============================================================================
# SECTION 3: Stochastic Dynamics
# =============================================================================
@lesson.section("Stochastic Dynamics")
def stochastic_dynamics():
    """
    STOCHASTIC DYNAMICS:

    Real environments are often stochastic!
    - Same state + action can lead to different outcomes
    - Model this with a distribution: z' ~ N(μ(z,a), σ(z,a))

    Benefits:
    - Captures environment uncertainty
    - Can sample diverse futures
    - Better for planning under uncertainty
    """
    class StochasticDynamics(nn.Module):
        """z' ~ N(μ, σ) where μ, σ = f(z, a)"""
        def __init__(self, latent_dim, action_dim, hidden_dim=256):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mean_head = nn.Linear(hidden_dim, latent_dim)
            self.logstd_head = nn.Linear(hidden_dim, latent_dim)

        def forward(self, z, a, deterministic=False):
            x = torch.cat([z, a], dim=-1)
            h = self.network(x)

            mean = self.mean_head(h)
            logstd = self.logstd_head(h)
            std = F.softplus(logstd) + 1e-4

            if deterministic:
                return mean, mean, std

            # Reparameterization trick
            eps = torch.randn_like(std)
            z_next = mean + std * eps
            return z_next, mean, std

    model = StochasticDynamics(latent_dim=32, action_dim=4)
    z = torch.randn(8, 32)
    a = torch.randn(8, 4)
    z_next, mean, std = model(z, a)

    print("Stochastic dynamics outputs a distribution:")
    print(f"  Sampled z': {z_next.shape}")
    print(f"  Mean:       {mean.shape}")
    print(f"  Std:        {std.shape}")
    print(f"  Std range:  [{std.min().item():.4f}, {std.max().item():.4f}]")

    return StochasticDynamics

# =============================================================================
# EXERCISE 2: Stochastic Dynamics
# =============================================================================
@lesson.exercise("Build Stochastic Dynamics", points=10)
def exercise_stochastic():
    """Build a stochastic dynamics model."""
    print("Create a StochasticModel that:")
    print("  - Takes z (dim=16) and a (dim=2)")
    print("  - Outputs mean and std for next z distribution")
    print("  - Uses reparameterization trick for sampling")

    class StochasticModel(nn.Module):
        def __init__(self, latent_dim=16, action_dim=2, hidden=64):
            super().__init__()
            # TODO: Implement stochastic dynamics
            self.encoder = nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden),
                nn.ReLU()
            )
            self.mean_head = nn.Linear(hidden, latent_dim)
            self.logstd_head = nn.Linear(hidden, latent_dim)

        def forward(self, z, a):
            x = torch.cat([z, a], dim=-1)
            h = self.encoder(x)
            mean = self.mean_head(h)
            std = F.softplus(self.logstd_head(h)) + 1e-4

            # Reparameterization
            eps = torch.randn_like(std)
            z_next = mean + std * eps
            return z_next, mean, std

    return StochasticModel

class TestStochastic(ExerciseTest):
    def run_tests(self, StochasticModel):
        model = StochasticModel()
        z = torch.randn(4, 16)
        a = torch.randn(4, 2)
        z_next, mean, std = model(z, a)

        self.test_equal(z_next.shape, torch.Size([4, 16]), "Sample shape")
        self.test_equal(mean.shape, torch.Size([4, 16]), "Mean shape")
        self.test_equal(std.shape, torch.Size([4, 16]), "Std shape")
        self.test_true((std > 0).all().item(), "Std is positive")

# =============================================================================
# SECTION 4: RSSM Introduction
# =============================================================================
@lesson.section("RSSM Introduction")
def rssm_introduction():
    """
    RSSM (Recurrent State-Space Model) from Dreamer:

    Combines deterministic recurrence with stochastic state:
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
        h_t = f(h_{t-1}, z_{t-1}, a_{t-1})   # Deterministic transition
        prior:     z_t ~ p(z_t | h_t)        # Predict without observation
        posterior: z_t ~ q(z_t | h_t, o_t)   # Refine with observation
    """
    print("RSSM = Deterministic RNN + Stochastic latent")
    print()
    print("Key insight: Two distributions")
    print("  Prior:     What we predict (for imagination)")
    print("  Posterior: What we observe (for training)")
    print()
    print("Training: Match prior to posterior (KL loss)")
    print("Imagination: Use prior only (no observations needed)")

# =============================================================================
# QUIZ 2: RSSM Concepts
# =============================================================================
@lesson.exercise("Quiz: Prior vs Posterior", points=10)
def quiz_rssm():
    print("Q: In RSSM, when do we use the prior vs posterior?")
    answer = ask_question([
        "A) Prior for training, posterior for imagination",
        "B) Prior for imagination, posterior for training",
        "C) Both are used equally always",
        "D) Prior for encoding, posterior for decoding"
    ])

    if answer == "B":
        print("Correct! Prior is used for imagination, posterior for training.")
        return True
    else:
        print("Prior = imagination (no observations), Posterior = training (with observations).")
        return False

# =============================================================================
# SECTION 5: Building RSSM
# =============================================================================
@lesson.section("Building RSSM")
def building_rssm():
    """
    Let's build a complete RSSM module step by step.
    """
    class RSSM(nn.Module):
        """Recurrent State-Space Model from Dreamer."""
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

            # Prior: p(z_t | h_t)
            self.prior_net = nn.Sequential(
                nn.Linear(hidden_dim, embed_dim),
                nn.ReLU(),
            )
            self.prior_mean = nn.Linear(embed_dim, latent_dim)
            self.prior_logstd = nn.Linear(embed_dim, latent_dim)

            # Posterior: q(z_t | h_t, o_t)
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
            std = F.softplus(self.prior_logstd(x)) + 0.1
            return mean, std

        def posterior(self, h, obs_embed):
            """Compute posterior q(z | h, o)."""
            x = torch.cat([h, obs_embed], dim=-1)
            x = self.posterior_net(x)
            mean = self.posterior_mean(x)
            std = F.softplus(self.posterior_logstd(x)) + 0.1
            return mean, std

        def sample(self, mean, std):
            """Sample z using reparameterization."""
            return mean + std * torch.randn_like(std)

        def observe_step(self, prev_h, prev_z, prev_action, obs):
            """Step with observation (for training)."""
            rnn_input = torch.cat([prev_z, prev_action], dim=-1)
            h = self.rnn(rnn_input, prev_h)

            prior_mean, prior_std = self.prior(h)
            obs_embed = self.obs_encoder(obs)
            post_mean, post_std = self.posterior(h, obs_embed)
            z = self.sample(post_mean, post_std)

            return h, z, prior_mean, prior_std, post_mean, post_std

        def imagine_step(self, prev_h, prev_z, prev_action):
            """Step without observation (for imagination)."""
            rnn_input = torch.cat([prev_z, prev_action], dim=-1)
            h = self.rnn(rnn_input, prev_h)
            prior_mean, prior_std = self.prior(h)
            z = self.sample(prior_mean, prior_std)
            return h, z, prior_mean, prior_std

    rssm = RSSM(obs_dim=64, action_dim=4, hidden_dim=200, latent_dim=30)
    print(rssm)

    # Test observe step
    batch = 8
    h = rssm.init_hidden(batch)
    z = torch.randn(batch, 30)
    action = torch.randn(batch, 4)
    obs = torch.randn(batch, 64)

    h_new, z_new, prior_m, prior_s, post_m, post_s = rssm.observe_step(h, z, action, obs)
    print(f"\nObserve step: h={h_new.shape}, z={z_new.shape}")

    # Test imagine step
    h_imag, z_imag, _, _ = rssm.imagine_step(h, z, action)
    print(f"Imagine step: h={h_imag.shape}, z={z_imag.shape}")

    return RSSM

# =============================================================================
# EXERCISE 3: RSSM Prior Network
# =============================================================================
@lesson.exercise("Build RSSM Prior", points=10)
def exercise_rssm_prior():
    """Build the prior network for RSSM."""
    print("Create a PriorNetwork that:")
    print("  - Takes hidden state h (dim=64)")
    print("  - Outputs mean and std for z (dim=16)")

    class PriorNetwork(nn.Module):
        def __init__(self, hidden_dim=64, latent_dim=16):
            super().__init__()
            # TODO: Implement prior p(z | h)
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU()
            )
            self.mean_head = nn.Linear(32, latent_dim)
            self.logstd_head = nn.Linear(32, latent_dim)

        def forward(self, h):
            x = self.net(h)
            mean = self.mean_head(x)
            std = F.softplus(self.logstd_head(x)) + 0.1
            return mean, std

    return PriorNetwork

class TestPrior(ExerciseTest):
    def run_tests(self, PriorNetwork):
        model = PriorNetwork()
        h = torch.randn(4, 64)
        mean, std = model(h)

        self.test_equal(mean.shape, torch.Size([4, 16]), "Mean shape")
        self.test_equal(std.shape, torch.Size([4, 16]), "Std shape")
        self.test_true((std > 0).all().item(), "Std is positive")

# =============================================================================
# SECTION 6: KL Divergence Loss
# =============================================================================
@lesson.section("KL Divergence Loss")
def kl_divergence_section():
    """
    RSSM TRAINING LOSS:

    1. Reconstruction loss: ||decode(z) - obs||²

    2. KL divergence: KL(posterior || prior)
       Forces prior (imagination) to match posterior (reality)

    3. Reward prediction loss: ||predict_reward(h, z) - reward||²

    Total: L = L_recon + β * L_kl + L_reward
    """
    def kl_divergence_gaussian(mean1, std1, mean2, std2):
        """KL divergence: KL(N(mean1, std1) || N(mean2, std2))"""
        var1 = std1 ** 2
        var2 = std2 ** 2
        kl = 0.5 * (
            torch.log(var2 / var1)
            + var1 / var2
            + (mean1 - mean2) ** 2 / var2
            - 1
        )
        return kl.sum(dim=-1).mean()

    # Example
    post_mean = torch.randn(8, 16)
    post_std = torch.ones(8, 16) * 0.5
    prior_mean = torch.randn(8, 16)
    prior_std = torch.ones(8, 16)

    kl = kl_divergence_gaussian(post_mean, post_std, prior_mean, prior_std)
    print(f"KL divergence: {kl.item():.4f}")
    print()
    print("Training minimizes KL(posterior || prior)")
    print("This forces the prior to match what we observe!")

    return kl_divergence_gaussian

# =============================================================================
# EXERCISE 4: KL Divergence
# =============================================================================
@lesson.exercise("Implement KL Divergence", points=10)
def exercise_kl_divergence():
    """Implement KL divergence between two Gaussians."""
    print("Implement kl_divergence(mean1, std1, mean2, std2)")
    print("  KL(N(μ1,σ1) || N(μ2,σ2))")
    print("  = 0.5 * (log(σ2²/σ1²) + σ1²/σ2² + (μ1-μ2)²/σ2² - 1)")

    def kl_divergence(mean1, std1, mean2, std2):
        # TODO: Implement KL divergence
        var1 = std1 ** 2
        var2 = std2 ** 2
        kl = 0.5 * (
            torch.log(var2 / var1)
            + var1 / var2
            + (mean1 - mean2) ** 2 / var2
            - 1
        )
        return kl.sum(dim=-1).mean()

    return kl_divergence

class TestKL(ExerciseTest):
    def run_tests(self, kl_divergence):
        # Same distributions -> KL = 0
        mean = torch.randn(4, 8)
        std = torch.ones(4, 8)
        kl_same = kl_divergence(mean, std, mean, std)
        self.test_true(abs(kl_same.item()) < 1e-5, "KL(p||p) = 0")

        # Different distributions -> KL > 0
        mean2 = mean + 1.0
        kl_diff = kl_divergence(mean, std, mean2, std)
        self.test_true(kl_diff.item() > 0, "KL(p||q) > 0 for different distributions")

# =============================================================================
# SECTION 7: Imagination Rollouts
# =============================================================================
@lesson.section("Imagination Rollouts")
def imagination_rollouts():
    """
    KEY CAPABILITY: Imagine future trajectories!

    Given:
    - Initial state (h_0, z_0)
    - Policy π(a | h, z)

    Generate:
    - Imagined trajectory: (h_0, z_0, a_0) -> (h_1, z_1, a_1) -> ...

    Use cases:
    - Planning: Evaluate action sequences
    - Training: Generate data for policy (Dreamer)
    - Visualization: See what model expects
    """
    class SimplePolicy(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Linear(state_dim, action_dim)

        def forward(self, state):
            return torch.tanh(self.net(state))

    # The imagination loop
    print("Imagination loop (pseudocode):")
    print("""
    def imagine_rollout(rssm, policy, init_h, init_z, horizon):
        h, z = init_h, init_z
        trajectory = []

        for t in range(horizon):
            # Get action from policy
            state = concat(h, z)
            action = policy(state)

            # Save current state
            trajectory.append((h, z, action))

            # Imagine next state (prior only!)
            h, z, _, _ = rssm.imagine_step(h, z, action)

        return trajectory
    """)

    return SimplePolicy

# =============================================================================
# EXERCISE 5: Imagination Rollout
# =============================================================================
@lesson.exercise("Implement Imagination Rollout", points=10)
def exercise_imagination():
    """Implement imagination rollout function."""
    print("Create a function that imagines future trajectories")
    print("  - Takes initial h, z, and a simple dynamics model")
    print("  - Returns list of (h, z) tuples for each step")

    class SimpleDynamics(nn.Module):
        def __init__(self, hidden_dim=32, latent_dim=16, action_dim=4):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim
            self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)
            self.prior = nn.Linear(hidden_dim, latent_dim * 2)

        def imagine_step(self, h, z, action):
            rnn_input = torch.cat([z, action], dim=-1)
            h_new = self.rnn(rnn_input, h)
            prior_out = self.prior(h_new)
            mean, logstd = prior_out.chunk(2, dim=-1)
            std = F.softplus(logstd) + 0.1
            z_new = mean + std * torch.randn_like(std)
            return h_new, z_new

    def imagine_rollout(dynamics, init_h, init_z, actions):
        """
        Imagine a trajectory given a sequence of actions.

        Args:
            dynamics: Dynamics model with imagine_step method
            init_h: Initial hidden state [batch, hidden_dim]
            init_z: Initial latent state [batch, latent_dim]
            actions: Action sequence [batch, horizon, action_dim]

        Returns:
            List of (h, z) tuples for each timestep
        """
        # TODO: Implement imagination loop
        h, z = init_h, init_z
        trajectory = []
        horizon = actions.shape[1]

        for t in range(horizon):
            trajectory.append((h.clone(), z.clone()))
            h, z = dynamics.imagine_step(h, z, actions[:, t])

        return trajectory

    return imagine_rollout, SimpleDynamics

class TestImagination(ExerciseTest):
    def run_tests(self, result):
        imagine_rollout, SimpleDynamics = result
        dynamics = SimpleDynamics()

        batch = 4
        horizon = 5
        h = torch.zeros(batch, 32)
        z = torch.randn(batch, 16)
        actions = torch.randn(batch, horizon, 4)

        trajectory = imagine_rollout(dynamics, h, z, actions)

        self.test_equal(len(trajectory), horizon, "Trajectory length")
        self.test_equal(trajectory[0][0].shape, torch.Size([batch, 32]), "Hidden shape")
        self.test_equal(trajectory[0][1].shape, torch.Size([batch, 16]), "Latent shape")

# =============================================================================
# SECTION 8: Complete Latent Dynamics Module
# =============================================================================
@lesson.section("Complete Latent Dynamics Module")
def complete_module():
    """
    A complete latent dynamics module includes:
    - Encoder: observation -> latent
    - RSSM: latent dynamics
    - Decoder: latent -> observation
    - Reward predictor: latent -> reward

    This is the core of world models like Dreamer!
    """
    class LatentDynamics(nn.Module):
        """Complete latent dynamics with encoder, RSSM, decoder."""
        def __init__(self, obs_dim, action_dim, hidden_dim=256, latent_dim=32):
            super().__init__()

            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            # RNN for recurrence
            self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)

            # Prior and posterior
            self.prior_mean = nn.Linear(hidden_dim, latent_dim)
            self.prior_logstd = nn.Linear(hidden_dim, latent_dim)
            self.post_net = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
            self.post_mean = nn.Linear(hidden_dim, latent_dim)
            self.post_logstd = nn.Linear(hidden_dim, latent_dim)

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim + latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, obs_dim),
            )

            # Reward predictor
            self.reward_head = nn.Sequential(
                nn.Linear(hidden_dim + latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, obs, actions):
            """Process observation sequence."""
            batch, seq_len = obs.shape[:2]

            h = torch.zeros(batch, self.hidden_dim, device=obs.device)
            z = torch.zeros(batch, self.latent_dim, device=obs.device)

            outputs = {'h': [], 'z': [], 'recon': [], 'reward': []}

            for t in range(seq_len):
                # Encode observation
                obs_embed = self.encoder(obs[:, t])

                # RNN step
                rnn_in = torch.cat([z, actions[:, t]], dim=-1)
                h = self.rnn(rnn_in, h)

                # Posterior
                post_in = torch.cat([h, obs_embed], dim=-1)
                post_h = F.relu(self.post_net(post_in))
                mean = self.post_mean(post_h)
                std = F.softplus(self.post_logstd(post_h)) + 0.1
                z = mean + std * torch.randn_like(std)

                # Decode
                features = torch.cat([h, z], dim=-1)
                recon = self.decoder(features)
                reward = self.reward_head(features)

                outputs['h'].append(h)
                outputs['z'].append(z)
                outputs['recon'].append(recon)
                outputs['reward'].append(reward)

            for k in outputs:
                outputs[k] = torch.stack(outputs[k], dim=1)
            return outputs

    model = LatentDynamics(obs_dim=64, action_dim=4, hidden_dim=128, latent_dim=32)
    obs = torch.randn(8, 20, 64)
    actions = torch.randn(8, 20, 4)
    out = model(obs, actions)

    print("Complete Latent Dynamics Module:")
    print(f"  Input obs:    {obs.shape}")
    print(f"  Input actions: {actions.shape}")
    print(f"  Output h:      {out['h'].shape}")
    print(f"  Output z:      {out['z'].shape}")
    print(f"  Reconstructions: {out['recon'].shape}")
    print(f"  Reward preds:  {out['reward'].shape}")

    return LatentDynamics

# =============================================================================
# FINAL CHALLENGE: Mini World Model
# =============================================================================
@lesson.exercise("Challenge: Mini World Model", points=10)
def challenge_world_model():
    """Build a minimal world model with all components."""
    print("Create a MiniWorldModel with:")
    print("  - Encoder: obs_dim -> latent_dim")
    print("  - Dynamics: (z, action) -> next_z")
    print("  - Decoder: z -> obs_dim")
    print("  - Method: predict(obs, action) -> next_obs")

    class MiniWorldModel(nn.Module):
        def __init__(self, obs_dim=32, action_dim=4, latent_dim=16):
            super().__init__()
            # TODO: Implement complete world model
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, latent_dim),
                nn.ReLU()
            )
            self.dynamics = nn.Sequential(
                nn.Linear(latent_dim + action_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, obs_dim)
            )

        def encode(self, obs):
            return self.encoder(obs)

        def step(self, z, action):
            x = torch.cat([z, action], dim=-1)
            return self.dynamics(x)

        def decode(self, z):
            return self.decoder(z)

        def predict(self, obs, action):
            """Predict next observation given current obs and action."""
            z = self.encode(obs)
            z_next = self.step(z, action)
            return self.decode(z_next)

    return MiniWorldModel

class TestMiniWorldModel(ExerciseTest):
    def run_tests(self, MiniWorldModel):
        model = MiniWorldModel()
        obs = torch.randn(4, 32)
        action = torch.randn(4, 4)

        # Test encode
        z = model.encode(obs)
        self.test_equal(z.shape, torch.Size([4, 16]), "Encode shape")

        # Test step
        z_next = model.step(z, action)
        self.test_equal(z_next.shape, torch.Size([4, 16]), "Dynamics shape")

        # Test decode
        obs_pred = model.decode(z_next)
        self.test_equal(obs_pred.shape, torch.Size([4, 32]), "Decode shape")

        # Test predict
        next_obs = model.predict(obs, action)
        self.test_equal(next_obs.shape, torch.Size([4, 32]), "Predict shape")

# =============================================================================
# MAIN
# =============================================================================
def main():
    """Run the lesson."""
    if "--test" in sys.argv:
        results = []

        # Run all tests
        DeterministicModel = exercise_deterministic()
        results.append(("Deterministic", TestDeterministic().run_tests(DeterministicModel)))

        StochasticModel = exercise_stochastic()
        results.append(("Stochastic", TestStochastic().run_tests(StochasticModel)))

        PriorNetwork = exercise_rssm_prior()
        results.append(("RSSM Prior", TestPrior().run_tests(PriorNetwork)))

        kl_div = exercise_kl_divergence()
        results.append(("KL Divergence", TestKL().run_tests(kl_div)))

        imag_result = exercise_imagination()
        results.append(("Imagination", TestImagination().run_tests(imag_result)))

        MiniWorldModel = challenge_world_model()
        results.append(("Mini World Model", TestMiniWorldModel().run_tests(MiniWorldModel)))

        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        for name, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")

        all_passed = all(r[1] for r in results)
        print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))
        return all_passed

    else:
        # Interactive mode
        lesson.run_section(latent_dynamics_concept)
        input("\nPress Enter to continue...")

        lesson.run_exercise(quiz_latent_space)
        input("\nPress Enter to continue...")

        lesson.run_section(deterministic_dynamics)
        input("\nPress Enter to continue...")

        lesson.run_exercise(exercise_deterministic, TestDeterministic)
        input("\nPress Enter to continue...")

        lesson.run_section(stochastic_dynamics)
        input("\nPress Enter to continue...")

        lesson.run_exercise(exercise_stochastic, TestStochastic)
        input("\nPress Enter to continue...")

        lesson.run_section(rssm_introduction)
        input("\nPress Enter to continue...")

        lesson.run_exercise(quiz_rssm)
        input("\nPress Enter to continue...")

        lesson.run_section(building_rssm)
        input("\nPress Enter to continue...")

        lesson.run_exercise(exercise_rssm_prior, TestPrior)
        input("\nPress Enter to continue...")

        lesson.run_section(kl_divergence_section)
        input("\nPress Enter to continue...")

        lesson.run_exercise(exercise_kl_divergence, TestKL)
        input("\nPress Enter to continue...")

        lesson.run_section(imagination_rollouts)
        input("\nPress Enter to continue...")

        lesson.run_exercise(exercise_imagination, TestImagination)
        input("\nPress Enter to continue...")

        lesson.run_section(complete_module)
        input("\nPress Enter to continue...")

        lesson.run_exercise(challenge_world_model, TestMiniWorldModel)

        show_progress(lesson)

if __name__ == "__main__":
    main()
