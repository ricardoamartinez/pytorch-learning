# =============================================================================
# LESSON 15: Complete World Model Implementation
# =============================================================================
# Putting it all together: A complete world model for learning and planning.
# Based on the Dreamer architecture (Hafner et al.)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import sys

from utils.lesson_utils import LessonRunner, ExerciseTest, ask_question, show_progress

lesson = LessonRunner(lesson_number=15, title="Complete World Model", total_points=80)

# =============================================================================
# SECTION 1: World Model Architecture
# =============================================================================
@lesson.section("World Model Architecture")
def world_model_architecture():
    """
    WORLD MODEL = Encoder + Dynamics Model + Decoder + Reward/Value Heads

        +------------------------------------------------------+
        |                    WORLD MODEL                       |
        |                                                      |
        |   Observation --> [Encoder] --> Latent z             |
        |                                    |                  |
        |                               [RSSM Dynamics] <-- Action
        |                                    |                  |
        |                               Next Latent z'         |
        |                                    |                  |
        |                     +--------------+-------------+    |
        |                     |                            |    |
        |            [Reward Head] --> r           [Decoder] --> o'
        +------------------------------------------------------+

    TRAINING LOOP:
    1. Collect real experience (observations, actions, rewards)
    2. Train world model on collected data
    3. Imagine trajectories using world model
    4. Train policy/value on imagined data
    5. Repeat

    This is the "Dreamer" approach!
    """
    print("World Model combines everything we've learned!")
    print()
    print("Components:")
    print("  1. Encoder (VAE/CNN) - Lesson 9-10")
    print("  2. Dynamics (RSSM) - Lesson 14")
    print("  3. Decoder - Lesson 9-10")
    print("  4. Actor-Critic - Lesson 13")
    print()
    print("Key insight: Train policy on IMAGINED experience!")

# =============================================================================
# QUIZ 1: World Model Concepts
# =============================================================================
@lesson.exercise("Quiz: World Model Training", points=10)
def quiz_world_model():
    print("Q: What is the main advantage of training on imagined trajectories?")
    answer = ask_question([
        "A) It's faster to compute gradients",
        "B) Massive sample efficiency - no real environment needed",
        "C) The model learns better representations",
        "D) It reduces GPU memory usage"
    ])

    if answer == "B":
        print("Correct! We can generate unlimited imagined experience!")
        return True
    else:
        print("Main advantage: sample efficiency - train on imagined data.")
        return False

# =============================================================================
# SECTION 2: Image Encoder/Decoder
# =============================================================================
@lesson.section("Image Encoder/Decoder")
def image_encoder_decoder():
    """
    For image-based environments, we use convolutional encoder/decoder.

    Encoder: Image (64x64x3) --> Features (256)
    Decoder: Features (256) --> Image (64x64x3)
    """
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

    encoder = ConvEncoder(input_channels=3, feature_dim=256)
    decoder = ConvDecoder(feature_dim=256, output_channels=3)

    img = torch.randn(4, 3, 64, 64)
    features = encoder(img)
    recon = decoder(features)

    print(f"Input image:   {img.shape}")
    print(f"Features:      {features.shape}")
    print(f"Reconstructed: {recon.shape}")

    return ConvEncoder, ConvDecoder

# =============================================================================
# EXERCISE 1: Simple Encoder
# =============================================================================
@lesson.exercise("Build Simple Encoder", points=10)
def exercise_encoder():
    """Build a simple encoder for vector observations."""
    print("Create a SimpleEncoder that:")
    print("  - Takes observation (dim=32)")
    print("  - Returns embedding (dim=64)")
    print("  - Uses two hidden layers with ReLU")

    class SimpleEncoder(nn.Module):
        def __init__(self, obs_dim=32, embed_dim=64, hidden=128):
            super().__init__()
            # TODO: Implement encoder
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, embed_dim)
            )

        def forward(self, x):
            return self.net(x)

    return SimpleEncoder

class TestEncoder(ExerciseTest):
    def run_tests(self, SimpleEncoder):
        encoder = SimpleEncoder()
        x = torch.randn(4, 32)
        embed = encoder(x)

        self.test_equal(embed.shape, torch.Size([4, 64]), "Output shape")
        self.test_true(embed.requires_grad, "Has gradient")

# =============================================================================
# SECTION 3: RSSM for Dynamics
# =============================================================================
@lesson.section("RSSM Dynamics Model")
def rssm_dynamics():
    """
    RSSM (Recurrent State-Space Model) for latent dynamics.

    State = (h, z) where:
    - h: Deterministic recurrent state (memory)
    - z: Stochastic state (sampled)

    Key methods:
    - observe(): Step with observation (training)
    - imagine(): Step without observation (planning)
    """
    class RSSM(nn.Module):
        def __init__(self, embed_dim, action_dim, hidden_dim, latent_dim):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim

            # Recurrent model
            self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)

            # Prior: p(z | h)
            self.prior_fc = nn.Linear(hidden_dim, hidden_dim)
            self.prior_mean = nn.Linear(hidden_dim, latent_dim)
            self.prior_std = nn.Linear(hidden_dim, latent_dim)

            # Posterior: q(z | h, embed)
            self.posterior_fc = nn.Linear(hidden_dim + embed_dim, hidden_dim)
            self.posterior_mean = nn.Linear(hidden_dim, latent_dim)
            self.posterior_std = nn.Linear(hidden_dim, latent_dim)

        def init_state(self, batch_size, device='cpu'):
            h = torch.zeros(batch_size, self.hidden_dim, device=device)
            z = torch.zeros(batch_size, self.latent_dim, device=device)
            return h, z

        def get_prior(self, h):
            x = F.relu(self.prior_fc(h))
            mean = self.prior_mean(x)
            std = F.softplus(self.prior_std(x)) + 0.1
            return mean, std

        def get_posterior(self, h, embed):
            x = torch.cat([h, embed], dim=-1)
            x = F.relu(self.posterior_fc(x))
            mean = self.posterior_mean(x)
            std = F.softplus(self.posterior_std(x)) + 0.1
            return mean, std

        def sample(self, mean, std):
            return mean + std * torch.randn_like(std)

        def observe(self, prev_h, prev_z, action, embed):
            """Step with observation (training)."""
            x = torch.cat([prev_z, action], dim=-1)
            h = self.rnn(x, prev_h)

            prior_mean, prior_std = self.get_prior(h)
            post_mean, post_std = self.get_posterior(h, embed)
            z = self.sample(post_mean, post_std)

            return h, z, prior_mean, prior_std, post_mean, post_std

        def imagine(self, prev_h, prev_z, action):
            """Step without observation (imagination)."""
            x = torch.cat([prev_z, action], dim=-1)
            h = self.rnn(x, prev_h)

            prior_mean, prior_std = self.get_prior(h)
            z = self.sample(prior_mean, prior_std)

            return h, z

    rssm = RSSM(embed_dim=64, action_dim=4, hidden_dim=128, latent_dim=32)

    batch = 4
    h, z = rssm.init_state(batch)
    action = torch.randn(batch, 4)
    embed = torch.randn(batch, 64)

    h_new, z_new, *_ = rssm.observe(h, z, action, embed)
    print(f"Observe step: h={h_new.shape}, z={z_new.shape}")

    h_imag, z_imag = rssm.imagine(h, z, action)
    print(f"Imagine step: h={h_imag.shape}, z={z_imag.shape}")

    return RSSM

# =============================================================================
# QUIZ 2: RSSM Concepts
# =============================================================================
@lesson.exercise("Quiz: Observe vs Imagine", points=10)
def quiz_rssm():
    print("Q: What's the difference between RSSM.observe() and RSSM.imagine()?")
    answer = ask_question([
        "A) observe uses prior, imagine uses posterior",
        "B) observe uses posterior (with observation), imagine uses prior",
        "C) They are the same, just different names",
        "D) observe predicts actions, imagine predicts states"
    ])

    if answer == "B":
        print("Correct! observe has access to real observations, imagine doesn't.")
        return True
    else:
        print("observe uses posterior (real data), imagine uses prior only.")
        return False

# =============================================================================
# SECTION 4: Complete World Model
# =============================================================================
@lesson.section("Complete World Model")
def complete_world_model():
    """
    Complete World Model combining:
    - Encoder/Decoder
    - RSSM dynamics
    - Reward predictor
    - Continue predictor
    """
    class WorldModel(nn.Module):
        def __init__(self, obs_dim=32, action_dim=4,
                     embed_dim=64, hidden_dim=128, latent_dim=32):
            super().__init__()

            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim
            self.feature_dim = hidden_dim + latent_dim

            # Encoder/Decoder
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.feature_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, obs_dim)
            )

            # RSSM components
            self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)
            self.prior_net = nn.Linear(hidden_dim, latent_dim * 2)
            self.post_net = nn.Linear(hidden_dim + embed_dim, latent_dim * 2)

            # Heads
            self.reward_head = nn.Linear(self.feature_dim, 1)
            self.continue_head = nn.Linear(self.feature_dim, 1)

        def init_state(self, batch_size, device='cpu'):
            h = torch.zeros(batch_size, self.hidden_dim, device=device)
            z = torch.zeros(batch_size, self.latent_dim, device=device)
            return h, z

        def get_features(self, h, z):
            return torch.cat([h, z], dim=-1)

        def observe_step(self, h, z, action, obs):
            # Encode observation
            embed = self.encoder(obs)

            # RNN step
            rnn_in = torch.cat([z, action], dim=-1)
            h = self.rnn(rnn_in, h)

            # Posterior
            post_out = self.post_net(torch.cat([h, embed], dim=-1))
            post_mean, post_logstd = post_out.chunk(2, dim=-1)
            post_std = F.softplus(post_logstd) + 0.1
            z = post_mean + post_std * torch.randn_like(post_std)

            # Prior (for KL loss)
            prior_out = self.prior_net(h)
            prior_mean, prior_logstd = prior_out.chunk(2, dim=-1)
            prior_std = F.softplus(prior_logstd) + 0.1

            return h, z, prior_mean, prior_std, post_mean, post_std

        def imagine_step(self, h, z, action):
            # RNN step
            rnn_in = torch.cat([z, action], dim=-1)
            h = self.rnn(rnn_in, h)

            # Sample from prior
            prior_out = self.prior_net(h)
            prior_mean, prior_logstd = prior_out.chunk(2, dim=-1)
            prior_std = F.softplus(prior_logstd) + 0.1
            z = prior_mean + prior_std * torch.randn_like(prior_std)

            return h, z

        def decode(self, h, z):
            features = self.get_features(h, z)
            return self.decoder(features)

        def predict_reward(self, h, z):
            features = self.get_features(h, z)
            return self.reward_head(features).squeeze(-1)

    model = WorldModel(obs_dim=32, action_dim=4)

    batch = 4
    h, z = model.init_state(batch)
    obs = torch.randn(batch, 32)
    action = torch.randn(batch, 4)

    h, z, *_ = model.observe_step(h, z, action, obs)
    recon = model.decode(h, z)
    reward = model.predict_reward(h, z)

    print(f"Hidden state: {h.shape}")
    print(f"Latent state: {z.shape}")
    print(f"Reconstruction: {recon.shape}")
    print(f"Reward prediction: {reward.shape}")

    return WorldModel

# =============================================================================
# EXERCISE 2: World Model Forward Pass
# =============================================================================
@lesson.exercise("Build World Model", points=10)
def exercise_world_model():
    """Build a simple world model with all components."""
    print("Create a MiniWorldModel with:")
    print("  - encoder: obs_dim -> embed_dim")
    print("  - rnn: GRUCell for dynamics")
    print("  - decoder: features -> obs_dim")
    print("  - reward_head: features -> 1")

    class MiniWorldModel(nn.Module):
        def __init__(self, obs_dim=16, action_dim=2, hidden_dim=32, latent_dim=8):
            super().__init__()
            # TODO: Implement world model
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim

            self.encoder = nn.Linear(obs_dim, hidden_dim)
            self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)
            self.prior = nn.Linear(hidden_dim, latent_dim * 2)
            self.decoder = nn.Linear(hidden_dim + latent_dim, obs_dim)
            self.reward_head = nn.Linear(hidden_dim + latent_dim, 1)

        def init_state(self, batch_size):
            h = torch.zeros(batch_size, self.hidden_dim)
            z = torch.zeros(batch_size, self.latent_dim)
            return h, z

        def step(self, h, z, action, obs=None):
            # RNN update
            rnn_in = torch.cat([z, action], dim=-1)
            h = self.rnn(rnn_in, h)

            # Sample z from prior
            prior_out = self.prior(h)
            mean, logstd = prior_out.chunk(2, dim=-1)
            std = F.softplus(logstd) + 0.1
            z = mean + std * torch.randn_like(std)

            return h, z

        def predict(self, h, z):
            features = torch.cat([h, z], dim=-1)
            obs_pred = self.decoder(features)
            reward_pred = self.reward_head(features).squeeze(-1)
            return obs_pred, reward_pred

    return MiniWorldModel

class TestWorldModel(ExerciseTest):
    def run_tests(self, MiniWorldModel):
        model = MiniWorldModel()
        batch = 4

        h, z = model.init_state(batch)
        self.test_equal(h.shape, torch.Size([batch, 32]), "Hidden shape")
        self.test_equal(z.shape, torch.Size([batch, 8]), "Latent shape")

        action = torch.randn(batch, 2)
        h, z = model.step(h, z, action)
        self.test_equal(h.shape, torch.Size([batch, 32]), "Updated hidden")
        self.test_equal(z.shape, torch.Size([batch, 8]), "Updated latent")

        obs_pred, reward_pred = model.predict(h, z)
        self.test_equal(obs_pred.shape, torch.Size([batch, 16]), "Obs prediction")
        self.test_equal(reward_pred.shape, torch.Size([batch]), "Reward prediction")

# =============================================================================
# SECTION 5: World Model Loss
# =============================================================================
@lesson.section("World Model Loss")
def world_model_loss_section():
    """
    WORLD MODEL TRAINING LOSS:

    1. Reconstruction loss: ||decode(z) - obs||²
    2. KL divergence: KL(posterior || prior)
    3. Reward loss: ||predict_reward(z) - reward||²
    4. Continue loss: BCE(predict_continue(z), done)

    Total: L = L_recon + β * L_kl + L_reward + L_continue
    """
    def compute_world_model_loss(recon, obs, prior_mean, prior_std,
                                  post_mean, post_std, reward_pred, reward,
                                  kl_weight=1.0):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, obs)

        # KL divergence
        prior_var = prior_std ** 2
        post_var = post_std ** 2
        kl = 0.5 * (
            torch.log(prior_var / post_var)
            + post_var / prior_var
            + (post_mean - prior_mean) ** 2 / prior_var
            - 1
        ).sum(dim=-1).mean()

        # Reward loss
        reward_loss = F.mse_loss(reward_pred, reward)

        total = recon_loss + kl_weight * kl + reward_loss

        return {
            'total': total,
            'recon': recon_loss,
            'kl': kl,
            'reward': reward_loss
        }

    # Example
    batch = 4
    latent_dim = 8

    obs = torch.randn(batch, 16)
    recon = torch.randn(batch, 16)
    prior_mean = torch.randn(batch, latent_dim)
    prior_std = torch.ones(batch, latent_dim)
    post_mean = torch.randn(batch, latent_dim)
    post_std = torch.ones(batch, latent_dim) * 0.5
    reward_pred = torch.randn(batch)
    reward = torch.randn(batch)

    losses = compute_world_model_loss(
        recon, obs, prior_mean, prior_std, post_mean, post_std,
        reward_pred, reward
    )

    print(f"Total loss:  {losses['total'].item():.4f}")
    print(f"Recon loss:  {losses['recon'].item():.4f}")
    print(f"KL loss:     {losses['kl'].item():.4f}")
    print(f"Reward loss: {losses['reward'].item():.4f}")

    return compute_world_model_loss

# =============================================================================
# EXERCISE 3: KL Loss
# =============================================================================
@lesson.exercise("Implement KL Loss", points=10)
def exercise_kl_loss():
    """Implement the KL divergence component of world model loss."""
    print("Implement kl_loss(prior_mean, prior_std, post_mean, post_std)")
    print("  Returns mean KL divergence over batch")

    def kl_loss(prior_mean, prior_std, post_mean, post_std):
        # TODO: Implement KL divergence
        prior_var = prior_std ** 2
        post_var = post_std ** 2
        kl = 0.5 * (
            torch.log(prior_var / post_var)
            + post_var / prior_var
            + (post_mean - prior_mean) ** 2 / prior_var
            - 1
        )
        return kl.sum(dim=-1).mean()

    return kl_loss

class TestKLLoss(ExerciseTest):
    def run_tests(self, kl_loss):
        # Same distribution -> KL = 0
        mean = torch.randn(4, 8)
        std = torch.ones(4, 8)
        kl = kl_loss(mean, std, mean, std)
        self.test_true(abs(kl.item()) < 1e-5, "KL(p||p) = 0")

        # Different -> KL > 0
        mean2 = mean + 1.0
        kl_diff = kl_loss(mean, std, mean2, std)
        self.test_true(kl_diff.item() > 0, "KL > 0 for different dists")

# =============================================================================
# SECTION 6: Actor-Critic for Imagination
# =============================================================================
@lesson.section("Actor-Critic in Imagination")
def actor_critic_imagination():
    """
    Train actor-critic on IMAGINED trajectories!

    This is the key insight of Dreamer:
    - Use world model to imagine future
    - Train policy on imagined rewards
    - No real environment interaction needed!
    """
    class Actor(nn.Module):
        def __init__(self, feature_dim, action_dim, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim * 2)
            )
            self.action_dim = action_dim

        def forward(self, features):
            out = self.net(features)
            mean, logstd = out.chunk(2, dim=-1)
            std = F.softplus(logstd) + 0.1
            return mean, std

        def sample(self, features):
            mean, std = self(features)
            eps = torch.randn_like(mean)
            action = mean + std * eps
            return torch.tanh(action)  # Bound to [-1, 1]

    class Critic(nn.Module):
        def __init__(self, feature_dim, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, features):
            return self.net(features).squeeze(-1)

    feature_dim = 64
    actor = Actor(feature_dim, action_dim=4)
    critic = Critic(feature_dim)

    features = torch.randn(8, feature_dim)
    action = actor.sample(features)
    value = critic(features)

    print(f"Features: {features.shape}")
    print(f"Action:   {action.shape}")
    print(f"Value:    {value.shape}")

    return Actor, Critic

# =============================================================================
# EXERCISE 4: Lambda Returns
# =============================================================================
@lesson.exercise("Implement Lambda Returns", points=10)
def exercise_lambda_returns():
    """Implement lambda-returns for advantage estimation."""
    print("Implement compute_returns(rewards, values, gamma, lambda_)")
    print("  Lambda-returns blend TD and MC estimates")

    def compute_returns(rewards, values, gamma=0.99, lambda_=0.95):
        """
        Compute lambda-returns for a trajectory.

        Args:
            rewards: [batch, T] tensor of rewards
            values: [batch, T] tensor of value estimates
            gamma: discount factor
            lambda_: GAE lambda

        Returns:
            returns: [batch, T] tensor of lambda-returns
        """
        # TODO: Implement lambda-returns
        B, T = rewards.shape
        returns = torch.zeros_like(rewards)

        # Bootstrap from last value
        returns[:, -1] = values[:, -1]

        for t in reversed(range(T - 1)):
            returns[:, t] = (
                rewards[:, t]
                + gamma * (
                    (1 - lambda_) * values[:, t + 1]
                    + lambda_ * returns[:, t + 1]
                )
            )

        return returns

    return compute_returns

class TestLambdaReturns(ExerciseTest):
    def run_tests(self, compute_returns):
        batch = 2
        T = 5
        rewards = torch.ones(batch, T)
        values = torch.zeros(batch, T)

        returns = compute_returns(rewards, values, gamma=1.0, lambda_=1.0)

        # With gamma=1, lambda=1, returns should sum up
        self.test_equal(returns.shape, torch.Size([batch, T]), "Output shape")
        # First return should be sum of all rewards
        self.test_true(returns[0, 0].item() > returns[0, -1].item(),
                      "Earlier returns are larger")

# =============================================================================
# SECTION 7: Dreamer Training Loop
# =============================================================================
@lesson.section("Dreamer Training Loop")
def dreamer_training():
    """
    DREAMER TRAINING LOOP (Pseudocode):

    1. ENVIRONMENT INTERACTION
       - Collect experience using current policy
       - Store in replay buffer

    2. WORLD MODEL TRAINING
       - Sample batch from replay
       - Compute reconstruction + KL + reward losses
       - Update world model

    3. BEHAVIOR LEARNING (IMAGINATION)
       - Get starting states from replay
       - Imagine trajectories using world model
       - Compute lambda-returns
       - Update actor to maximize returns
       - Update critic to predict returns

    4. REPEAT
    """
    print("Dreamer Training Loop:")
    print()
    print("1. Collect real experience -> Replay Buffer")
    print("2. Train World Model on replay data")
    print("3. Imagine trajectories (15-50 steps)")
    print("4. Train Actor-Critic on imagined data")
    print("5. Repeat")
    print()
    print("Key insight: Step 3-4 uses NO real environment!")
    print("This is what makes Dreamer sample-efficient.")

# =============================================================================
# QUIZ 3: Dreamer Concepts
# =============================================================================
@lesson.exercise("Quiz: Dreamer Training", points=10)
def quiz_dreamer():
    print("Q: Where does most of the policy training happen in Dreamer?")
    answer = ask_question([
        "A) Directly in the real environment",
        "B) In imagination using the world model",
        "C) In the replay buffer",
        "D) During preprocessing"
    ])

    if answer == "B":
        print("Correct! Policy trains on imagined trajectories from world model.")
        return True
    else:
        print("Dreamer trains policy on IMAGINED trajectories, not real ones.")
        return False

# =============================================================================
# SECTION 8: World Model Summary
# =============================================================================
@lesson.section("World Model Summary")
def world_model_summary():
    """
    CONGRATULATIONS!

    You've learned to build a complete world model with:

    1. ENCODER (Lessons 9-10)
       - Compress observations to latent space
       - VAE, CNN, VQ-VAE

    2. DYNAMICS MODEL (Lesson 14)
       - Predict latent state transitions
       - RSSM with prior/posterior

    3. DECODER (Lessons 9-10)
       - Reconstruct observations from latent
       - For visualization and training

    4. ACTOR-CRITIC (Lesson 13)
       - Policy for action selection
       - Value function for returns

    FAMOUS WORLD MODELS:
    - Dreamer (v1, v2, v3): RSSM + Actor-Critic
    - MuZero: Learned model + MCTS planning
    - IRIS: Transformer dynamics + discrete tokens
    - Genie: Generative world model from video

    You now have the foundation to understand and implement
    state-of-the-art world models!
    """
    print("=" * 50)
    print("  CONGRATULATIONS ON COMPLETING THE COURSE!")
    print("=" * 50)
    print()
    print("You've learned:")
    print("  - PyTorch basics (Lessons 1-4)")
    print("  - Training loops (Lesson 5)")
    print("  - CNNs (Lesson 6)")
    print("  - RNNs/LSTMs (Lessons 7-8)")
    print("  - Autoencoders/VAEs (Lessons 9-10)")
    print("  - Attention/Transformers (Lessons 11-12)")
    print("  - Reinforcement Learning (Lesson 13)")
    print("  - Latent Dynamics (Lesson 14)")
    print("  - World Models (Lesson 15)")
    print()
    print("You're ready to explore state-of-the-art world models!")

# =============================================================================
# FINAL CHALLENGE: Complete Imagination Loop
# =============================================================================
@lesson.exercise("Challenge: Imagination Loop", points=10)
def challenge_imagination():
    """Implement a complete imagination loop."""
    print("Create an imagine_trajectory function that:")
    print("  - Takes world model, actor, initial state")
    print("  - Imagines H steps into the future")
    print("  - Returns features, actions, rewards")

    class SimpleWorldModel(nn.Module):
        def __init__(self, hidden_dim=32, latent_dim=8, action_dim=2):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim
            self.feature_dim = hidden_dim + latent_dim

            self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)
            self.prior = nn.Linear(hidden_dim, latent_dim * 2)
            self.reward_head = nn.Linear(self.feature_dim, 1)

        def init_state(self, batch):
            return (torch.zeros(batch, self.hidden_dim),
                    torch.zeros(batch, self.latent_dim))

        def imagine_step(self, h, z, action):
            rnn_in = torch.cat([z, action], dim=-1)
            h = self.rnn(rnn_in, h)
            prior_out = self.prior(h)
            mean, logstd = prior_out.chunk(2, dim=-1)
            std = F.softplus(logstd) + 0.1
            z = mean + std * torch.randn_like(std)
            return h, z

        def get_features(self, h, z):
            return torch.cat([h, z], dim=-1)

        def predict_reward(self, h, z):
            features = self.get_features(h, z)
            return self.reward_head(features).squeeze(-1)

    class SimpleActor(nn.Module):
        def __init__(self, feature_dim, action_dim):
            super().__init__()
            self.net = nn.Linear(feature_dim, action_dim)

        def sample(self, features):
            return torch.tanh(self.net(features))

    def imagine_trajectory(world_model, actor, init_h, init_z, horizon):
        """
        Imagine a trajectory.

        Returns:
            dict with 'features', 'actions', 'rewards' tensors
        """
        # TODO: Implement imagination loop
        h, z = init_h, init_z
        features_list = []
        actions_list = []
        rewards_list = []

        for _ in range(horizon):
            features = world_model.get_features(h, z)
            action = actor.sample(features)
            reward = world_model.predict_reward(h, z)

            features_list.append(features)
            actions_list.append(action)
            rewards_list.append(reward)

            h, z = world_model.imagine_step(h, z, action)

        return {
            'features': torch.stack(features_list, dim=1),
            'actions': torch.stack(actions_list, dim=1),
            'rewards': torch.stack(rewards_list, dim=1)
        }

    return imagine_trajectory, SimpleWorldModel, SimpleActor

class TestImagination(ExerciseTest):
    def run_tests(self, result):
        imagine_trajectory, SimpleWorldModel, SimpleActor = result

        world_model = SimpleWorldModel()
        actor = SimpleActor(world_model.feature_dim, action_dim=2)

        batch = 4
        horizon = 10
        h, z = world_model.init_state(batch)

        traj = imagine_trajectory(world_model, actor, h, z, horizon)

        self.test_equal(traj['features'].shape,
                       torch.Size([batch, horizon, world_model.feature_dim]),
                       "Features shape")
        self.test_equal(traj['actions'].shape,
                       torch.Size([batch, horizon, 2]),
                       "Actions shape")
        self.test_equal(traj['rewards'].shape,
                       torch.Size([batch, horizon]),
                       "Rewards shape")

# =============================================================================
# MAIN
# =============================================================================
def main():
    """Run the lesson."""
    if "--test" in sys.argv:
        results = []

        # Run all tests
        SimpleEncoder = exercise_encoder()
        results.append(("Encoder", TestEncoder().run_tests(SimpleEncoder)))

        MiniWorldModel = exercise_world_model()
        results.append(("World Model", TestWorldModel().run_tests(MiniWorldModel)))

        kl_loss = exercise_kl_loss()
        results.append(("KL Loss", TestKLLoss().run_tests(kl_loss)))

        compute_returns = exercise_lambda_returns()
        results.append(("Lambda Returns", TestLambdaReturns().run_tests(compute_returns)))

        imagination_result = challenge_imagination()
        results.append(("Imagination Loop", TestImagination().run_tests(imagination_result)))

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
        lesson.run_section(world_model_architecture)
        input("\nPress Enter to continue...")

        lesson.run_exercise(quiz_world_model)
        input("\nPress Enter to continue...")

        lesson.run_section(image_encoder_decoder)
        input("\nPress Enter to continue...")

        lesson.run_exercise(exercise_encoder, TestEncoder)
        input("\nPress Enter to continue...")

        lesson.run_section(rssm_dynamics)
        input("\nPress Enter to continue...")

        lesson.run_exercise(quiz_rssm)
        input("\nPress Enter to continue...")

        lesson.run_section(complete_world_model)
        input("\nPress Enter to continue...")

        lesson.run_exercise(exercise_world_model, TestWorldModel)
        input("\nPress Enter to continue...")

        lesson.run_section(world_model_loss_section)
        input("\nPress Enter to continue...")

        lesson.run_exercise(exercise_kl_loss, TestKLLoss)
        input("\nPress Enter to continue...")

        lesson.run_section(actor_critic_imagination)
        input("\nPress Enter to continue...")

        lesson.run_exercise(exercise_lambda_returns, TestLambdaReturns)
        input("\nPress Enter to continue...")

        lesson.run_section(dreamer_training)
        input("\nPress Enter to continue...")

        lesson.run_exercise(quiz_dreamer)
        input("\nPress Enter to continue...")

        lesson.run_section(world_model_summary)
        input("\nPress Enter to continue...")

        lesson.run_exercise(challenge_imagination, TestImagination)

        show_progress(lesson)

if __name__ == "__main__":
    main()
