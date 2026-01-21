# =============================================================================
# LESSON 13: Reinforcement Learning Basics
# =============================================================================
# World models are often used in RL for planning and imagination.
# Let's understand the fundamentals of RL with PyTorch.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# -----------------------------------------------------------------------------
# THE CONCEPT: Reinforcement Learning
# -----------------------------------------------------------------------------
"""
RL FRAMEWORK:
    Agent <-> Environment

At each timestep t:
    1. Agent observes state s_t
    2. Agent takes action a_t ~ π(a|s)
    3. Environment returns reward r_t and next state s_{t+1}
    4. Goal: Maximize cumulative reward Σ γ^t r_t

KEY TERMS:
- Policy π(a|s): Probability of action given state
- Value V(s): Expected cumulative reward from state s
- Q-value Q(s,a): Expected cumulative reward from state s, action a
- Reward r: Immediate feedback signal
- Discount γ: How much to value future rewards (0.99 typical)

WHY WORLD MODELS FOR RL?
- Model-free: Learn directly from experience (slow, sample inefficient)
- Model-based: Learn a world model, then plan using it (faster, more efficient)
"""

# -----------------------------------------------------------------------------
# STEP 1: Simple Environment (CartPole-like)
# -----------------------------------------------------------------------------
print("=" * 60)
print("SIMPLE ENVIRONMENT")
print("=" * 60)

class SimpleEnv:
    """
    Simplified environment for demonstration.
    State: position, velocity
    Action: 0 (left) or 1 (right)
    Goal: Keep position near 0
    """
    def __init__(self):
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.array([np.random.uniform(-0.5, 0.5),
                               np.random.uniform(-0.1, 0.1)])
        return self.state.copy()

    def step(self, action):
        pos, vel = self.state

        # Apply action (force)
        force = 0.1 if action == 1 else -0.1
        vel = vel + force
        pos = pos + vel

        # Compute reward (closer to 0 is better)
        reward = -abs(pos)

        # Check if done
        done = abs(pos) > 2.0

        self.state = np.array([pos, vel])
        return self.state.copy(), reward, done

env = SimpleEnv()
state = env.reset()
print(f"Initial state: {state}")
next_state, reward, done = env.step(1)
print(f"After action 1: state={next_state}, reward={reward:.3f}, done={done}")

# -----------------------------------------------------------------------------
# STEP 2: Policy Network
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("POLICY NETWORK")
print("=" * 60)

class PolicyNetwork(nn.Module):
    """
    Simple policy network.
    Input: state
    Output: probability distribution over actions
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        logits = self.network(state)
        return F.softmax(logits, dim=-1)

    def get_action(self, state):
        """Sample action from policy."""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

policy = PolicyNetwork(state_dim=2, action_dim=2)
print(policy)

# Test
state = env.reset()
action, log_prob = policy.get_action(state)
print(f"\nState: {state}")
print(f"Action: {action}, Log prob: {log_prob.item():.4f}")

# -----------------------------------------------------------------------------
# STEP 3: Value Network (Critic)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("VALUE NETWORK")
print("=" * 60)

class ValueNetwork(nn.Module):
    """
    Value function estimator.
    Input: state
    Output: estimated value V(s)
    """
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state)

value_net = ValueNetwork(state_dim=2)
print(value_net)

state_tensor = torch.FloatTensor(state).unsqueeze(0)
value = value_net(state_tensor)
print(f"\nState: {state}")
print(f"Estimated value: {value.item():.4f}")

# -----------------------------------------------------------------------------
# STEP 4: REINFORCE Algorithm (Policy Gradient)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("REINFORCE ALGORITHM")
print("=" * 60)

"""
REINFORCE (Monte Carlo Policy Gradient):

1. Collect trajectory: (s_0, a_0, r_0), (s_1, a_1, r_1), ...
2. Compute returns: G_t = Σ γ^k r_{t+k}
3. Update policy: θ += α * Σ ∇log π(a_t|s_t) * G_t

Intuition: Increase probability of actions that led to high returns
"""

def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def reinforce_episode(env, policy, optimizer, gamma=0.99):
    """Run one episode of REINFORCE."""
    log_probs = []
    rewards = []

    state = env.reset()
    done = False

    while not done:
        action, log_prob = policy.get_action(state)
        next_state, reward, done = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)

        state = next_state

        if len(rewards) > 200:  # Max episode length
            break

    # Compute returns
    returns = compute_returns(rewards, gamma)
    returns = torch.FloatTensor(returns)

    # Normalize returns (variance reduction)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Policy gradient loss
    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss -= log_prob * G  # Negative because we do gradient descent

    # Update policy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return sum(rewards)

# Train for a few episodes
policy = PolicyNetwork(state_dim=2, action_dim=2)
optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

print("Training REINFORCE...")
for episode in range(100):
    total_reward = reinforce_episode(env, policy, optimizer)
    if (episode + 1) % 20 == 0:
        print(f"Episode {episode+1}: Total reward = {total_reward:.2f}")

# -----------------------------------------------------------------------------
# STEP 5: Actor-Critic Algorithm
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ACTOR-CRITIC ALGORITHM")
print("=" * 60)

"""
ACTOR-CRITIC:
- Actor: Policy network (chooses actions)
- Critic: Value network (evaluates states)

Advantage: A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s)

Update:
- Critic: Minimize TD error (r + γV(s') - V(s))²
- Actor: θ += α * ∇log π(a|s) * A(s,a)

Benefits:
- Lower variance than REINFORCE
- Can learn online (not just episodic)
"""

class ActorCritic(nn.Module):
    """Combined actor-critic network with shared features."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        features = self.shared(state)
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs, value = self(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

def actor_critic_step(env, model, optimizer, gamma=0.99):
    """One step of actor-critic learning."""
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, log_prob, value = model.get_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward

        # Get next value
        with torch.no_grad():
            _, next_value = model(torch.FloatTensor(next_state).unsqueeze(0))

        # TD target and advantage
        if done:
            td_target = torch.tensor([[reward]])
        else:
            td_target = reward + gamma * next_value

        advantage = td_target - value

        # Losses
        actor_loss = -log_prob * advantage.detach()
        critic_loss = F.mse_loss(value, td_target.detach())
        loss = actor_loss + 0.5 * critic_loss

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        if total_reward < -100:  # Early stop if doing badly
            break

    return total_reward

# Train actor-critic
ac_model = ActorCritic(state_dim=2, action_dim=2)
ac_optimizer = torch.optim.Adam(ac_model.parameters(), lr=0.001)

print("Training Actor-Critic...")
for episode in range(100):
    total_reward = actor_critic_step(env, ac_model, ac_optimizer)
    if (episode + 1) % 20 == 0:
        print(f"Episode {episode+1}: Total reward = {total_reward:.2f}")

# -----------------------------------------------------------------------------
# STEP 6: Q-Learning and DQN
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DEEP Q-NETWORK (DQN)")
print("=" * 60)

"""
DQN (Deep Q-Network):
- Learn Q(s,a) directly
- Action = argmax_a Q(s,a)
- Train with TD learning: Q(s,a) -> r + γ max_a' Q(s',a')

Key tricks:
- Experience replay: Store transitions, sample randomly
- Target network: Separate network for stable targets
"""

class DQN(nn.Module):
    """Deep Q-Network."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.network(state)  # Q-values for each action

class ReplayBuffer:
    """Simple replay buffer for experience replay."""
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buffer)

dqn = DQN(state_dim=2, action_dim=2)
target_dqn = DQN(state_dim=2, action_dim=2)
target_dqn.load_state_dict(dqn.state_dict())  # Copy weights

print(dqn)
print("\nDQN uses experience replay and target network for stability")

# -----------------------------------------------------------------------------
# STEP 7: Model-Based RL Concept
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL-BASED REINFORCEMENT LEARNING")
print("=" * 60)

"""
MODEL-FREE VS MODEL-BASED RL:

MODEL-FREE:
    - Learn policy/value directly from experience
    - No explicit model of environment
    - Examples: REINFORCE, A2C, PPO, SAC, DQN
    - Pro: Simple, no model errors
    - Con: Sample inefficient (need lots of experience)

MODEL-BASED (with World Models):
    - Learn a model of environment dynamics: s' = f(s, a)
    - Use model for planning/imagination
    - Examples: Dreamer, MuZero, IRIS
    - Pro: Sample efficient (learn from imagined experience)
    - Con: Model errors can compound

WORLD MODEL IN RL:
    1. Collect real experience
    2. Train world model on experience
    3. Imagine trajectories using world model
    4. Train policy on imagined trajectories
    5. Repeat

This is the "Dreamer" approach we'll implement later!
"""

class SimpleWorldModel(nn.Module):
    """
    A basic world model that predicts next state and reward.
    s', r = f(s, a)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        self.state_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.reward_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        """Predict next state and reward."""
        # One-hot encode action
        action_onehot = F.one_hot(action, num_classes=2).float()
        x = torch.cat([state, action_onehot], dim=-1)

        next_state = self.state_predictor(x)
        reward = self.reward_predictor(x)

        return next_state, reward

world_model = SimpleWorldModel(state_dim=2, action_dim=2)
print(world_model)

# Test world model
state = torch.randn(4, 2)
action = torch.randint(0, 2, (4,))
pred_next_state, pred_reward = world_model(state, action)
print(f"\nState: {state.shape}")
print(f"Predicted next state: {pred_next_state.shape}")
print(f"Predicted reward: {pred_reward.shape}")

# -----------------------------------------------------------------------------
# KEY CONCEPTS FOR WORLD MODELS
# -----------------------------------------------------------------------------
"""
HOW WORLD MODELS FIT INTO RL:

1. IMAGINATION (Dreamer, MuZero):
   - Imagine many trajectories without real env
   - Train policy on imagined data
   - Massive sample efficiency gains

2. PLANNING (MPC, MCTS):
   - Use world model to evaluate action sequences
   - Select best action based on predicted outcomes
   - AlphaGo/MuZero use MCTS with learned models

3. REPRESENTATION LEARNING:
   - World model learns useful state representations
   - Latent space captures dynamics-relevant features

4. DREAMER ARCHITECTURE:
   - Observation -> Encoder -> Latent z
   - Latent z, action a -> RSSM -> Next latent z'
   - Train actor-critic in imagination
   - Only interact with real env to collect data

NEXT: We'll implement latent dynamics models (the core of world models)!
"""

# -----------------------------------------------------------------------------
# EXERCISE:
# 1. Implement PPO (Proximal Policy Optimization)
# 2. Train DQN with experience replay on SimpleEnv
# 3. Use world_model to generate imaginary rollouts
# 4. Compare sample efficiency: model-free vs model-based
# 5. Install gymnasium and try CartPole-v1
# -----------------------------------------------------------------------------
