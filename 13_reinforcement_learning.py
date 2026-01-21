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
import sys

from utils.lesson_utils import LessonRunner, ExerciseTest, ask_question, show_progress

lesson = LessonRunner(lesson_number=13, title="Reinforcement Learning Basics", total_points=80)

# =============================================================================
# SECTION 1: RL Fundamentals
# =============================================================================
@lesson.section("RL Fundamentals")
def rl_fundamentals():
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
    print("The RL loop: Agent observes state -> takes action -> receives reward -> repeat")
    print()
    print("Think of it like a game:")
    print("  - State: What you see on the screen")
    print("  - Action: Button you press")
    print("  - Reward: Points you gain/lose")
    print("  - Policy: Your strategy for playing")

# =============================================================================
# QUIZ 1: RL Concepts
# =============================================================================
@lesson.exercise("Quiz: RL Terminology", points=10)
def quiz_rl_concepts():
    print("Q: What does the policy π(a|s) represent?")
    answer = ask_question([
        "A) The reward at a state",
        "B) The probability distribution over actions given a state",
        "C) The next state prediction",
        "D) The discount factor"
    ])

    if answer == "B":
        print("Correct! The policy outputs a probability distribution over actions.")
        return True
    else:
        print("Not quite. The policy π(a|s) outputs action probabilities given the state.")
        return False

# =============================================================================
# SECTION 2: Simple Environment
# =============================================================================
@lesson.section("Simple Environment")
def simple_environment():
    """
    Let's create a simple environment to understand RL mechanics.
    State: position, velocity
    Action: 0 (left) or 1 (right)
    Goal: Keep position near 0
    """
    class SimpleEnv:
        """CartPole-like simplified environment."""
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

            # Reward: closer to 0 is better
            reward = -abs(pos)

            # Done if too far
            done = abs(pos) > 2.0

            self.state = np.array([pos, vel])
            return self.state.copy(), reward, done

    env = SimpleEnv()
    state = env.reset()
    print(f"Initial state: {state}")

    next_state, reward, done = env.step(1)  # Push right
    print(f"After action 1: state={next_state}, reward={reward:.3f}, done={done}")

    return SimpleEnv

# =============================================================================
# SECTION 3: Policy Network
# =============================================================================
@lesson.section("Policy Network")
def policy_network():
    """
    POLICY NETWORK:

    A neural network that maps states to action probabilities.

        state -> Network -> softmax -> P(action|state)

    Training: Increase probability of actions that led to high rewards.
    """
    class PolicyNetwork(nn.Module):
        """Simple policy network."""
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

    # Test with random state
    state = np.array([0.1, 0.05])
    action, log_prob = policy.get_action(state)
    print(f"\nState: {state}")
    print(f"Sampled action: {action}, Log prob: {log_prob.item():.4f}")

    return PolicyNetwork

# =============================================================================
# EXERCISE 1: Build a Policy Network
# =============================================================================
@lesson.exercise("Build Policy Network", points=10)
def exercise_policy_network():
    """Create a policy network with specific architecture."""
    print("Create a PolicyNet class with:")
    print("  - Input: state_dim (3)")
    print("  - Hidden: Two layers with 32 units each and ReLU")
    print("  - Output: action_dim (4) probabilities via softmax")

    class PolicyNet(nn.Module):
        # TODO: Implement the policy network
        def __init__(self, state_dim=3, action_dim=4, hidden=32):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return F.softmax(self.fc3(x), dim=-1)

    return PolicyNet

class TestPolicyNet(ExerciseTest):
    def run_tests(self, PolicyNet):
        model = PolicyNet()
        x = torch.randn(2, 3)
        out = model(x)

        self.test_equal(out.shape, torch.Size([2, 4]), "Output shape")
        self.test_true(torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-5),
                      "Output sums to 1 (valid probability)")
        self.test_true((out >= 0).all().item(), "All probabilities non-negative")

# =============================================================================
# SECTION 4: Value Network (Critic)
# =============================================================================
@lesson.section("Value Network")
def value_network():
    """
    VALUE NETWORK:

    Estimates the expected cumulative reward from a state.

        V(s) = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t = s]

    Used to:
    - Reduce variance in policy gradients
    - Compute advantage A(s,a) = Q(s,a) - V(s)
    """
    class ValueNetwork(nn.Module):
        """Value function estimator."""
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

    state = torch.randn(4, 2)
    values = value_net(state)
    print(f"\nBatch of states: {state.shape}")
    print(f"Predicted values: {values.squeeze().tolist()}")

    return ValueNetwork

# =============================================================================
# QUIZ 2: Value Functions
# =============================================================================
@lesson.exercise("Quiz: Value vs Q-Value", points=10)
def quiz_value_functions():
    print("Q: What is the difference between V(s) and Q(s,a)?")
    answer = ask_question([
        "A) V(s) considers a specific action, Q(s,a) averages over actions",
        "B) Q(s,a) considers a specific action, V(s) averages over actions",
        "C) They are the same thing",
        "D) V(s) is for continuous actions, Q(s,a) is for discrete"
    ])

    if answer == "B":
        print("Correct! V(s) = E_a[Q(s,a)] under the policy.")
        return True
    else:
        print("Not quite. Q(s,a) is for a specific action, V(s) averages over policy.")
        return False

# =============================================================================
# SECTION 5: REINFORCE Algorithm
# =============================================================================
@lesson.section("REINFORCE Algorithm")
def reinforce_algorithm():
    """
    REINFORCE (Monte Carlo Policy Gradient):

    1. Collect trajectory: (s_0, a_0, r_0), (s_1, a_1, r_1), ...
    2. Compute returns: G_t = Σ γ^k r_{t+k}
    3. Update policy: θ += α * Σ ∇log π(a_t|s_t) * G_t

    Intuition: Increase probability of actions that led to high returns.
    """
    def compute_returns(rewards, gamma=0.99):
        """Compute discounted returns."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns

    # Example
    rewards = [1, 0, -1, 2]
    returns = compute_returns(rewards, gamma=0.99)
    print(f"Rewards: {rewards}")
    print(f"Returns (γ=0.99): {[f'{r:.3f}' for r in returns]}")

    # Show the formula
    print("\nPolicy gradient: ∇J(θ) = E[∇log π(a|s) * G]")
    print("Update: θ += α * ∇log π(a|s) * G")

    return compute_returns

# =============================================================================
# EXERCISE 2: Compute Returns
# =============================================================================
@lesson.exercise("Implement Return Computation", points=10)
def exercise_compute_returns():
    """Implement discounted return computation."""
    print("Implement compute_discounted_returns that:")
    print("  - Takes a list of rewards and discount factor gamma")
    print("  - Returns list of discounted returns G_t = Σ γ^k * r_{t+k}")

    def compute_discounted_returns(rewards, gamma=0.99):
        # TODO: Compute returns from right to left
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns

    return compute_discounted_returns

class TestReturns(ExerciseTest):
    def run_tests(self, compute_returns):
        # Test with gamma=1 (no discounting)
        rewards = [1, 1, 1]
        returns = compute_returns(rewards, gamma=1.0)
        self.test_equal(returns, [3, 2, 1], "No discounting case")

        # Test with gamma=0 (only immediate)
        returns_0 = compute_returns(rewards, gamma=0.0)
        self.test_equal(returns_0, [1, 1, 1], "No future reward case")

        # Test with gamma=0.5
        returns_half = compute_returns([1, 2], gamma=0.5)
        expected = [1 + 0.5*2, 2]  # [2.0, 2]
        self.test_true(abs(returns_half[0] - 2.0) < 1e-5, "Gamma=0.5 first return")

# =============================================================================
# SECTION 6: Actor-Critic
# =============================================================================
@lesson.section("Actor-Critic Algorithm")
def actor_critic():
    """
    ACTOR-CRITIC:
    - Actor: Policy network (chooses actions)
    - Critic: Value network (evaluates states)

    Advantage: A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s)

    Update:
    - Critic: Minimize TD error (r + γV(s') - V(s))²
    - Actor: θ += α * ∇log π(a|s) * A(s,a)

    Benefits over REINFORCE:
    - Lower variance (uses TD learning)
    - Can learn online (not just episodic)
    """
    class ActorCritic(nn.Module):
        """Combined actor-critic with shared features."""
        def __init__(self, state_dim, action_dim, hidden_dim=64):
            super().__init__()

            # Shared feature extractor
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
            )

            # Actor head (policy)
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

            # Critic head (value)
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

    ac = ActorCritic(state_dim=4, action_dim=2)
    state = torch.randn(1, 4)
    probs, value = ac(state)

    print("Actor-Critic architecture:")
    print(f"  State {state.shape} -> Shared features -> Actor + Critic")
    print(f"  Action probs: {probs.detach().numpy().round(3)}")
    print(f"  State value: {value.item():.4f}")

    return ActorCritic

# =============================================================================
# EXERCISE 3: Actor-Critic Network
# =============================================================================
@lesson.exercise("Build Actor-Critic", points=10)
def exercise_actor_critic():
    """Build a combined actor-critic network."""
    print("Create an ActorCriticNet with:")
    print("  - Shared: Linear(state_dim, 64) + ReLU")
    print("  - Actor: Linear(64, action_dim) + softmax")
    print("  - Critic: Linear(64, 1)")

    class ActorCriticNet(nn.Module):
        def __init__(self, state_dim=4, action_dim=2):
            super().__init__()
            # TODO: Implement shared + actor + critic heads
            self.shared = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU()
            )
            self.actor_head = nn.Linear(64, action_dim)
            self.critic_head = nn.Linear(64, 1)

        def forward(self, x):
            features = self.shared(x)
            action_probs = F.softmax(self.actor_head(features), dim=-1)
            value = self.critic_head(features)
            return action_probs, value

    return ActorCriticNet

class TestActorCritic(ExerciseTest):
    def run_tests(self, ActorCriticNet):
        model = ActorCriticNet()
        x = torch.randn(3, 4)
        probs, values = model(x)

        self.test_equal(probs.shape, torch.Size([3, 2]), "Action probs shape")
        self.test_equal(values.shape, torch.Size([3, 1]), "Value shape")
        self.test_true(torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5),
                      "Action probs sum to 1")

# =============================================================================
# SECTION 7: Deep Q-Network (DQN)
# =============================================================================
@lesson.section("Deep Q-Network (DQN)")
def dqn_section():
    """
    DQN (Deep Q-Network):
    - Learn Q(s,a) directly
    - Action = argmax_a Q(s,a)
    - Train with TD: Q(s,a) -> r + γ max_a' Q(s',a')

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
        """Experience replay buffer."""
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
            batch = [self.buffer[i] for i in indices]
            states, actions, rewards, next_states, dones = zip(*batch)
            return (torch.FloatTensor(np.array(states)),
                    torch.LongTensor(actions),
                    torch.FloatTensor(rewards),
                    torch.FloatTensor(np.array(next_states)),
                    torch.FloatTensor(dones))

        def __len__(self):
            return len(self.buffer)

    dqn = DQN(state_dim=4, action_dim=2)
    print("DQN outputs Q-values for each action:")
    state = torch.randn(1, 4)
    q_values = dqn(state)
    print(f"  State: {state.shape}")
    print(f"  Q-values: {q_values.detach().numpy().round(3)}")
    print(f"  Best action: {q_values.argmax().item()}")

    return DQN, ReplayBuffer

# =============================================================================
# QUIZ 3: DQN Concepts
# =============================================================================
@lesson.exercise("Quiz: Experience Replay", points=10)
def quiz_experience_replay():
    print("Q: Why does DQN use experience replay?")
    answer = ask_question([
        "A) To speed up training",
        "B) To break correlation between consecutive samples",
        "C) To increase the Q-values",
        "D) To reduce the network size"
    ])

    if answer == "B":
        print("Correct! Replay breaks temporal correlation, stabilizing training.")
        return True
    else:
        print("The key reason is to break correlation between consecutive samples.")
        return False

# =============================================================================
# SECTION 8: Model-Based RL
# =============================================================================
@lesson.section("Model-Based RL")
def model_based_rl():
    """
    MODEL-FREE VS MODEL-BASED RL:

    MODEL-FREE:
        - Learn policy/value directly from experience
        - No explicit model of environment
        - Examples: REINFORCE, A2C, PPO, SAC, DQN
        - Pro: Simple, no model errors
        - Con: Sample inefficient

    MODEL-BASED (World Models):
        - Learn model of environment: s' = f(s, a)
        - Use model for planning/imagination
        - Examples: Dreamer, MuZero, IRIS
        - Pro: Sample efficient
        - Con: Model errors can compound

    WORLD MODEL IN RL:
        1. Collect real experience
        2. Train world model on experience
        3. Imagine trajectories using world model
        4. Train policy on imagined trajectories
        5. Repeat
    """
    class SimpleWorldModel(nn.Module):
        """Predicts next state and reward: s', r = f(s, a)"""
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

        def forward(self, state, action_onehot):
            """Predict next state and reward."""
            x = torch.cat([state, action_onehot], dim=-1)
            next_state = self.state_predictor(x)
            reward = self.reward_predictor(x)
            return next_state, reward

    world_model = SimpleWorldModel(state_dim=4, action_dim=2)

    # Test prediction
    state = torch.randn(2, 4)
    action = F.one_hot(torch.tensor([0, 1]), num_classes=2).float()
    pred_next, pred_reward = world_model(state, action)

    print("World Model predicts environment dynamics:")
    print(f"  State: {state.shape}")
    print(f"  Action (one-hot): {action.shape}")
    print(f"  Predicted next state: {pred_next.shape}")
    print(f"  Predicted reward: {pred_reward.shape}")

    return SimpleWorldModel

# =============================================================================
# EXERCISE 4: World Model
# =============================================================================
@lesson.exercise("Build Simple World Model", points=10)
def exercise_world_model():
    """Build a world model that predicts transitions."""
    print("Create a WorldModel that:")
    print("  - Takes state (dim=4) and one-hot action (dim=2)")
    print("  - Predicts next_state (dim=4) and reward (dim=1)")

    class WorldModel(nn.Module):
        def __init__(self, state_dim=4, action_dim=2, hidden=64):
            super().__init__()
            # TODO: Implement world model
            input_dim = state_dim + action_dim
            self.dynamics = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, state_dim)
            )
            self.reward_head = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )

        def forward(self, state, action_onehot):
            x = torch.cat([state, action_onehot], dim=-1)
            next_state = self.dynamics(x)
            reward = self.reward_head(x)
            return next_state, reward

    return WorldModel

class TestWorldModel(ExerciseTest):
    def run_tests(self, WorldModel):
        model = WorldModel()
        state = torch.randn(3, 4)
        action = F.one_hot(torch.randint(0, 2, (3,)), num_classes=2).float()

        next_state, reward = model(state, action)

        self.test_equal(next_state.shape, torch.Size([3, 4]), "Next state shape")
        self.test_equal(reward.shape, torch.Size([3, 1]), "Reward shape")

# =============================================================================
# FINAL CHALLENGE: RL Training Loop
# =============================================================================
@lesson.exercise("Challenge: REINFORCE Training Loop", points=10)
def challenge_reinforce():
    """Implement a complete REINFORCE training step."""
    print("Implement a training function that:")
    print("  1. Collects a trajectory (states, actions, rewards)")
    print("  2. Computes discounted returns")
    print("  3. Computes policy gradient loss")
    print("  4. Returns the loss tensor")

    class SimplePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )

        def forward(self, x):
            return F.softmax(self.fc(x), dim=-1)

    def compute_policy_loss(policy, states, actions, returns):
        """
        Compute REINFORCE policy gradient loss.

        Args:
            policy: Policy network
            states: Tensor of states [T, state_dim]
            actions: Tensor of actions taken [T]
            returns: Tensor of returns [T]

        Returns:
            loss: Policy gradient loss (scalar tensor)
        """
        # TODO: Compute -sum(log_prob * return)
        probs = policy(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)

        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient: maximize E[log_prob * return]
        # Minimize negative of this
        loss = -(log_probs * returns).sum()
        return loss

    return compute_policy_loss, SimplePolicy

class TestReinforce(ExerciseTest):
    def run_tests(self, result):
        compute_policy_loss, SimplePolicy = result

        policy = SimplePolicy()
        states = torch.randn(5, 2)
        actions = torch.randint(0, 2, (5,))
        returns = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.0])

        loss = compute_policy_loss(policy, states, actions, returns)

        self.test_true(loss.dim() == 0, "Loss is scalar")
        self.test_true(loss.requires_grad, "Loss has gradient")

        # Check gradient flows
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in policy.parameters())
        self.test_true(has_grad, "Gradients flow through policy")

# =============================================================================
# MAIN
# =============================================================================
def main():
    """Run the lesson."""
    if "--test" in sys.argv:
        # Run all exercises in test mode
        results = []

        # Test policy network
        PolicyNet = exercise_policy_network()
        results.append(("Policy Network", TestPolicyNet().run_tests(PolicyNet)))

        # Test returns
        compute_returns = exercise_compute_returns()
        results.append(("Compute Returns", TestReturns().run_tests(compute_returns)))

        # Test actor-critic
        ActorCriticNet = exercise_actor_critic()
        results.append(("Actor-Critic", TestActorCritic().run_tests(ActorCriticNet)))

        # Test world model
        WorldModel = exercise_world_model()
        results.append(("World Model", TestWorldModel().run_tests(WorldModel)))

        # Test REINFORCE
        reinforce_result = challenge_reinforce()
        results.append(("REINFORCE", TestReinforce().run_tests(reinforce_result)))

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
        lesson.run_section(rl_fundamentals)
        input("\nPress Enter to continue...")

        lesson.run_exercise(quiz_rl_concepts)
        input("\nPress Enter to continue...")

        lesson.run_section(simple_environment)
        input("\nPress Enter to continue...")

        lesson.run_section(policy_network)
        input("\nPress Enter to continue...")

        lesson.run_exercise(exercise_policy_network, TestPolicyNet)
        input("\nPress Enter to continue...")

        lesson.run_section(value_network)
        input("\nPress Enter to continue...")

        lesson.run_exercise(quiz_value_functions)
        input("\nPress Enter to continue...")

        lesson.run_section(reinforce_algorithm)
        input("\nPress Enter to continue...")

        lesson.run_exercise(exercise_compute_returns, TestReturns)
        input("\nPress Enter to continue...")

        lesson.run_section(actor_critic)
        input("\nPress Enter to continue...")

        lesson.run_exercise(exercise_actor_critic, TestActorCritic)
        input("\nPress Enter to continue...")

        lesson.run_section(dqn_section)
        input("\nPress Enter to continue...")

        lesson.run_exercise(quiz_experience_replay)
        input("\nPress Enter to continue...")

        lesson.run_section(model_based_rl)
        input("\nPress Enter to continue...")

        lesson.run_exercise(exercise_world_model, TestWorldModel)
        input("\nPress Enter to continue...")

        lesson.run_exercise(challenge_reinforce, TestReinforce)

        show_progress(lesson)

if __name__ == "__main__":
    main()
