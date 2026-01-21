# =============================================================================
# LESSON 4: Building Neural Networks with nn.Module
# =============================================================================
# PyTorch provides nn.Module as the base class for all neural networks.
# This is where your ML knowledge meets code!

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# LAYERS - the building blocks
# -----------------------------------------------------------------------------

# Linear layer: y = xW^T + b
linear = nn.Linear(in_features=10, out_features=5)
print("Linear layer:")
print(f"  Weight shape: {linear.weight.shape}")  # [5, 10]
print(f"  Bias shape: {linear.bias.shape}")      # [5]

# Test it with dummy input
dummy_input = torch.randn(1, 10)  # Batch of 1, 10 features
output = linear(dummy_input)
print(f"  Input shape: {dummy_input.shape}")
print(f"  Output shape: {output.shape}")

# -----------------------------------------------------------------------------
# ACTIVATION FUNCTIONS
# -----------------------------------------------------------------------------
print("\nActivation functions:")

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

relu = nn.ReLU()
print(f"  ReLU({x.tolist()}) = {relu(x).tolist()}")

sigmoid = nn.Sigmoid()
print(f"  Sigmoid({x.tolist()}) = {[f'{v:.3f}' for v in sigmoid(x).tolist()]}")

tanh = nn.Tanh()
print(f"  Tanh({x.tolist()}) = {[f'{v:.3f}' for v in tanh(x).tolist()]}")

# -----------------------------------------------------------------------------
# BUILDING A NETWORK - using nn.Sequential
# -----------------------------------------------------------------------------
# Quick way to stack layers in order

simple_network = nn.Sequential(
    nn.Linear(784, 256),   # Input layer: 784 -> 256
    nn.ReLU(),             # Activation
    nn.Linear(256, 128),   # Hidden layer: 256 -> 128
    nn.ReLU(),             # Activation
    nn.Linear(128, 10),    # Output layer: 128 -> 10 (for 10 classes)
)

print("\nSimple Sequential Network:")
print(simple_network)

# Test forward pass
batch = torch.randn(32, 784)  # Batch of 32 images (28x28 = 784 pixels)
output = simple_network(batch)
print(f"Input shape: {batch.shape}")
print(f"Output shape: {output.shape}")

# -----------------------------------------------------------------------------
# BUILDING A NETWORK - using nn.Module (more flexible)
# -----------------------------------------------------------------------------

class MyNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()  # Always call parent's __init__

        # Define layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 20% dropout

    def forward(self, x):
        """Define how data flows through the network."""
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer3(x)
        return x

# Create an instance
model = MyNeuralNetwork(input_size=784, hidden_size=256, num_classes=10)
print("\nCustom Neural Network:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# -----------------------------------------------------------------------------
# TRAIN vs EVAL MODE
# -----------------------------------------------------------------------------
# Some layers (Dropout, BatchNorm) behave differently during training

model.train()  # Enable dropout
print(f"\nTraining mode: {model.training}")

model.eval()   # Disable dropout for inference
print(f"Eval mode: {model.training}")

# -----------------------------------------------------------------------------
# ACCESSING PARAMETERS
# -----------------------------------------------------------------------------
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

# -----------------------------------------------------------------------------
# EXERCISE: Create a network for binary classification
# - Input: 20 features
# - Hidden layers: 64 -> 32
# - Output: 1 (use Sigmoid at the end for probability)
# -----------------------------------------------------------------------------
