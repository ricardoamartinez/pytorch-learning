# =============================================================================
# LESSON 3: Automatic Differentiation (Autograd)
# =============================================================================
# This is PyTorch's killer feature! It computes gradients automatically.
# You know backpropagation theory - here's how PyTorch implements it.

import torch

# -----------------------------------------------------------------------------
# TRACKING GRADIENTS
# -----------------------------------------------------------------------------
# requires_grad=True tells PyTorch to track operations for backprop

x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"x = {x}")
print(f"x.requires_grad = {x.requires_grad}")

# -----------------------------------------------------------------------------
# FORWARD PASS - building the computational graph
# -----------------------------------------------------------------------------
# Let's compute y = x^2 + 3x + 1 (a simple polynomial)

y = x**2 + 3*x + 1
print(f"\ny = x^2 + 3x + 1")
print(f"y = {y}")

# PyTorch builds a graph of operations as you compute!

# -----------------------------------------------------------------------------
# BACKWARD PASS - computing gradients
# -----------------------------------------------------------------------------
# For backprop, we need a scalar. Sum the outputs.
loss = y.sum()
print(f"\nloss = y.sum() = {loss}")

# Compute gradients via backpropagation
loss.backward()

# dy/dx = 2x + 3
# At x = [2, 3]: gradients should be [2*2+3, 2*3+3] = [7, 9]
print(f"\nGradients (dy/dx): {x.grad}")
print("Expected: [7, 9] because dy/dx = 2x + 3")

# -----------------------------------------------------------------------------
# GRADIENT DESCENT STEP (manual)
# -----------------------------------------------------------------------------
learning_rate = 0.1

# Create new tensor for demonstration
weights = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Forward pass
prediction = weights.sum()
target = 10.0
loss = (prediction - target) ** 2

# Backward pass
loss.backward()

print(f"\nWeights before update: {weights.data}")
print(f"Gradients: {weights.grad}")

# Update weights (gradient descent)
with torch.no_grad():  # Don't track this operation
    weights -= learning_rate * weights.grad

print(f"Weights after update: {weights.data}")

# -----------------------------------------------------------------------------
# IMPORTANT: Zero gradients before next iteration!
# -----------------------------------------------------------------------------
# Gradients accumulate by default. You must reset them.

weights.grad.zero_()  # The underscore means "in-place operation"
print(f"Gradients after zeroing: {weights.grad}")

# -----------------------------------------------------------------------------
# DETACHING FROM GRAPH
# -----------------------------------------------------------------------------
# Sometimes you want values without gradient tracking

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2

# Detach creates a new tensor that doesn't require grad
y_detached = y.detach()
print(f"\ny.requires_grad = {y.requires_grad}")
print(f"y_detached.requires_grad = {y_detached.requires_grad}")

# Or use torch.no_grad() context manager
with torch.no_grad():
    z = x * 3
    print(f"z.requires_grad = {z.requires_grad}")

# -----------------------------------------------------------------------------
# EXERCISE: Implement gradient descent to find minimum of f(x) = (x-5)^2
# Start with x=0, use learning_rate=0.1, run for 20 iterations
# The minimum should be at x=5
# -----------------------------------------------------------------------------
