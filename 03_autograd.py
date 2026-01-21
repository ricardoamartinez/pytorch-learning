#!/usr/bin/env python3
"""
================================================================================
LESSON 03: Automatic Differentiation (Autograd)
================================================================================
PyTorch's autograd system computes gradients automatically - the foundation
of backpropagation. This is what makes training neural networks possible!

Run: python 03_autograd.py
================================================================================
"""

import sys
sys.path.insert(0, '.')

import torch
from utils import (
    header, subheader, success, error, info, warning,
    ExerciseTest, LessonRunner, ask_question,
    test_equal, test_shape, test_true
)

# Create the lesson runner
lesson = LessonRunner(
    "03_autograd",
    "Master automatic differentiation - the engine behind neural network training."
)

# =============================================================================
# SECTION 1: The Computational Graph
# =============================================================================

@lesson.section("The Computational Graph")
def section_graph():
    """Understand how PyTorch builds and tracks operations."""

    print("PyTorch builds a COMPUTATIONAL GRAPH as you perform operations.\n")
    print("This graph tracks how each value was computed, enabling backprop.\n")

    print("-" * 50)
    print("ENABLING GRADIENT TRACKING:")
    print("-" * 50)

    x = torch.tensor([2.0, 3.0], requires_grad=True)
    print(f"\nx = torch.tensor([2.0, 3.0], requires_grad=True)")
    print(f"x.requires_grad = {x.requires_grad}")

    print("\n# Perform operations - PyTorch tracks them!")
    y = x ** 2
    z = y.sum()

    print(f"y = x ** 2 = {y}")
    print(f"z = y.sum() = {z}")

    print("\n# Each tensor knows how it was created:")
    print(f"y.grad_fn = {y.grad_fn}")
    print(f"z.grad_fn = {z.grad_fn}")

    print("\n" + "-" * 50)
    print("COMPUTATIONAL GRAPH VISUALIZATION:")
    print("-" * 50)
    print("""
    x = [2, 3] (leaf, requires_grad=True)
         |
         v
    y = x^2 = [4, 9] (grad_fn=PowBackward)
         |
         v
    z = sum(y) = 13 (grad_fn=SumBackward)
    """)

    print("-" * 50)
    print("KEY INSIGHT: The graph is built DYNAMICALLY during forward pass!")
    print("This is 'define-by-run' - unlike static graphs in TensorFlow v1.")
    print("-" * 50)


@lesson.exercise("Gradient Tracking Quiz", points=1)
def exercise_tracking():
    """Test understanding of gradient tracking."""

    answer = ask_question(
        "By default, tensors created with torch.tensor() have requires_grad=?",
        ["True", "False", "None", "Error"]
    )

    test = ExerciseTest("Gradient Tracking", hint="PyTorch is conservative - you must opt-in")
    test.check_true(answer == 1, "Correct! requires_grad=False by default for efficiency")
    return test.run()


# =============================================================================
# SECTION 2: Computing Gradients
# =============================================================================

@lesson.section("Computing Gradients with backward()")
def section_backward():
    """Learn to compute gradients via backpropagation."""

    print("Once we have a scalar output, we can compute gradients.\n")

    # Create a simple computation
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = x ** 2 + 3 * x + 1
    loss = y.sum()

    print(f"x = {x}")
    print(f"y = x² + 3x + 1 = {y}")
    print(f"loss = sum(y) = {loss}")

    print("\n" + "-" * 50)
    print("CALLING backward():")
    print("-" * 50)

    print("\nloss.backward()  # Compute gradients")
    loss.backward()

    print(f"\nGradients stored in x.grad:")
    print(f"x.grad = {x.grad}")

    print("\n" + "-" * 50)
    print("MATH VERIFICATION:")
    print("-" * 50)
    print("""
    y = x² + 3x + 1
    dy/dx = 2x + 3

    At x = [2, 3]:
    dy/dx = [2*2+3, 2*3+3] = [7, 9] ✓
    """)

    print("-" * 50)
    print("KEY INSIGHT: .backward() populates .grad for all leaf tensors!")
    print("Only leaf tensors (input tensors) get .grad by default.")
    print("-" * 50)


@lesson.exercise("Compute Gradients", points=1)
def exercise_backward():
    """Test understanding of backward()."""

    answer = ask_question(
        "What does .backward() require to work?",
        [
            "A scalar (0-D tensor) as the final result",
            "A vector of any size",
            "requires_grad=False",
            "A CUDA tensor"
        ]
    )

    test = ExerciseTest("backward()", hint="Think about what makes sense for a single loss value")
    test.check_true(answer == 0, "Correct! backward() needs a scalar - like a loss value")
    return test.run()


# =============================================================================
# SECTION 3: Gradient Accumulation
# =============================================================================

@lesson.section("Gradient Accumulation")
def section_accumulation():
    """Understand that gradients accumulate by default."""

    print("CRITICAL: Gradients ACCUMULATE - they don't reset automatically!\n")

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    print(f"x = {x}")
    print(f"x.grad before any backward: {x.grad}")

    # First backward
    y1 = (x ** 2).sum()
    y1.backward()
    print(f"\nAfter first backward (y = x²):")
    print(f"x.grad = {x.grad}  # Expected: 2x = [2, 4, 6]")

    # Second backward WITHOUT zeroing
    y2 = (x ** 2).sum()
    y2.backward()
    print(f"\nAfter second backward (same y = x²):")
    print(f"x.grad = {x.grad}  # Accumulated! Now [4, 8, 12]")

    print("\n" + "-" * 50)
    print("ZEROING GRADIENTS:")
    print("-" * 50)

    x.grad.zero_()  # In-place zero (note the underscore!)
    print(f"\nAfter x.grad.zero_():")
    print(f"x.grad = {x.grad}")

    # Now backward gives expected result
    y3 = (x ** 2).sum()
    y3.backward()
    print(f"\nAfter backward (y = x²):")
    print(f"x.grad = {x.grad}  # Fresh: [2, 4, 6]")

    print("\n" + "-" * 50)
    print("IN TRAINING LOOPS, ALWAYS:")
    print("-" * 50)
    print("""
    for batch in dataloader:
        optimizer.zero_grad()   # <-- ZERO GRADIENTS FIRST!
        loss = model(batch)
        loss.backward()
        optimizer.step()
    """)


@lesson.exercise("Gradient Accumulation Quiz", points=1)
def exercise_accumulation():
    """Test understanding of gradient accumulation."""

    answer = ask_question(
        "If you call backward() twice without zeroing, gradients will:",
        [
            "Be overwritten with new values",
            "Accumulate (add together)",
            "Raise an error",
            "Be averaged"
        ]
    )

    test = ExerciseTest("Accumulation", hint="PyTorch adds new gradients to existing ones")
    test.check_true(answer == 1, "Correct! Gradients accumulate - always zero them in training!")
    return test.run()


# =============================================================================
# SECTION 4: Disabling Gradient Computation
# =============================================================================

@lesson.section("Disabling Gradient Computation")
def section_no_grad():
    """Learn when and how to disable gradient tracking."""

    print("Sometimes you DON'T want to track gradients:\n")
    print("  - During evaluation/inference (faster, less memory)")
    print("  - When updating weights manually")
    print("  - When computing metrics\n")

    x = torch.tensor([1.0, 2.0], requires_grad=True)

    print("-" * 50)
    print("METHOD 1: torch.no_grad() context manager")
    print("-" * 50)

    print(f"\nx = {x}, requires_grad = {x.requires_grad}")

    with torch.no_grad():
        y = x * 2
        print(f"Inside no_grad: y = x * 2")
        print(f"y.requires_grad = {y.requires_grad}")

    y_outside = x * 2
    print(f"\nOutside no_grad: y = x * 2")
    print(f"y.requires_grad = {y_outside.requires_grad}")

    print("\n" + "-" * 50)
    print("METHOD 2: .detach()")
    print("-" * 50)

    y = x * 2
    y_detached = y.detach()
    print(f"\ny = x * 2")
    print(f"y.requires_grad = {y.requires_grad}")
    print(f"y.detach().requires_grad = {y_detached.requires_grad}")

    print("\n" + "-" * 50)
    print("METHOD 3: @torch.no_grad() decorator")
    print("-" * 50)

    print("""
    @torch.no_grad()
    def evaluate(model, data):
        # Everything here runs without gradients
        return model(data)
    """)

    print("-" * 50)
    print("KEY INSIGHT: Use no_grad() for inference - up to 50% faster!")
    print("-" * 50)


@lesson.exercise("No Grad Quiz", points=1)
def exercise_no_grad():
    """Test understanding of disabling gradients."""

    answer = ask_question(
        "Which is the recommended way to run evaluation without gradients?",
        [
            "Set requires_grad=False on all tensors",
            "Use torch.no_grad() context manager",
            "Delete the .grad attribute",
            "Use tensor.detach() on every tensor"
        ]
    )

    test = ExerciseTest("No Grad", hint="Context managers are clean and efficient")
    test.check_true(answer == 1, "Correct! torch.no_grad() is the cleanest approach")
    return test.run()


# =============================================================================
# SECTION 5: Manual Gradient Descent
# =============================================================================

@lesson.section("Manual Gradient Descent")
def section_manual_gd():
    """Implement gradient descent manually to understand the process."""

    print("Let's implement gradient descent from scratch!\n")
    print("Goal: Find x that minimizes f(x) = (x - 5)²")
    print("Answer should be x = 5 (where the parabola has minimum)\n")

    # Initialize
    x = torch.tensor([0.0], requires_grad=True)
    learning_rate = 0.1
    n_steps = 20

    print("-" * 50)
    print("GRADIENT DESCENT LOOP:")
    print("-" * 50)

    print(f"\nStarting x = {x.item():.4f}")
    print(f"Learning rate = {learning_rate}")
    print(f"Target: x = 5.0\n")

    for step in range(n_steps):
        # Forward pass: compute loss
        loss = (x - 5) ** 2

        # Backward pass: compute gradient
        loss.backward()

        # Update step (must be inside no_grad!)
        with torch.no_grad():
            x -= learning_rate * x.grad

        # Zero gradient for next iteration
        x.grad.zero_()

        if step % 5 == 0 or step == n_steps - 1:
            print(f"Step {step:2d}: x = {x.item():7.4f}, loss = {loss.item():.6f}")

    print("\n" + "-" * 50)
    print("THE GRADIENT DESCENT PATTERN:")
    print("-" * 50)
    print("""
    1. Forward:  loss = f(x)
    2. Backward: loss.backward()
    3. Update:   x = x - lr * x.grad
    4. Zero:     x.grad.zero_()
    5. Repeat!
    """)


@lesson.exercise("Gradient Descent Math", points=1)
def exercise_gd():
    """Test understanding of gradient descent."""

    print("f(x) = (x - 5)²")
    print("df/dx = 2(x - 5)")
    print("\nAt x = 3:")

    answer = ask_question(
        "What is the gradient df/dx at x = 3?",
        ["-4", "4", "-2", "2"]
    )

    test = ExerciseTest("Gradient Calculation", hint="df/dx = 2(x-5) = 2(3-5) = 2(-2)")
    test.check_true(answer == 0, "Correct! df/dx = 2(3-5) = 2(-2) = -4")
    return test.run()


# =============================================================================
# SECTION 6: Gradients with Neural Network Layers
# =============================================================================

@lesson.section("Gradients in Neural Networks")
def section_nn_grads():
    """See how gradients flow through neural network layers."""

    import torch.nn as nn

    print("Let's see gradients flow through a simple network.\n")

    # Create a simple linear layer
    linear = nn.Linear(3, 2)

    print(f"Layer: Linear(3, 2)")
    print(f"Weight shape: {linear.weight.shape}")
    print(f"Weight requires_grad: {linear.weight.requires_grad}")

    # Forward pass
    x = torch.randn(1, 3)
    y = linear(x)
    loss = y.sum()

    print(f"\nInput x shape: {x.shape}")
    print(f"Output y shape: {y.shape}")
    print(f"Loss (sum of outputs): {loss.item():.4f}")

    # Backward pass
    loss.backward()

    print("\n" + "-" * 50)
    print("GRADIENTS AFTER backward():")
    print("-" * 50)
    print(f"\nWeight gradient shape: {linear.weight.grad.shape}")
    print(f"Weight gradient:\n{linear.weight.grad}")
    print(f"\nBias gradient shape: {linear.bias.grad.shape}")
    print(f"Bias gradient: {linear.bias.grad}")

    print("\n" + "-" * 50)
    print("KEY INSIGHT: nn.Module parameters have requires_grad=True by default!")
    print("Backprop automatically computes gradients for ALL parameters.")
    print("-" * 50)


@lesson.exercise("Network Gradients Quiz", points=1)
def exercise_nn_grads():
    """Test understanding of gradients in networks."""

    answer = ask_question(
        "In a neural network, which parameters get gradients after backward()?",
        [
            "Only the first layer",
            "Only the last layer",
            "All layers with requires_grad=True",
            "Only layers explicitly marked"
        ]
    )

    test = ExerciseTest("Network Gradients", hint="Backprop flows through the entire graph")
    test.check_true(answer == 2, "Correct! All parameters with requires_grad=True get gradients")
    return test.run()


# =============================================================================
# SECTION 7: Common Pitfalls
# =============================================================================

@lesson.section("Common Autograd Pitfalls")
def section_pitfalls():
    """Learn about common mistakes with autograd."""

    print("Watch out for these common autograd mistakes!\n")

    print("-" * 50)
    print("PITFALL 1: In-place operations can break the graph")
    print("-" * 50)
    print("""
    x = torch.tensor([1.0], requires_grad=True)
    x += 1  # ERROR! In-place modification

    # Instead, do:
    x = x + 1  # Creates new tensor
    """)

    print("\n" + "-" * 50)
    print("PITFALL 2: Forgetting to zero gradients")
    print("-" * 50)
    print("""
    # WRONG:
    for batch in dataloader:
        loss = model(batch)
        loss.backward()  # Gradients accumulate!

    # RIGHT:
    for batch in dataloader:
        optimizer.zero_grad()  # Zero first!
        loss = model(batch)
        loss.backward()
    """)

    print("\n" + "-" * 50)
    print("PITFALL 3: backward() on non-scalar")
    print("-" * 50)
    print("""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x * 2  # y is a vector!
    y.backward()  # ERROR! Can only backward on scalar

    # Instead:
    y.sum().backward()  # OK
    # or
    y.backward(torch.ones_like(y))  # OK (provides gradient argument)
    """)

    print("\n" + "-" * 50)
    print("PITFALL 4: Modifying weights without no_grad()")
    print("-" * 50)
    print("""
    # WRONG:
    weights = weights - lr * weights.grad  # Builds new graph!

    # RIGHT:
    with torch.no_grad():
        weights -= lr * weights.grad
    """)


@lesson.exercise("Pitfall Quiz", points=1)
def exercise_pitfalls():
    """Test understanding of common pitfalls."""

    answer = ask_question(
        "Why must weight updates be inside torch.no_grad()?",
        [
            "For speed only",
            "To prevent building a graph of the update operation",
            "Because gradients can't be computed",
            "It's optional, just a convention"
        ]
    )

    test = ExerciseTest("Pitfalls", hint="We don't need gradients of the update itself!")
    test.check_true(answer == 1, "Correct! We don't want to track the update as an operation")
    return test.run()


# =============================================================================
# FINAL CODING EXERCISE
# =============================================================================

@lesson.exercise("Coding Challenge: Optimize a Quadratic", points=3)
def exercise_final():
    """Implement gradient descent to minimize a quadratic function."""

    print("Find the minimum of f(x, y) = (x-3)² + (y+2)²\n")
    print("This is a 2D quadratic with minimum at (3, -2)")

    # Initialize
    params = torch.tensor([0.0, 0.0], requires_grad=True)
    lr = 0.1
    n_steps = 50

    print(f"\nStarting at: ({params[0].item():.2f}, {params[1].item():.2f})")
    print(f"Target: (3.00, -2.00)\n")

    for step in range(n_steps):
        # Compute loss
        x, y = params[0], params[1]
        loss = (x - 3) ** 2 + (y + 2) ** 2

        # Backward
        loss.backward()

        # Update
        with torch.no_grad():
            params -= lr * params.grad

        # Zero gradients
        params.grad.zero_()

    final_x, final_y = params[0].item(), params[1].item()
    print(f"Final position: ({final_x:.4f}, {final_y:.4f})")

    # Verify
    test = ExerciseTest("Quadratic Optimization")

    test.check_true(
        abs(final_x - 3.0) < 0.01,
        f"x ≈ 3.0 (got {final_x:.4f})"
    )
    test.check_true(
        abs(final_y - (-2.0)) < 0.01,
        f"y ≈ -2.0 (got {final_y:.4f})"
    )

    final_loss = (final_x - 3) ** 2 + (final_y + 2) ** 2
    test.check_true(
        final_loss < 0.001,
        f"Loss ≈ 0 (got {final_loss:.6f})"
    )

    return test.run()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if "--test" in sys.argv:
        lesson.run()
    else:
        lesson.run_interactive()
