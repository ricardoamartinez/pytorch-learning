#!/usr/bin/env python3
"""
================================================================================
LESSON 05: The Complete Training Loop
================================================================================
Learn the essential pattern for training neural networks in PyTorch.
This ties together data, models, loss functions, and optimizers.

Run: python 05_training_loop.py
================================================================================
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import (
    header, subheader, success, error, info, warning,
    ExerciseTest, LessonRunner, ask_question,
    test_equal, test_shape, test_true
)

# Create the lesson runner
lesson = LessonRunner(
    "05_training_loop",
    "Master the training loop - the core pattern for all neural network training."
)

# =============================================================================
# SECTION 1: DataLoader and Batching
# =============================================================================

@lesson.section("DataLoader and Batching")
def section_dataloader():
    """Learn to efficiently load and batch data."""

    print("DataLoader handles batching, shuffling, and parallel loading.\n")

    # Create synthetic data
    torch.manual_seed(42)
    X = torch.randn(100, 10)  # 100 samples, 10 features
    y = torch.randint(0, 2, (100,))  # Binary labels

    print(f"Full dataset: X={X.shape}, y={y.shape}")

    print("\n" + "-" * 50)
    print("CREATING A DATALOADER:")
    print("-" * 50)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True
    )

    print(f"""
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
        dataset,
        batch_size=16,    # Samples per batch
        shuffle=True      # Randomize order each epoch
    )
    """)

    print(f"Number of batches: {len(dataloader)}")
    print(f"Samples per batch: 16 (last batch may be smaller)")

    print("\n" + "-" * 50)
    print("ITERATING THROUGH BATCHES:")
    print("-" * 50)

    for i, (batch_X, batch_y) in enumerate(dataloader):
        print(f"Batch {i}: X={batch_X.shape}, y={batch_y.shape}")
        if i >= 2:
            print("...")
            break

    print("\n" + "-" * 50)
    print("KEY PARAMETERS:")
    print("-" * 50)
    print("""
    DataLoader(
        dataset,
        batch_size=32,       # Typical: 16, 32, 64, 128
        shuffle=True,        # True for training, False for eval
        num_workers=4,       # Parallel data loading (0 on Windows)
        pin_memory=True,     # Faster GPU transfer
        drop_last=True       # Drop incomplete final batch
    )
    """)


@lesson.exercise("DataLoader Quiz", points=1)
def exercise_dataloader():
    """Test understanding of DataLoader."""

    answer = ask_question(
        "With 1000 samples and batch_size=64, how many complete batches?",
        ["15", "16", "15.625", "64"]
    )

    test = ExerciseTest("Batching", hint="1000 // 64 = ?")
    test.check_true(answer == 0, "Correct! 1000 // 64 = 15 complete batches (40 samples left over)")
    return test.run()


# =============================================================================
# SECTION 2: Loss Functions
# =============================================================================

@lesson.section("Loss Functions")
def section_loss():
    """Learn about common loss functions for different tasks."""

    print("Loss functions measure how wrong our predictions are.\n")

    print("-" * 50)
    print("CLASSIFICATION LOSSES:")
    print("-" * 50)

    # Binary Cross Entropy
    print("\n1. BCELoss / BCEWithLogitsLoss (Binary Classification)")
    pred = torch.tensor([0.8, 0.2, 0.6])
    target = torch.tensor([1.0, 0.0, 1.0])
    bce = nn.BCELoss()
    print(f"   Predictions: {pred.tolist()}")
    print(f"   Targets:     {target.tolist()}")
    print(f"   BCE Loss:    {bce(pred, target).item():.4f}")

    # Cross Entropy
    print("\n2. CrossEntropyLoss (Multi-class Classification)")
    logits = torch.tensor([[2.0, 0.5, 0.1],
                           [0.1, 2.0, 0.3]])  # 2 samples, 3 classes
    labels = torch.tensor([0, 1])  # Class indices
    ce = nn.CrossEntropyLoss()
    print(f"   Logits shape: {logits.shape} (batch, num_classes)")
    print(f"   Labels:       {labels.tolist()} (class indices)")
    print(f"   CE Loss:      {ce(logits, labels).item():.4f}")

    print("\n" + "-" * 50)
    print("REGRESSION LOSSES:")
    print("-" * 50)

    pred = torch.tensor([2.5, 0.0, 2.0])
    target = torch.tensor([3.0, -0.5, 2.0])

    # MSE
    mse = nn.MSELoss()
    print(f"\n3. MSELoss (Mean Squared Error)")
    print(f"   Predictions: {pred.tolist()}")
    print(f"   Targets:     {target.tolist()}")
    print(f"   MSE Loss:    {mse(pred, target).item():.4f}")

    # L1
    l1 = nn.L1Loss()
    print(f"\n4. L1Loss (Mean Absolute Error)")
    print(f"   L1 Loss:     {l1(pred, target).item():.4f}")

    print("\n" + "-" * 50)
    print("QUICK REFERENCE:")
    print("-" * 50)
    print("""
    Task                    Loss Function
    ─────────────────────────────────────────────
    Binary classification   BCEWithLogitsLoss
    Multi-class             CrossEntropyLoss
    Regression              MSELoss or L1Loss
    """)


@lesson.exercise("Loss Function Quiz", points=1)
def exercise_loss():
    """Test understanding of loss functions."""

    answer = ask_question(
        "For classifying images into 10 categories, which loss?",
        ["BCELoss", "MSELoss", "CrossEntropyLoss", "L1Loss"]
    )

    test = ExerciseTest("Loss Selection", hint="Multi-class classification")
    test.check_true(answer == 2, "Correct! CrossEntropyLoss for multi-class classification")
    return test.run()


# =============================================================================
# SECTION 3: Optimizers
# =============================================================================

@lesson.section("Optimizers")
def section_optimizers():
    """Learn about optimization algorithms."""

    print("Optimizers update model weights based on gradients.\n")

    # Create a simple model for demonstration
    model = nn.Linear(10, 2)

    print("-" * 50)
    print("CREATING AN OPTIMIZER:")
    print("-" * 50)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"""
    optimizer = torch.optim.Adam(
        model.parameters(),   # What to optimize
        lr=0.001              # Learning rate
    )
    """)

    print("-" * 50)
    print("COMMON OPTIMIZERS:")
    print("-" * 50)
    print("""
    1. SGD - Stochastic Gradient Descent
       torch.optim.SGD(params, lr=0.01, momentum=0.9)
       - Simple, widely used
       - Add momentum for faster convergence

    2. Adam - Adaptive Moment Estimation
       torch.optim.Adam(params, lr=0.001)
       - Adapts learning rate per parameter
       - Good default choice for most tasks

    3. AdamW - Adam with Weight Decay
       torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
       - Better regularization than Adam
       - Preferred for transformers

    4. RMSprop
       torch.optim.RMSprop(params, lr=0.01)
       - Good for RNNs
    """)

    print("-" * 50)
    print("OPTIMIZER METHODS:")
    print("-" * 50)
    print("""
    optimizer.zero_grad()   # Clear old gradients
    loss.backward()         # Compute new gradients
    optimizer.step()        # Update weights
    """)


@lesson.exercise("Optimizer Quiz", points=1)
def exercise_optimizers():
    """Test understanding of optimizers."""

    answer = ask_question(
        "What must you call BEFORE loss.backward() in each iteration?",
        ["optimizer.step()", "optimizer.zero_grad()", "model.eval()", "loss.item()"]
    )

    test = ExerciseTest("Optimizer Usage", hint="Gradients accumulate by default")
    test.check_true(answer == 1, "Correct! Always zero_grad() before backward()")
    return test.run()


# =============================================================================
# SECTION 4: The Training Loop Pattern
# =============================================================================

@lesson.section("The Training Loop Pattern")
def section_pattern():
    """Learn the essential training loop structure."""

    print("The training loop is THE core pattern in deep learning.\n")

    print("-" * 50)
    print("THE PATTERN:")
    print("-" * 50)
    print("""
    for epoch in range(num_epochs):
        model.train()                    # Training mode

        for batch_X, batch_y in train_loader:
            # 1. Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            # 2. Backward pass
            optimizer.zero_grad()        # Clear gradients
            loss.backward()              # Compute gradients
            optimizer.step()             # Update weights

        # 3. Evaluation (optional, per epoch)
        model.eval()
        with torch.no_grad():
            # Compute validation metrics
            ...
    """)

    print("-" * 50)
    print("STEP BY STEP:")
    print("-" * 50)
    print("""
    1. FORWARD:  predictions = model(inputs)
       - Pass data through the network

    2. LOSS:     loss = criterion(predictions, targets)
       - Measure prediction error

    3. ZERO:     optimizer.zero_grad()
       - Clear previous gradients

    4. BACKWARD: loss.backward()
       - Compute gradients via backprop

    5. UPDATE:   optimizer.step()
       - Adjust weights using gradients

    6. REPEAT!
    """)


@lesson.exercise("Training Loop Order", points=1)
def exercise_pattern():
    """Test understanding of training loop order."""

    answer = ask_question(
        "What is the correct order?",
        [
            "backward → zero_grad → step",
            "zero_grad → backward → step",
            "step → backward → zero_grad",
            "backward → step → zero_grad"
        ]
    )

    test = ExerciseTest("Loop Order", hint="Zero first, then backward, then step")
    test.check_true(answer == 1, "Correct! zero_grad → backward → step")
    return test.run()


# =============================================================================
# SECTION 5: Complete Training Example
# =============================================================================

@lesson.section("Complete Training Example")
def section_complete():
    """See a complete training loop in action."""

    print("Let's train a model on synthetic data!\n")

    # Setup
    torch.manual_seed(42)

    # Generate data: y = 1 if sum(x[:5]) > 0 else 0
    X = torch.randn(1000, 10)
    y = (X[:, :5].sum(dim=1) > 0).float().unsqueeze(1)

    train_X, test_X = X[:800], X[800:]
    train_y, test_y = y[:800], y[800:]

    train_loader = DataLoader(
        TensorDataset(train_X, train_y),
        batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_X, test_y),
        batch_size=32
    )

    print(f"Training samples: {len(train_X)}")
    print(f"Test samples:     {len(test_X)}")

    # Model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\n" + "-" * 50)
    print("TRAINING:")
    print("-" * 50)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            # Forward
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f}")

    print("\n" + "-" * 50)
    print("EVALUATION:")
    print("-" * 50)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            predicted = (predictions > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.2%}")


@lesson.exercise("Training Comprehension", points=1)
def exercise_complete():
    """Test understanding of the complete training process."""

    answer = ask_question(
        "Why use model.eval() and torch.no_grad() during testing?",
        [
            "To make the model more accurate",
            "To disable dropout and save memory/time",
            "To reset the model weights",
            "To enable backpropagation"
        ]
    )

    test = ExerciseTest("Evaluation Mode", hint="Think about dropout and gradient tracking")
    test.check_true(answer == 1, "Correct! eval() disables dropout, no_grad() saves memory")
    return test.run()


# =============================================================================
# SECTION 6: Saving and Loading Models
# =============================================================================

@lesson.section("Saving and Loading Models")
def section_save_load():
    """Learn to save and restore trained models."""

    print("Save your trained models to use them later!\n")

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )

    print("-" * 50)
    print("SAVING OPTIONS:")
    print("-" * 50)
    print("""
    1. Save state_dict (RECOMMENDED):
       torch.save(model.state_dict(), 'model_weights.pth')

    2. Save entire model (less portable):
       torch.save(model, 'model_full.pth')
    """)

    print("-" * 50)
    print("LOADING:")
    print("-" * 50)
    print("""
    # Create model with same architecture
    model = MyModel()

    # Load weights
    model.load_state_dict(torch.load('model_weights.pth'))

    # Set to evaluation mode
    model.eval()
    """)

    print("-" * 50)
    print("SAVING CHECKPOINTS (for resuming training):")
    print("-" * 50)
    print("""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, 'checkpoint.pth')

    # Loading checkpoint
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    """)


@lesson.exercise("Save/Load Quiz", points=1)
def exercise_save_load():
    """Test understanding of saving/loading models."""

    answer = ask_question(
        "What's the recommended way to save a PyTorch model?",
        [
            "torch.save(model, 'model.pth')",
            "torch.save(model.state_dict(), 'model.pth')",
            "model.save('model.pth')",
            "pickle.dump(model, file)"
        ]
    )

    test = ExerciseTest("Model Saving", hint="state_dict is more portable")
    test.check_true(answer == 1, "Correct! Save state_dict for portability")
    return test.run()


# =============================================================================
# SECTION 7: Training Tips
# =============================================================================

@lesson.section("Training Tips and Best Practices")
def section_tips():
    """Learn practical tips for better training."""

    print("-" * 50)
    print("TRAINING BEST PRACTICES:")
    print("-" * 50)
    print("""
    1. LEARNING RATE
       - Start with 1e-3 for Adam, 1e-2 for SGD
       - Use learning rate schedulers for better convergence

    2. BATCH SIZE
       - Larger = faster training, more memory
       - Smaller = better generalization
       - Common: 32, 64, 128, 256

    3. MONITORING
       - Track training AND validation loss
       - Watch for overfitting (train loss ↓, val loss ↑)

    4. EARLY STOPPING
       - Stop when validation loss stops improving
       - Prevents overfitting

    5. GRADIENT CLIPPING
       - torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
       - Prevents exploding gradients in RNNs

    6. DATA AUGMENTATION
       - Increases effective dataset size
       - Improves generalization

    7. REPRODUCIBILITY
       - Set seeds: torch.manual_seed(42)
       - Use deterministic algorithms
    """)

    print("-" * 50)
    print("LEARNING RATE SCHEDULING:")
    print("-" * 50)
    print("""
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1
    )

    for epoch in range(epochs):
        train(...)
        scheduler.step()  # Update learning rate
    """)


# =============================================================================
# FINAL CODING EXERCISE
# =============================================================================

@lesson.exercise("Coding Challenge: Train a Classifier", points=3)
def exercise_final():
    """Implement a complete training loop from scratch."""

    print("Train a classifier to achieve >85% accuracy!\n")

    # Setup
    torch.manual_seed(42)

    # Generate data
    X = torch.randn(500, 20)
    y = (X[:, :10].sum(dim=1) > 0).long()  # Binary classification

    train_X, test_X = X[:400], X[400:]
    train_y, test_y = y[:400], y[400:]

    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=32)

    # Model
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    print("Training...")
    for epoch in range(30):
        model.train()
        for batch_X, batch_y in train_loader:
            pred = model(batch_X)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%}")

    # Verify
    test = ExerciseTest("Training Challenge")
    test.check_true(accuracy > 0.85, f"Accuracy > 85% (got {accuracy:.2%})")
    test.check_true(model.training == False, "Model in eval mode after testing")

    return test.run()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if "--test" in sys.argv:
        lesson.run()
    else:
        lesson.run_interactive()
