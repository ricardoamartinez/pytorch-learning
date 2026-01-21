# =============================================================================
# LESSON 5: The Complete Training Loop
# =============================================================================
# This ties everything together: data, model, loss, optimizer, training!

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# STEP 1: Prepare Data
# -----------------------------------------------------------------------------
# Let's create synthetic data for a simple classification problem

# Generate fake data: 1000 samples, 20 features each
torch.manual_seed(42)  # For reproducibility
X = torch.randn(1000, 20)
# Create labels: simple rule (sum of first 5 features > 0 -> class 1)
y = (X[:, :5].sum(dim=1) > 0).float().unsqueeze(1)

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Class distribution: {y.sum().item():.0f} positive, {(1-y).sum().item():.0f} negative")

# Split into train/test
train_X, test_X = X[:800], X[800:]
train_y, test_y = y[:800], y[800:]

# Create DataLoaders (handles batching and shuffling)
train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"\nTraining batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# -----------------------------------------------------------------------------
# STEP 2: Define Model
# -----------------------------------------------------------------------------

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )

    def forward(self, x):
        return self.network(x)

model = BinaryClassifier()
print(f"\nModel:\n{model}")

# -----------------------------------------------------------------------------
# STEP 3: Define Loss Function and Optimizer
# -----------------------------------------------------------------------------

# Binary Cross Entropy for binary classification
criterion = nn.BCELoss()

# Adam optimizer with learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\nLoss function: {criterion}")
print(f"Optimizer: {optimizer}")

# -----------------------------------------------------------------------------
# STEP 4: Training Loop
# -----------------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer):
    """Train for one epoch."""
    model.train()  # Set to training mode
    total_loss = 0

    for batch_X, batch_y in dataloader:
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update weights

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    """Evaluate model on a dataset."""
    model.eval()  # Set to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed for evaluation
        for batch_X, batch_y in dataloader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()

            # Calculate accuracy
            predicted_labels = (predictions > 0.5).float()
            correct += (predicted_labels == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# -----------------------------------------------------------------------------
# STEP 5: Train the Model!
# -----------------------------------------------------------------------------

num_epochs = 20
print("\nTraining started...")
print("-" * 50)

for epoch in range(num_epochs):
    # Train
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)

    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.2%}")

print("-" * 50)
print("Training complete!")

# -----------------------------------------------------------------------------
# STEP 6: Save and Load Model
# -----------------------------------------------------------------------------

# Save model weights
torch.save(model.state_dict(), "model_weights.pth")
print("\nModel saved to 'model_weights.pth'")

# Load model weights (into a new model instance)
new_model = BinaryClassifier()
new_model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
new_model.eval()
print("Model loaded successfully!")

# Verify it works
test_loss, test_acc = evaluate(new_model, test_loader, criterion)
print(f"Loaded model accuracy: {test_acc:.2%}")

# -----------------------------------------------------------------------------
# STEP 7: Make Predictions
# -----------------------------------------------------------------------------

# Single prediction
sample = test_X[0:1]  # Keep batch dimension
model.eval()
with torch.no_grad():
    prob = model(sample).item()
    predicted_class = 1 if prob > 0.5 else 0
    actual_class = int(test_y[0].item())

print(f"\nSample prediction:")
print(f"  Probability: {prob:.4f}")
print(f"  Predicted: {predicted_class}, Actual: {actual_class}")

# -----------------------------------------------------------------------------
# SUMMARY: The Training Loop Pattern
# -----------------------------------------------------------------------------
"""
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        # 1. Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # 2. Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""

# -----------------------------------------------------------------------------
# EXERCISE: Modify this code to:
# 1. Add a learning rate scheduler (torch.optim.lr_scheduler)
# 2. Implement early stopping if validation loss doesn't improve for 5 epochs
# 3. Track and plot training/test loss over epochs
# -----------------------------------------------------------------------------
