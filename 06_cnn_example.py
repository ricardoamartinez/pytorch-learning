# =============================================================================
# LESSON 6: Convolutional Neural Network (CNN) for Image Classification
# =============================================================================
# A real-world example: MNIST digit classification

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# STEP 1: Data Loading with Transforms
# -----------------------------------------------------------------------------

# Transforms: preprocessing pipeline
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor [0, 1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Download and load MNIST dataset
print("Downloading MNIST dataset...")
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Inspect one sample
sample_image, sample_label = train_dataset[0]
print(f"Image shape: {sample_image.shape}")  # [1, 28, 28] = [channels, height, width]
print(f"Label: {sample_label}")

# -----------------------------------------------------------------------------
# STEP 2: Define CNN Architecture
# -----------------------------------------------------------------------------

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        # Conv2d(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling

        # Dropout for regularization
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Fully connected layers
        # After 2 pooling operations: 28 -> 14 -> 7
        # So we have 64 channels * 7 * 7 = 3136 features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Input: [batch, 1, 28, 28]

        # First conv block
        x = self.conv1(x)           # [batch, 32, 28, 28]
        x = F.relu(x)
        x = self.pool(x)            # [batch, 32, 14, 14]

        # Second conv block
        x = self.conv2(x)           # [batch, 64, 14, 14]
        x = F.relu(x)
        x = self.pool(x)            # [batch, 64, 7, 7]
        x = self.dropout1(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)   # [batch, 3136]

        # Fully connected layers
        x = self.fc1(x)             # [batch, 128]
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)             # [batch, 10]

        return x  # Raw logits (no softmax - CrossEntropyLoss handles it)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = CNN().to(device)
print(f"\nModel architecture:\n{model}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# -----------------------------------------------------------------------------
# STEP 3: Loss and Optimizer
# -----------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------------------------------------------------------
# STEP 4: Training Functions
# -----------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # Get class with highest score
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total


def test(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

# -----------------------------------------------------------------------------
# STEP 5: Train the Model
# -----------------------------------------------------------------------------

num_epochs = 5
print("\nStarting training...")
print("-" * 60)

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

print("-" * 60)
print("Training complete!")

# -----------------------------------------------------------------------------
# STEP 6: Visualize Predictions (optional - requires matplotlib)
# -----------------------------------------------------------------------------

# Make predictions on a few test images
model.eval()
test_images, test_labels = next(iter(test_loader))
test_images, test_labels = test_images.to(device), test_labels.to(device)

with torch.no_grad():
    outputs = model(test_images[:10])
    _, predictions = outputs.max(1)

print("\nSample predictions:")
print(f"Predictions: {predictions.tolist()}")
print(f"Actual:      {test_labels[:10].tolist()}")

# -----------------------------------------------------------------------------
# CNN LAYER CHEAT SHEET
# -----------------------------------------------------------------------------
"""
CONVOLUTIONAL LAYERS:
- nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
- Output size: (W - K + 2P) / S + 1

POOLING LAYERS:
- nn.MaxPool2d(kernel_size, stride=None)  # stride defaults to kernel_size
- nn.AvgPool2d(kernel_size, stride=None)
- nn.AdaptiveAvgPool2d(output_size)  # Forces specific output size

NORMALIZATION:
- nn.BatchNorm2d(num_features)  # After conv layers
- nn.LayerNorm(normalized_shape)  # For transformers

REGULARIZATION:
- nn.Dropout(p)      # For fully connected layers
- nn.Dropout2d(p)    # For conv layers (drops entire channels)
"""

# -----------------------------------------------------------------------------
# EXERCISE:
# 1. Add BatchNorm2d after each conv layer
# 2. Try adding a third convolutional layer
# 3. Experiment with different kernel sizes and channel numbers
# 4. Implement a learning rate scheduler
# -----------------------------------------------------------------------------
