#!/usr/bin/env python3
"""
================================================================================
LESSON 06: Convolutional Neural Networks (CNN)
================================================================================
Learn to build CNNs for image classification. This lesson applies everything
you've learned to a real computer vision task.

Run: python 06_cnn_example.py
================================================================================
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import (
    header, subheader, success, error, info, warning,
    ExerciseTest, LessonRunner, ask_question,
    test_equal, test_shape, test_true
)

# Create the lesson runner
lesson = LessonRunner(
    "06_cnn",
    "Build Convolutional Neural Networks for image classification."
)

# =============================================================================
# SECTION 1: Convolution Operations
# =============================================================================

@lesson.section("Convolution Operations")
def section_conv():
    """Understand how convolution works on images."""

    print("Convolutions slide a kernel over an image to extract features.\n")

    print("-" * 50)
    print("HOW CONVOLUTION WORKS:")
    print("-" * 50)
    print("""
    Input Image     Kernel (3x3)      Output
    ┌─────────┐     ┌─────────┐
    │ 1 2 3 4 │     │ 1 0 1 │       Slide kernel
    │ 5 6 7 8 │  *  │ 0 1 0 │  →    across input,
    │ 9 0 1 2 │     │ 1 0 1 │       compute dot products
    │ 3 4 5 6 │     └─────────┘
    └─────────┘
    """)

    print("-" * 50)
    print("nn.Conv2d PARAMETERS:")
    print("-" * 50)

    conv = nn.Conv2d(
        in_channels=3,     # RGB = 3 channels
        out_channels=16,   # Number of filters/features
        kernel_size=3,     # 3x3 filter
        stride=1,          # Move 1 pixel at a time
        padding=1          # Add border to preserve size
    )

    print(f"""
    nn.Conv2d(
        in_channels=3,     # Input channels (RGB=3, grayscale=1)
        out_channels=16,   # Number of filters to learn
        kernel_size=3,     # Filter size (3x3)
        stride=1,          # Step size
        padding=1          # Border padding
    )
    """)

    # Demonstrate
    x = torch.randn(1, 3, 32, 32)  # (batch, channels, height, width)
    y = conv(x)

    print(f"Input shape:  {x.shape}  (batch, channels, H, W)")
    print(f"Output shape: {y.shape}")

    print("\n" + "-" * 50)
    print("OUTPUT SIZE FORMULA:")
    print("-" * 50)
    print("""
    output_size = (input_size - kernel_size + 2*padding) / stride + 1

    Example: (32 - 3 + 2*1) / 1 + 1 = 32
    With padding=1, kernel=3, stride=1 → size preserved!
    """)


@lesson.exercise("Convolution Quiz", points=1)
def exercise_conv():
    """Test understanding of convolution."""

    answer = ask_question(
        "Conv2d with in_channels=3, out_channels=32, kernel=3x3 has how many parameters?",
        ["96", "288", "864 + 32 = 896", "32"]
    )

    # 3 * 32 * 3 * 3 + 32 (bias) = 864 + 32 = 896
    test = ExerciseTest("Conv Parameters", hint="in * out * k * k + bias")
    test.check_true(answer == 2, "Correct! 3×32×3×3 + 32 = 896 parameters")
    return test.run()


# =============================================================================
# SECTION 2: Pooling Layers
# =============================================================================

@lesson.section("Pooling Layers")
def section_pooling():
    """Learn how pooling reduces spatial dimensions."""

    print("Pooling reduces spatial size while keeping important features.\n")

    print("-" * 50)
    print("MAX POOLING:")
    print("-" * 50)
    print("""
    Input (4x4)           MaxPool2d(2)      Output (2x2)
    ┌───┬───┬───┬───┐                       ┌───┬───┐
    │ 1 │ 3 │ 2 │ 1 │                       │   │   │
    ├───┼───┼───┼───┤     Take max          │ 4 │ 6 │
    │ 4 │ 2 │ 6 │ 5 │  ───────────────→     ├───┼───┤
    ├───┼───┼───┼───┤     in each           │ 8 │ 9 │
    │ 7 │ 8 │ 1 │ 2 │     2x2 region        │   │   │
    ├───┼───┼───┼───┤                       └───┴───┘
    │ 3 │ 5 │ 9 │ 4 │
    └───┴───┴───┴───┘
    """)

    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    x = torch.randn(1, 16, 32, 32)
    y = maxpool(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}  (halved spatial dimensions)")

    print("\n" + "-" * 50)
    print("POOLING TYPES:")
    print("-" * 50)
    print("""
    nn.MaxPool2d(2)           # Take max in each region
    nn.AvgPool2d(2)           # Take average
    nn.AdaptiveAvgPool2d((1,1))  # Output fixed size (for any input)
    """)

    print("-" * 50)
    print("WHY POOLING?")
    print("-" * 50)
    print("""
    1. Reduces spatial dimensions → fewer parameters
    2. Provides translation invariance
    3. Controls overfitting
    """)


@lesson.exercise("Pooling Quiz", points=1)
def exercise_pooling():
    """Test understanding of pooling."""

    answer = ask_question(
        "After MaxPool2d(2) on a 64x64 image, what's the size?",
        ["128x128", "32x32", "64x64", "16x16"]
    )

    test = ExerciseTest("Pooling Output", hint="Pool(2) halves each dimension")
    test.check_true(answer == 1, "Correct! 64/2 = 32 → output is 32x32")
    return test.run()


# =============================================================================
# SECTION 3: CNN Architecture
# =============================================================================

@lesson.section("CNN Architecture")
def section_architecture():
    """Learn the typical structure of a CNN."""

    print("CNNs follow a pattern: CONV → RELU → POOL → ... → FC\n")

    print("-" * 50)
    print("TYPICAL CNN STRUCTURE:")
    print("-" * 50)
    print("""
    Input Image (3, 64, 64)
           │
           ▼
    ┌─────────────────────────────────┐
    │  CONVOLUTIONAL BLOCKS           │
    │  Conv → ReLU → Pool (repeat)    │
    │  Extract hierarchical features  │
    └─────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────┐
    │  FLATTEN                        │
    │  Reshape to 1D vector           │
    └─────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────┐
    │  FULLY CONNECTED LAYERS         │
    │  Linear → ReLU → Linear         │
    │  Classification/regression      │
    └─────────────────────────────────┘
           │
           ▼
    Output (num_classes)
    """)

    print("-" * 50)
    print("WHAT EACH PART LEARNS:")
    print("-" * 50)
    print("""
    Early conv layers: Edges, colors, simple textures
    Middle conv layers: Shapes, patterns, object parts
    Later conv layers: Complex features, object parts
    FC layers: Combine features for classification
    """)


@lesson.exercise("Architecture Quiz", points=1)
def exercise_architecture():
    """Test understanding of CNN architecture."""

    answer = ask_question(
        "What operation connects conv layers to fully connected layers?",
        ["MaxPool", "Flatten", "ReLU", "Softmax"]
    )

    test = ExerciseTest("CNN Structure", hint="Conv outputs are 3D, FC expects 1D")
    test.check_true(answer == 1, "Correct! Flatten converts 3D features to 1D vector")
    return test.run()


# =============================================================================
# SECTION 4: Building a CNN
# =============================================================================

@lesson.section("Building a CNN")
def section_building():
    """Build a CNN from scratch."""

    print("Let's build a CNN for 32x32 RGB images (like CIFAR-10).\n")

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()

            # Convolutional layers
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

            # Pooling
            self.pool = nn.MaxPool2d(2, 2)

            # Fully connected layers
            # After 3 pools: 32 -> 16 -> 8 -> 4
            self.fc1 = nn.Linear(128 * 4 * 4, 256)
            self.fc2 = nn.Linear(256, num_classes)

            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            # Conv block 1: (3, 32, 32) -> (32, 16, 16)
            x = self.pool(F.relu(self.conv1(x)))

            # Conv block 2: (32, 16, 16) -> (64, 8, 8)
            x = self.pool(F.relu(self.conv2(x)))

            # Conv block 3: (64, 8, 8) -> (128, 4, 4)
            x = self.pool(F.relu(self.conv3(x)))

            # Flatten: (128, 4, 4) -> (2048)
            x = x.view(x.size(0), -1)

            # Fully connected
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

            return x

    model = SimpleCNN(num_classes=10)
    print("Model architecture:")
    print(model)

    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    y = model(x)

    print(f"\nInput:  {x.shape}")
    print(f"Output: {y.shape}")

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")


@lesson.exercise("Building Quiz", points=1)
def exercise_building():
    """Test understanding of CNN building."""

    answer = ask_question(
        "Why use padding=1 with kernel_size=3?",
        [
            "To increase output size",
            "To preserve spatial dimensions",
            "To reduce computation",
            "To add regularization"
        ]
    )

    test = ExerciseTest("Padding", hint="(32 - 3 + 2*1)/1 + 1 = 32")
    test.check_true(answer == 1, "Correct! padding=1 with kernel=3 preserves size")
    return test.run()


# =============================================================================
# SECTION 5: Image Preprocessing
# =============================================================================

@lesson.section("Image Preprocessing")
def section_preprocessing():
    """Learn to preprocess images for CNNs."""

    print("Images need preprocessing before feeding to a CNN.\n")

    print("-" * 50)
    print("TORCHVISION TRANSFORMS:")
    print("-" * 50)
    print("""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((64, 64)),      # Resize to fixed size
        transforms.ToTensor(),             # PIL Image → Tensor [0,1]
        transforms.Normalize(              # Standardize
            mean=[0.485, 0.456, 0.406],   # ImageNet means
            std=[0.229, 0.224, 0.225]     # ImageNet stds
        ),
    ])
    """)

    print("-" * 50)
    print("DATA AUGMENTATION (training only):")
    print("-" * 50)
    print("""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(...),
    ])
    """)

    print("-" * 50)
    print("IMAGE DIMENSIONS:")
    print("-" * 50)
    print("""
    PIL Image: (H, W, C) - Height, Width, Channels
    PyTorch:   (C, H, W) - Channels first!
    Batch:     (N, C, H, W) - Batch, Channels, Height, Width
    """)


@lesson.exercise("Preprocessing Quiz", points=1)
def exercise_preprocessing():
    """Test understanding of preprocessing."""

    answer = ask_question(
        "What does transforms.ToTensor() do?",
        [
            "Converts tensor to PIL Image",
            "Converts PIL Image to tensor and scales to [0,1]",
            "Normalizes with mean and std",
            "Resizes the image"
        ]
    )

    test = ExerciseTest("Transforms", hint="ToTensor converts and scales pixel values")
    test.check_true(answer == 1, "Correct! ToTensor converts PIL→Tensor and scales 0-255 to 0-1")
    return test.run()


# =============================================================================
# SECTION 6: Training a CNN
# =============================================================================

@lesson.section("Training a CNN")
def section_training():
    """Train a CNN on synthetic image data."""

    print("Let's train a CNN on synthetic data!\n")

    # Create synthetic "image" data
    torch.manual_seed(42)

    # Fake images: 500 samples of 3x16x16
    X = torch.randn(500, 3, 16, 16)
    # Label based on mean of red channel
    y = (X[:, 0].mean(dim=(1, 2)) > 0).long()

    train_X, test_X = X[:400], X[400:]
    train_y, test_y = y[:400], y[400:]

    train_loader = DataLoader(
        list(zip(train_X, train_y)), batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        list(zip(test_X, test_y)), batch_size=32
    )

    # Simple CNN
    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(32 * 4 * 4, 64)
            self.fc2 = nn.Linear(64, 2)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # 16 -> 8
            x = self.pool(F.relu(self.conv2(x)))  # 8 -> 4
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    model = TinyCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    for epoch in range(20):
        model.train()
        for batch_X, batch_y in train_loader:
            pred = model(batch_X)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/20 completed")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    print(f"\nTest Accuracy: {correct/total:.2%}")


@lesson.exercise("CNN Training Quiz", points=1)
def exercise_training():
    """Test understanding of CNN training."""

    answer = ask_question(
        "Why is data augmentation only applied during training?",
        [
            "It would make testing too slow",
            "We want consistent, reproducible test results",
            "Augmentation doesn't work on test data",
            "Test data is already augmented"
        ]
    )

    test = ExerciseTest("Augmentation", hint="We need consistent evaluation")
    test.check_true(answer == 1, "Correct! Test data should be consistent for fair comparison")
    return test.run()


# =============================================================================
# SECTION 7: Famous CNN Architectures
# =============================================================================

@lesson.section("Famous CNN Architectures")
def section_famous():
    """Learn about influential CNN architectures."""

    print("-" * 50)
    print("EVOLUTION OF CNN ARCHITECTURES:")
    print("-" * 50)
    print("""
    LeNet-5 (1998)
    └── First successful CNN for digit recognition
        5 layers, ~60K parameters

    AlexNet (2012)
    └── Won ImageNet, started deep learning revolution
        8 layers, 60M parameters, used ReLU and dropout

    VGG (2014)
    └── Very deep (16-19 layers), all 3x3 convolutions
        Simple and uniform architecture

    ResNet (2015)
    └── Skip connections enable very deep networks (152+ layers)
        Key insight: residual learning

    EfficientNet (2019)
    └── Compound scaling of width, depth, resolution
        State-of-the-art efficiency
    """)

    print("-" * 50)
    print("USING PRETRAINED MODELS:")
    print("-" * 50)
    print("""
    from torchvision import models

    # Load pretrained ResNet
    model = models.resnet18(pretrained=True)

    # Replace final layer for your task
    model.fc = nn.Linear(512, num_classes)

    # Freeze early layers (optional)
    for param in model.parameters():
        param.requires_grad = False
    model.fc.requires_grad = True
    """)


# =============================================================================
# FINAL CODING EXERCISE
# =============================================================================

@lesson.exercise("Coding Challenge: Build a CNN", points=3)
def exercise_final():
    """Build and test a CNN architecture."""

    print("Build a CNN for 28x28 grayscale images (like MNIST).\n")
    print("Requirements:")
    print("  - Input: (1, 28, 28) grayscale image")
    print("  - Two conv layers with pooling")
    print("  - Output: 10 classes")

    class MyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Conv layers
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2)

            # After 2 pools: 28 -> 14 -> 7
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(0.25)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # 28 -> 14
            x = self.pool(F.relu(self.conv2(x)))  # 14 -> 7
            x = x.view(x.size(0), -1)             # Flatten
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = MyCNN()
    print("Created CNN:")
    print(model)

    # Test
    test = ExerciseTest("CNN Architecture")

    # Test forward pass
    x = torch.randn(8, 1, 28, 28)
    y = model(x)

    test.check_shape(y, (8, 10), "Output shape should be (batch, 10)")

    # Check it has conv layers
    has_conv = any(isinstance(m, nn.Conv2d) for m in model.modules())
    test.check_true(has_conv, "Model should have Conv2d layers")

    # Check reasonable parameter count
    total = sum(p.numel() for p in model.parameters())
    test.check_true(total > 10000, f"Model has {total:,} parameters (should be substantial)")
    test.check_true(total < 10000000, f"Model not too large ({total:,} params)")

    return test.run()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if "--test" in sys.argv:
        lesson.run()
    else:
        lesson.run_interactive()
