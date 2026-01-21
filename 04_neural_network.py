#!/usr/bin/env python3
"""
================================================================================
LESSON 04: Building Neural Networks with nn.Module
================================================================================
Learn to construct neural networks using PyTorch's nn.Module - the base class
for all neural network components.

Run: python 04_neural_network.py
================================================================================
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (
    header, subheader, success, error, info, warning,
    ExerciseTest, LessonRunner, ask_question,
    test_equal, test_shape, test_true, verify_model
)

# Create the lesson runner
lesson = LessonRunner(
    "04_neural_network",
    "Build neural networks using PyTorch's nn.Module system."
)

# =============================================================================
# SECTION 1: Linear Layers
# =============================================================================

@lesson.section("Linear Layers")
def section_linear():
    """Understand the fundamental building block: linear layers."""

    print("Linear layers perform: y = xW^T + b\n")
    print("This is the core operation in neural networks!\n")

    print("-" * 50)
    print("CREATING A LINEAR LAYER:")
    print("-" * 50)

    # Create a linear layer
    linear = nn.Linear(in_features=10, out_features=5)

    print(f"\nnn.Linear(in_features=10, out_features=5)")
    print(f"\nThis transforms: 10 inputs -> 5 outputs")
    print(f"\nParameters:")
    print(f"  Weight shape: {linear.weight.shape}  (out_features × in_features)")
    print(f"  Bias shape:   {linear.bias.shape}     (out_features)")

    # Forward pass
    x = torch.randn(3, 10)  # Batch of 3, 10 features each
    y = linear(x)

    print(f"\n" + "-" * 50)
    print("FORWARD PASS:")
    print("-" * 50)
    print(f"\nInput shape:  {x.shape}  (batch_size=3, features=10)")
    print(f"Output shape: {y.shape}  (batch_size=3, features=5)")

    print("\n" + "-" * 50)
    print("PARAMETER COUNT:")
    print("-" * 50)
    total = sum(p.numel() for p in linear.parameters())
    print(f"\nWeight: 10 × 5 = 50")
    print(f"Bias: 5")
    print(f"Total: {total}")


@lesson.exercise("Linear Layer Quiz", points=1)
def exercise_linear():
    """Test understanding of linear layers."""

    answer = ask_question(
        "nn.Linear(100, 50) has how many parameters (with bias)?",
        ["5000", "5050", "150", "5100"]
    )

    # 100*50 (weights) + 50 (bias) = 5050
    test = ExerciseTest("Linear Parameters", hint="weights + bias = (in × out) + out")
    test.check_true(answer == 1, "Correct! 100×50 + 50 = 5050 parameters")
    return test.run()


# =============================================================================
# SECTION 2: Activation Functions
# =============================================================================

@lesson.section("Activation Functions")
def section_activations():
    """Learn about non-linear activation functions."""

    print("Activations add non-linearity - essential for learning complex patterns!\n")
    print("Without activations, stacked linear layers = just one linear layer.\n")

    x = torch.tensor([-2., -1., 0., 1., 2.])

    print("-" * 50)
    print("COMMON ACTIVATION FUNCTIONS:")
    print("-" * 50)

    print(f"\nInput x = {x.tolist()}")

    # ReLU
    relu = nn.ReLU()
    print(f"\nReLU(x)     = {relu(x).tolist()}")
    print("  Formula: max(0, x)")
    print("  Use: Most common for hidden layers")

    # Sigmoid
    sigmoid = nn.Sigmoid()
    print(f"\nSigmoid(x)  = {[f'{v:.3f}' for v in sigmoid(x).tolist()]}")
    print("  Formula: 1 / (1 + e^-x)")
    print("  Use: Binary classification output, gates")

    # Tanh
    tanh = nn.Tanh()
    print(f"\nTanh(x)     = {[f'{v:.3f}' for v in tanh(x).tolist()]}")
    print("  Formula: (e^x - e^-x) / (e^x + e^-x)")
    print("  Use: RNNs, outputs in [-1, 1]")

    # Softmax
    softmax = nn.Softmax(dim=0)
    print(f"\nSoftmax(x)  = {[f'{v:.3f}' for v in softmax(x).tolist()]}")
    print(f"  Sum = {softmax(x).sum():.3f}")
    print("  Use: Multi-class classification output")

    # LeakyReLU
    leaky = nn.LeakyReLU(0.1)
    print(f"\nLeakyReLU(x)= {leaky(x).tolist()}")
    print("  Formula: max(0.1x, x)")
    print("  Use: Prevents 'dead' neurons")

    print("\n" + "-" * 50)
    print("FUNCTIONAL vs MODULE API:")
    print("-" * 50)
    print("""
    # As modules (in __init__):
    self.relu = nn.ReLU()
    y = self.relu(x)

    # As functions (in forward):
    import torch.nn.functional as F
    y = F.relu(x)
    """)


@lesson.exercise("Activation Quiz", points=1)
def exercise_activations():
    """Test understanding of activations."""

    answer = ask_question(
        "Which activation outputs probabilities that sum to 1?",
        ["ReLU", "Sigmoid", "Softmax", "Tanh"]
    )

    test = ExerciseTest("Activations", hint="Think about multi-class classification")
    test.check_true(answer == 2, "Correct! Softmax outputs a probability distribution")
    return test.run()


# =============================================================================
# SECTION 3: Building with nn.Sequential
# =============================================================================

@lesson.section("Building with nn.Sequential")
def section_sequential():
    """Create networks by stacking layers in sequence."""

    print("nn.Sequential: Stack layers in order - simplest way to build networks!\n")

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    print("-" * 50)
    print("NETWORK ARCHITECTURE:")
    print("-" * 50)
    print(model)

    print("\n" + "-" * 50)
    print("LAYER DETAILS:")
    print("-" * 50)
    print("\n  Input:  784 (e.g., 28×28 image flattened)")
    print("  Hidden: 256 -> ReLU -> 128 -> ReLU")
    print("  Output: 10 (e.g., 10 digit classes)")

    # Forward pass
    x = torch.randn(32, 784)  # Batch of 32 images
    y = model(x)

    print(f"\n" + "-" * 50)
    print("FORWARD PASS:")
    print("-" * 50)
    print(f"\nInput:  {x.shape}")
    print(f"Output: {y.shape}")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")


@lesson.exercise("Sequential Quiz", points=1)
def exercise_sequential():
    """Test understanding of nn.Sequential."""

    answer = ask_question(
        "What goes BETWEEN linear layers in most networks?",
        [
            "Another linear layer",
            "Activation function",
            "Loss function",
            "Optimizer"
        ]
    )

    test = ExerciseTest("Sequential", hint="We need non-linearity!")
    test.check_true(answer == 1, "Correct! Activations provide non-linearity")
    return test.run()


# =============================================================================
# SECTION 4: Custom nn.Module Classes
# =============================================================================

@lesson.section("Custom nn.Module Classes")
def section_custom():
    """Create custom neural networks by subclassing nn.Module."""

    print("For more control, create custom classes inheriting from nn.Module.\n")

    print("-" * 50)
    print("TEMPLATE:")
    print("-" * 50)
    print("""
    class MyNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()  # Always call parent's __init__!

            # Define layers
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.layer2 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            # Define how data flows through the network
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            return x
    """)

    # Create actual implementation
    class MLP(nn.Module):
        """Multi-Layer Perceptron with configurable architecture."""

        def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.2):
            super().__init__()

            # Build layers dynamically
            layers = []
            prev_size = input_size

            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, num_classes))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    print("-" * 50)
    print("EXAMPLE: Flexible MLP")
    print("-" * 50)

    model = MLP(
        input_size=784,
        hidden_sizes=[512, 256, 128],
        num_classes=10,
        dropout=0.2
    )

    print(model)

    x = torch.randn(8, 784)
    y = model(x)
    print(f"\nInput:  {x.shape}")
    print(f"Output: {y.shape}")


@lesson.exercise("Custom Module Quiz", points=1)
def exercise_custom():
    """Test understanding of custom modules."""

    answer = ask_question(
        "In nn.Module, where do you define the layer STRUCTURE?",
        [
            "In forward()",
            "In __init__()",
            "In backward()",
            "Outside the class"
        ]
    )

    test = ExerciseTest("Custom Module", hint="Where are layers created?")
    test.check_true(answer == 1, "Correct! Layers are defined in __init__, used in forward()")
    return test.run()


# =============================================================================
# SECTION 5: Model Inspection and Parameters
# =============================================================================

@lesson.section("Model Inspection")
def section_inspection():
    """Learn to inspect and work with model parameters."""

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )

    print("-" * 50)
    print("VIEWING PARAMETERS:")
    print("-" * 50)

    print("\n# All parameters:")
    print("for name, param in model.named_parameters():")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

    print("\n# Parameter count:")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total}")
    print(f"Trainable parameters: {trainable}")

    print("\n" + "-" * 50)
    print("FREEZING PARAMETERS:")
    print("-" * 50)
    print("""
    # Freeze all parameters (e.g., for fine-tuning)
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specific layer
    for param in model.classifier.parameters():
        param.requires_grad = True
    """)

    print("\n" + "-" * 50)
    print("MOVING MODEL TO DEVICE:")
    print("-" * 50)
    print("""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Move all parameters to GPU
    """)


@lesson.exercise("Model Parameters Quiz", points=1)
def exercise_inspection():
    """Test understanding of model parameters."""

    model = nn.Linear(100, 50, bias=True)
    total = sum(p.numel() for p in model.parameters())

    answer = ask_question(
        f"A Linear(100, 50) layer with bias has {total} parameters. What if bias=False?",
        ["5000", "5050", "50", "100"]
    )

    test = ExerciseTest("Parameters", hint="Without bias, only weights remain")
    test.check_true(answer == 0, "Correct! Without bias: 100×50 = 5000")
    return test.run()


# =============================================================================
# SECTION 6: Train vs Eval Mode
# =============================================================================

@lesson.section("Train vs Eval Mode")
def section_modes():
    """Understand training vs evaluation modes."""

    print("Some layers behave differently during training vs inference!\n")

    print("-" * 50)
    print("AFFECTED LAYERS:")
    print("-" * 50)
    print("""
    Dropout:
      - Train: Randomly zeros elements (regularization)
      - Eval:  Does nothing (all elements kept)

    BatchNorm:
      - Train: Uses batch statistics, updates running stats
      - Eval:  Uses stored running statistics
    """)

    print("-" * 50)
    print("SWITCHING MODES:")
    print("-" * 50)

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Dropout(0.5),
        nn.Linear(20, 5),
    )

    print(f"\nmodel.training = {model.training}")

    model.eval()
    print(f"After model.eval(): model.training = {model.training}")

    model.train()
    print(f"After model.train(): model.training = {model.training}")

    print("\n" + "-" * 50)
    print("TYPICAL PATTERN:")
    print("-" * 50)
    print("""
    # Training
    model.train()
    for batch in train_loader:
        ...

    # Evaluation
    model.eval()
    with torch.no_grad():  # Also disable gradients for speed
        for batch in test_loader:
            ...
    """)


@lesson.exercise("Train/Eval Quiz", points=1)
def exercise_modes():
    """Test understanding of train/eval modes."""

    answer = ask_question(
        "During model.eval(), what happens to Dropout layers?",
        [
            "They drop more neurons",
            "They drop fewer neurons",
            "They are disabled (pass input through)",
            "They raise an error"
        ]
    )

    test = ExerciseTest("Modes", hint="We want consistent behavior during inference")
    test.check_true(answer == 2, "Correct! Dropout is disabled during eval")
    return test.run()


# =============================================================================
# SECTION 7: Common Layer Types
# =============================================================================

@lesson.section("Common Layer Types")
def section_layers():
    """Overview of commonly used layer types."""

    print("-" * 50)
    print("LAYER TYPES REFERENCE:")
    print("-" * 50)

    print("""
    LINEAR LAYERS:
      nn.Linear(in, out)           Fully connected layer
      nn.Bilinear(in1, in2, out)   Bilinear transformation

    CONVOLUTIONAL (for images):
      nn.Conv1d(in_ch, out_ch, kernel)   1D convolution
      nn.Conv2d(in_ch, out_ch, kernel)   2D convolution (images!)
      nn.ConvTranspose2d(...)            Upsampling convolution

    POOLING:
      nn.MaxPool2d(kernel)          Max pooling
      nn.AvgPool2d(kernel)          Average pooling
      nn.AdaptiveAvgPool2d(size)    Output size is fixed

    RECURRENT (for sequences):
      nn.RNN(input, hidden)         Basic RNN
      nn.LSTM(input, hidden)        Long Short-Term Memory
      nn.GRU(input, hidden)         Gated Recurrent Unit

    NORMALIZATION:
      nn.BatchNorm1d(features)      Batch normalization (1D)
      nn.BatchNorm2d(channels)      Batch normalization (2D)
      nn.LayerNorm(shape)           Layer normalization

    REGULARIZATION:
      nn.Dropout(p)                 Random dropout
      nn.Dropout2d(p)               Dropout for conv layers

    EMBEDDING (for NLP):
      nn.Embedding(vocab, dim)      Lookup table for words

    TRANSFORMER:
      nn.Transformer(...)           Full transformer
      nn.MultiheadAttention(...)    Multi-head attention
    """)


# =============================================================================
# FINAL CODING EXERCISE
# =============================================================================

@lesson.exercise("Coding Challenge: Build a Classifier", points=3)
def exercise_final():
    """Build a neural network classifier from scratch."""

    print("Build a network for classifying 28×28 images into 10 classes.\n")
    print("Requirements:")
    print("  - Input: 784 (28×28 flattened)")
    print("  - Two hidden layers: 256 and 128 neurons")
    print("  - ReLU activations")
    print("  - Dropout (0.2) after each hidden layer")
    print("  - Output: 10 classes")

    class ImageClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(784, 256)
            self.layer2 = nn.Linear(256, 128)
            self.layer3 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = self.dropout(self.relu(self.layer1(x)))
            x = self.dropout(self.relu(self.layer2(x)))
            x = self.layer3(x)
            return x

    model = ImageClassifier()

    print("\nCreated model:")
    print(model)

    # Test
    test = ExerciseTest("Image Classifier")

    # Test forward pass
    x = torch.randn(32, 784)
    y = model(x)

    test.check_shape(y, (32, 10), "Output shape should be (batch, 10)")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    expected = 784*256 + 256 + 256*128 + 128 + 128*10 + 10
    test.check_true(total == expected, f"Parameter count: {total} (expected {expected})")

    # Test train/eval modes
    model.eval()
    test.check_true(model.training == False, "model.eval() sets training=False")

    model.train()
    test.check_true(model.training == True, "model.train() sets training=True")

    return test.run()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if "--test" in sys.argv:
        lesson.run()
    else:
        lesson.run_interactive()
