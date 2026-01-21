# =============================================================================
# LESSON 9: Autoencoders - Learning Compressed Representations
# =============================================================================
# Autoencoders learn to compress data into a latent space and reconstruct it.
# This is crucial for world models: we need compact state representations!
#
# Run interactively: python 09_autoencoders.py
# Run tests only:    python 09_autoencoders.py --test
# =============================================================================

import sys
sys.path.insert(0, '.')
from utils.lesson_utils import LessonRunner, ExerciseTest, ask_question, show_progress

import torch
import torch.nn as nn
import torch.nn.functional as F

# Create the lesson runner
lesson = LessonRunner("Lesson 9: Autoencoders", total_points=10)

# =============================================================================
# SECTION 1: The Concept of Autoencoders
# =============================================================================
@lesson.section("The Concept of Autoencoders")
def section_1():
    """
    AUTOENCODER ARCHITECTURE
    ========================

    INPUT -> [ENCODER] -> LATENT CODE -> [DECODER] -> RECONSTRUCTION
      x   ->    E(x)   ->      z      ->    D(z)   ->      x'

    Loss = ||x - x'||^2  (reconstruction error)

    The key insight: we force the network through a BOTTLENECK (latent code).
    To reconstruct well, it must learn a compressed representation!

    WHY AUTOENCODERS FOR WORLD MODELS?
    ==================================
    1. Compress high-dimensional observations (images) into compact latent codes
    2. Latent space is easier to model/predict than raw pixels
    3. Denoising autoencoders learn robust representations
    """
    print("Autoencoders: Compression and Reconstruction")
    print("=" * 50)

    print("""
    ARCHITECTURE DIAGRAM
    ====================

    Input (784 pixels)
          │
          ▼
    ┌───────────┐
    │  ENCODER  │  (Compress)
    │  784→256  │
    │  256→128  │
    │  128→32   │
    └─────┬─────┘
          │
          ▼
    ┌───────────┐
    │  LATENT   │  (Bottleneck: 32 dimensions)
    │   CODE z  │
    └─────┬─────┘
          │
          ▼
    ┌───────────┐
    │  DECODER  │  (Reconstruct)
    │  32→128   │
    │  128→256  │
    │  256→784  │
    └─────┬─────┘
          │
          ▼
    Output (784 pixels)
    """)

    print("-" * 50)
    print("WHY THIS WORKS")
    print("-" * 50)

    print("""
    The bottleneck forces compression:
    - Input:  784 dimensions (28x28 image)
    - Latent: 32 dimensions
    - Compression ratio: 784/32 = 24.5x

    To minimize reconstruction error, the encoder must:
    - Learn to extract the MOST IMPORTANT features
    - Discard redundant information

    The latent code z is a LEARNED REPRESENTATION!
    """)

    # Simple demonstration
    print("-" * 50)
    print("TYPES OF AUTOENCODERS")
    print("-" * 50)

    print("""
    1. VANILLA AUTOENCODER
       - Direct compression and reconstruction
       - Simple but effective

    2. DENOISING AUTOENCODER
       - Input corrupted with noise
       - Learn to reconstruct clean output
       - More robust representations

    3. SPARSE AUTOENCODER
       - Enforce sparsity in latent code
       - Only few neurons active at a time

    4. VARIATIONAL AUTOENCODER (VAE)
       - Probabilistic latent space
       - Can sample new data! (next lesson)
    """)

# -----------------------------------------------------------------------------
# QUIZ 1
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: Autoencoder Purpose", points=1)
def quiz_1():
    if '--test' not in sys.argv:
        answer = ask_question(
            "What forces an autoencoder to learn meaningful features?",
            [
                "The activation functions",
                "The bottleneck (latent code dimension smaller than input)",
                "Using many hidden layers",
                "The batch size"
            ]
        )
        return answer == 1  # Bottleneck
    return True

# =============================================================================
# SECTION 2: Fully Connected Autoencoder
# =============================================================================
@lesson.section("Fully Connected Autoencoder")
def section_2():
    """
    SIMPLE FULLY CONNECTED AUTOENCODER
    ==================================

    For images, we flatten them to vectors first.
    28x28 = 784 pixels → compress to latent_dim → reconstruct 784 pixels
    """
    print("Building a Fully Connected Autoencoder")
    print("=" * 50)

    class FCAutoencoder(nn.Module):
        def __init__(self, input_dim=784, latent_dim=32):
            super().__init__()

            # Encoder: compress to latent dimension
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim),
            )

            # Decoder: reconstruct from latent
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim),
                nn.Sigmoid(),  # Output in [0, 1] for images
            )

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z

    # Create model
    model = FCAutoencoder(input_dim=784, latent_dim=32)

    print("Model architecture:")
    print(model)

    print(f"\nConfiguration:")
    print(f"  Input dimension:  784 (flattened 28x28 image)")
    print(f"  Latent dimension: 32")
    print(f"  Compression ratio: {784/32:.1f}x")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    print("\n" + "-" * 50)
    print("FORWARD PASS")
    print("-" * 50)

    # Test
    x = torch.randn(8, 784)  # batch of 8 flattened images
    x_recon, z = model(x)

    print(f"Input shape:          {list(x.shape)}")
    print(f"Latent shape:         {list(z.shape)}")
    print(f"Reconstruction shape: {list(x_recon.shape)}")

    return FCAutoencoder, model

# -----------------------------------------------------------------------------
# EXERCISE 1: Build Simple Autoencoder
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Build Autoencoder", points=1)
def exercise_1():
    """Build a simple autoencoder with a specific architecture."""
    test = ExerciseTest("Simple Autoencoder")

    class SimpleAutoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            # YOUR TASK: Build encoder and decoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
                nn.Sigmoid(),
            )

        def forward(self, x):
            z = self.encoder(x)
            x_recon = self.decoder(z)
            return x_recon, z

    # Create and test
    model = SimpleAutoencoder(input_dim=100, latent_dim=10)
    x = torch.randn(4, 100)
    x_recon, z = model(x)

    test.check_shape(z, (4, 10), "latent shape")
    test.check_shape(x_recon, (4, 100), "reconstruction shape")
    test.check_true(
        x_recon.min() >= 0 and x_recon.max() <= 1,
        "output is in [0, 1] range (Sigmoid)"
    )

    return test.run()

# =============================================================================
# SECTION 3: Convolutional Autoencoder
# =============================================================================
@lesson.section("Convolutional Autoencoder")
def section_3():
    """
    CONVOLUTIONAL AUTOENCODER
    =========================

    For images, CNNs are much better than fully connected networks!

    Encoder: Uses Conv2d with stride=2 to downsample
    Decoder: Uses ConvTranspose2d to upsample

    This preserves spatial structure and uses fewer parameters.
    """
    print("Building a Convolutional Autoencoder")
    print("=" * 50)

    class ConvAutoencoder(nn.Module):
        def __init__(self, latent_dim=64):
            super().__init__()
            self.latent_dim = latent_dim

            # Encoder: Image -> Latent
            self.encoder_conv = nn.Sequential(
                # (1, 28, 28) -> (32, 14, 14)
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                # (32, 14, 14) -> (64, 7, 7)
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )

            # Flatten and project to latent
            self.encoder_fc = nn.Linear(64 * 7 * 7, latent_dim)

            # Decoder: Latent -> Image
            self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)

            self.decoder_conv = nn.Sequential(
                # (64, 7, 7) -> (32, 14, 14)
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                # (32, 14, 14) -> (1, 28, 28)
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            )

        def encode(self, x):
            x = self.encoder_conv(x)
            x = x.view(x.size(0), -1)  # Flatten
            z = self.encoder_fc(x)
            return z

        def decode(self, z):
            x = self.decoder_fc(z)
            x = x.view(-1, 64, 7, 7)  # Reshape
            x = self.decoder_conv(x)
            return x

        def forward(self, x):
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z

    model = ConvAutoencoder(latent_dim=64)

    print("Model structure:")
    print("-" * 50)
    print("ENCODER:")
    print("  Input:    (1, 28, 28)")
    print("  Conv2d:   (32, 14, 14)  stride=2 downsamples")
    print("  Conv2d:   (64, 7, 7)")
    print("  Flatten:  3136")
    print("  Linear:   64 (latent)")
    print("\nDECODER:")
    print("  Linear:   3136")
    print("  Reshape:  (64, 7, 7)")
    print("  ConvT2d:  (32, 14, 14)  stride=2 upsamples")
    print("  ConvT2d:  (1, 28, 28)")

    print("\n" + "-" * 50)
    print("FORWARD PASS TEST")
    print("-" * 50)

    x = torch.randn(4, 1, 28, 28)
    x_recon, z = model(x)

    print(f"Input shape:          {list(x.shape)}")
    print(f"Latent shape:         {list(z.shape)}")
    print(f"Reconstruction shape: {list(x_recon.shape)}")

    # Compare parameters
    fc_params = 784*256 + 256*128 + 128*32 + 32*128 + 128*256 + 256*784  # Rough FC count
    conv_params = sum(p.numel() for p in model.parameters())
    print(f"\nConv autoencoder parameters: {conv_params:,}")
    print("CNNs are much more parameter efficient for images!")

    return ConvAutoencoder, model

# -----------------------------------------------------------------------------
# QUIZ 2
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: ConvTranspose2d", points=1)
def quiz_2():
    if '--test' not in sys.argv:
        answer = ask_question(
            "What does ConvTranspose2d do in a decoder?",
            [
                "Reduces spatial dimensions (downsamples)",
                "Increases spatial dimensions (upsamples)",
                "Changes the number of channels only",
                "Applies pooling"
            ]
        )
        return answer == 1  # Upsamples
    return True

# =============================================================================
# SECTION 4: Training an Autoencoder
# =============================================================================
@lesson.section("Training an Autoencoder")
def section_4():
    """
    TRAINING LOOP
    =============

    Key insight: We DON'T need labels!
    - Input: x (the image)
    - Target: x (same image!)
    - Loss: ||x - reconstruction||^2

    This is SELF-SUPERVISED learning.
    """
    print("Training an Autoencoder")
    print("=" * 50)

    # Create model
    class SimpleConvAE(nn.Module):
        def __init__(self, latent_dim=32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32 * 7 * 7),
                nn.Unflatten(1, (32, 7, 7)),
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    model = SimpleConvAE(latent_dim=32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training configuration:")
    print(f"  Loss function: MSELoss (reconstruction error)")
    print(f"  Optimizer: Adam, lr=1e-3")
    print(f"  Latent dimension: 32")

    print("\n" + "-" * 50)
    print("SELF-SUPERVISED LEARNING")
    print("-" * 50)

    print("""
    Standard supervised learning:
        Loss = CrossEntropy(model(x), label)

    Autoencoder (self-supervised):
        Loss = MSE(model(x), x)  # Target is input itself!

    No labels needed - the data supervises itself!
    """)

    # Generate synthetic data for demo
    print("-" * 50)
    print("TRAINING DEMO")
    print("-" * 50)

    # Create simple synthetic data (random patterns)
    train_data = torch.rand(100, 1, 28, 28)

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        recon, _ = model(train_data)
        loss = criterion(recon, train_data)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")

    print("\nLoss decreasing = model learning to reconstruct!")

    return SimpleConvAE, model

# -----------------------------------------------------------------------------
# EXERCISE 2: Training Loop
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Train Autoencoder", points=1)
def exercise_2():
    """Complete a training loop for an autoencoder."""
    test = ExerciseTest("Training Loop")

    # Simple model
    class TinyAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Linear(64, 784),
                nn.Sigmoid(),
            )

        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return recon.view(-1, 1, 28, 28), z

    model = TinyAE()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Create synthetic training data
    train_data = torch.rand(50, 1, 28, 28)

    # YOUR TASK: Complete training loop
    initial_loss = criterion(model(train_data)[0], train_data).item()

    for epoch in range(30):
        optimizer.zero_grad()
        recon, _ = model(train_data)
        loss = criterion(recon, train_data)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(train_data)[0], train_data).item()

    test.check_true(
        final_loss < initial_loss,
        f"loss decreased ({initial_loss:.4f} -> {final_loss:.4f})"
    )
    test.check_true(
        final_loss < 0.1,
        f"final loss is low ({final_loss:.4f} < 0.1)"
    )

    return test.run()

# =============================================================================
# SECTION 5: Denoising Autoencoder
# =============================================================================
@lesson.section("Denoising Autoencoder")
def section_5():
    """
    DENOISING AUTOENCODER
    =====================

    Add noise to input, train to reconstruct CLEAN output.

    Benefits:
    - Prevents learning identity mapping
    - Forces learning of robust features
    - Better generalization

    Training:
    - Input: x + noise
    - Target: x (clean)
    """
    print("Denoising Autoencoder")
    print("=" * 50)

    print("""
    REGULAR AUTOENCODER:
    ====================
    Input x ──────> Encoder ──────> Decoder ──────> x'
                                                    │
    Loss = ||x - x'||²                              │
                    <───────────────────────────────┘

    DENOISING AUTOENCODER:
    ======================
    Input x ──> Add Noise ──> x̃ ──> Encoder ──> Decoder ──> x'
                                                             │
    Loss = ||x - x'||²   (compare to CLEAN x!)               │
               <─────────────────────────────────────────────┘

    The model must learn to REMOVE noise, not just copy!
    """)

    class DenoisingAutoencoder(nn.Module):
        def __init__(self, latent_dim=32, noise_factor=0.3):
            super().__init__()
            self.noise_factor = noise_factor

            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim),
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 784),
                nn.Sigmoid(),
            )

        def add_noise(self, x):
            """Add Gaussian noise to input."""
            noise = torch.randn_like(x) * self.noise_factor
            noisy = x + noise
            return torch.clamp(noisy, 0., 1.)

        def forward(self, x, add_noise=True):
            # Add noise only during training
            if add_noise and self.training:
                x_input = self.add_noise(x)
            else:
                x_input = x

            z = self.encoder(x_input)
            x_recon = self.decoder(z)
            return x_recon.view(-1, 1, 28, 28), z

    model = DenoisingAutoencoder(latent_dim=32, noise_factor=0.3)

    print("-" * 50)
    print("DEMONSTRATION")
    print("-" * 50)

    x = torch.rand(2, 1, 28, 28)  # Clean input

    model.train()
    noisy_recon, _ = model(x, add_noise=True)
    print(f"Training mode: Noise is added (noise_factor=0.3)")

    model.eval()
    clean_recon, _ = model(x, add_noise=False)
    print(f"Eval mode: No noise added")

    print("\nWhy this works:")
    print("- Can't just learn identity (input is corrupted)")
    print("- Must learn actual structure of the data")
    print("- Results in more robust features")

    return DenoisingAutoencoder, model

# -----------------------------------------------------------------------------
# EXERCISE 3: Denoising Autoencoder
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Denoising AE", points=1)
def exercise_3():
    """Build and test a denoising autoencoder."""
    test = ExerciseTest("Denoising AE")

    class DenoiseAE(nn.Module):
        def __init__(self, noise_factor=0.5):
            super().__init__()
            self.noise_factor = noise_factor
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(100, 20),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(20, 100),
                nn.Sigmoid(),
            )

        def add_noise(self, x):
            noise = torch.randn_like(x) * self.noise_factor
            return torch.clamp(x + noise, 0., 1.)

        def forward(self, x, training=True):
            if training:
                x = self.add_noise(x)
            z = self.encoder(x)
            return self.decoder(z), z

    model = DenoiseAE(noise_factor=0.5)

    # Test that noise is added during training
    x = torch.rand(4, 100)
    model.train()
    recon1, _ = model(x, training=True)
    recon2, _ = model(x, training=True)

    test.check_true(
        not torch.allclose(recon1, recon2),
        "different reconstructions with noise (stochastic)"
    )

    # Test noise is not added during eval
    model.eval()
    recon3, _ = model(x, training=False)
    recon4, _ = model(x, training=False)

    test.check_true(
        torch.allclose(recon3, recon4),
        "same reconstructions without noise (deterministic)"
    )

    return test.run()

# =============================================================================
# SECTION 6: Latent Space Exploration
# =============================================================================
@lesson.section("Latent Space Exploration")
def section_6():
    """
    USING THE LATENT SPACE
    ======================

    The latent code z is a learned compressed representation.
    We can use it for many tasks!

    Applications:
    1. Dimensionality reduction (like PCA, but nonlinear)
    2. Feature extraction for downstream tasks
    3. Similarity search
    4. World models: predict dynamics in latent space!
    """
    print("Exploring the Latent Space")
    print("=" * 50)

    # Create a simple autoencoder
    class SimpleAE(nn.Module):
        def __init__(self, latent_dim=16):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 784),
                nn.Sigmoid(),
            )

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z).view(-1, 1, 28, 28)

        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z

    model = SimpleAE(latent_dim=16)

    # Generate some data and encode it
    x = torch.rand(10, 1, 28, 28)
    _, latents = model(x)

    print("LATENT CODE ANALYSIS")
    print("-" * 50)
    print(f"Input shape:  {list(x.shape)}")
    print(f"Latent shape: {list(latents.shape)}")
    print(f"\nLatent statistics:")
    print(f"  Mean: {latents.mean().item():.4f}")
    print(f"  Std:  {latents.std().item():.4f}")
    print(f"  Min:  {latents.min().item():.4f}")
    print(f"  Max:  {latents.max().item():.4f}")

    print("\n" + "-" * 50)
    print("SIMILARITY SEARCH")
    print("-" * 50)

    def find_similar(query_latent, all_latents, top_k=3):
        """Find most similar items by latent distance."""
        distances = torch.norm(all_latents - query_latent, dim=1)
        _, indices = distances.topk(top_k, largest=False)
        return indices, distances[indices]

    # Find items similar to the first one
    query = latents[0:1]
    indices, distances = find_similar(query, latents, top_k=3)

    print("Finding items similar to query (index 0):")
    for idx, dist in zip(indices.tolist(), distances.tolist()):
        print(f"  Index {idx}: distance = {dist:.4f}")

    print("\nSimilarity in latent space = semantic similarity!")

    return SimpleAE, model

# -----------------------------------------------------------------------------
# QUIZ 3
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: Latent Space", points=1)
def quiz_3():
    if '--test' not in sys.argv:
        answer = ask_question(
            "Why is the latent space useful for world models?",
            [
                "It uses more memory",
                "Predicting dynamics in latent space is easier than in pixel space",
                "It makes images look better",
                "It speeds up image loading"
            ]
        )
        return answer == 1  # Easier to predict
    return True

# =============================================================================
# SECTION 7: Latent Space Interpolation
# =============================================================================
@lesson.section("Latent Space Interpolation")
def section_7():
    """
    INTERPOLATION IN LATENT SPACE
    =============================

    We can smoothly transition between two items by:
    1. Encode both to latent space
    2. Linearly interpolate between latent codes
    3. Decode the interpolated codes

    If interpolations are smooth, the latent space is well-structured!
    """
    print("Latent Space Interpolation")
    print("=" * 50)

    class SimpleAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Linear(64, 784),
                nn.Sigmoid(),
            )

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z).view(-1, 1, 28, 28)

    model = SimpleAE()

    def interpolate(model, z1, z2, steps=5):
        """Generate items by interpolating between two latent codes."""
        model.eval()
        interpolations = []

        with torch.no_grad():
            for alpha in torch.linspace(0, 1, steps):
                z = z1 * (1 - alpha) + z2 * alpha
                img = model.decode(z)
                interpolations.append(img)

        return torch.cat(interpolations, dim=0)

    print("INTERPOLATION DEMO")
    print("-" * 50)

    # Create two different inputs
    x1 = torch.rand(1, 1, 28, 28)
    x2 = torch.rand(1, 1, 28, 28)

    # Encode
    z1 = model.encode(x1.flatten(1))
    z2 = model.encode(x2.flatten(1))

    print(f"z1 shape: {list(z1.shape)}")
    print(f"z2 shape: {list(z2.shape)}")

    # Interpolate
    interp_images = interpolate(model, z1, z2, steps=5)
    print(f"\nInterpolated images: {list(interp_images.shape)}")

    print("\nInterpolation formula:")
    print("  z = z1 * (1 - alpha) + z2 * alpha")
    print("  where alpha goes from 0 to 1")

    print("""
    VISUALIZATION:

    z1 ────────────────────> z2
    α=0    α=0.25   α=0.5   α=0.75   α=1.0
    img1   interp   interp  interp   img2

    Smooth interpolations = continuous latent space!
    """)

    return interpolate

# -----------------------------------------------------------------------------
# EXERCISE 4: Interpolation
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Latent Interpolation", points=1)
def exercise_4():
    """Implement latent space interpolation."""
    test = ExerciseTest("Interpolation")

    class TinyAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(10, 4)
            self.decoder = nn.Linear(4, 10)

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z)

    model = TinyAE()

    def interpolate(model, z1, z2, steps):
        """YOUR TASK: Implement interpolation."""
        results = []
        for alpha in torch.linspace(0, 1, steps):
            z = z1 * (1 - alpha) + z2 * alpha
            decoded = model.decode(z)
            results.append(decoded)
        return torch.cat(results, dim=0)

    # Test
    z1 = torch.randn(1, 4)
    z2 = torch.randn(1, 4)
    interp = interpolate(model, z1, z2, steps=5)

    test.check_shape(interp, (5, 10), "interpolation output shape")

    # Check endpoints
    with torch.no_grad():
        start = model.decode(z1)
        end = model.decode(z2)
        test.check_true(
            torch.allclose(interp[0], start.squeeze(), atol=1e-5),
            "first interpolation is start point"
        )
        test.check_true(
            torch.allclose(interp[-1], end.squeeze(), atol=1e-5),
            "last interpolation is end point"
        )

    return test.run()

# =============================================================================
# SECTION 8: Autoencoders for World Models
# =============================================================================
@lesson.section("Autoencoders for World Models")
def section_8():
    """
    WHY AUTOENCODERS MATTER FOR WORLD MODELS
    ========================================

    World models need to:
    1. Process high-dimensional observations (images)
    2. Predict future states
    3. Plan in an efficient representation

    Autoencoders provide the VISION component!
    """
    print("Autoencoders in World Models")
    print("=" * 50)

    print("""
    WORLD MODEL ARCHITECTURE
    ========================

    ┌──────────────────────────────────────────────────┐
    │                  WORLD MODEL                      │
    │                                                   │
    │  Observation ───> ┌────────────┐ ───> z_t        │
    │   (image)         │  ENCODER   │   (latent)      │
    │                   │ (from AE)  │                 │
    │                   └────────────┘                 │
    │                         │                        │
    │                         v                        │
    │  Previous z ────> ┌────────────┐                │
    │  + Action         │  DYNAMICS  │ ───> z_{t+1}   │
    │                   │   MODEL    │                │
    │                   └────────────┘                │
    │                         │                        │
    │                         v                        │
    │                   ┌────────────┐ ───> pred      │
    │                   │  DECODER   │   (image)      │
    │                   │ (from AE)  │                │
    │                   └────────────┘                │
    └──────────────────────────────────────────────────┘

    The autoencoder provides:
    - ENCODER: Compress observation to latent z
    - DECODER: Reconstruct observation from z (for visualization)
    """)

    print("-" * 50)
    print("WHY PREDICT IN LATENT SPACE?")
    print("-" * 50)

    print("""
    PIXEL SPACE (BAD):
    - Predicting 64x64x3 = 12,288 values
    - Tiny pixel errors compound
    - Hard to capture semantics

    LATENT SPACE (GOOD):
    - Predicting 32-256 values
    - Captures semantic structure
    - Much easier to model!

    Example:
    - Pixel: "predict color of pixel (32, 45)"
    - Latent: "predict position/velocity of object"
    """)

    print("\n" + "-" * 50)
    print("LIMITATION: VANILLA AUTOENCODERS")
    print("-" * 50)

    print("""
    Problems with vanilla autoencoders:
    1. Latent space may have "holes"
    2. Can't sample new valid latent codes
    3. No probabilistic interpretation

    SOLUTION: Variational Autoencoders (VAEs)!
    - Structured, continuous latent space
    - Can sample new data
    - Probabilistic - captures uncertainty

    NEXT LESSON: VAEs for world models!
    """)

# -----------------------------------------------------------------------------
# FINAL CHALLENGE
# -----------------------------------------------------------------------------
@lesson.exercise("Final Challenge: Complete Autoencoder", points=2)
def final_challenge():
    """
    Build a complete convolutional autoencoder for images!

    Requirements:
    - Input: 1x28x28 images
    - Latent dimension: 32
    - Must include encode() and decode() methods
    """
    test = ExerciseTest("Complete Autoencoder")

    class CompleteAutoencoder(nn.Module):
        def __init__(self, latent_dim=32):
            super().__init__()
            self.latent_dim = latent_dim

            # Encoder
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28 -> 14
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14 -> 7
                nn.ReLU(),
                nn.Flatten(),
            )
            self.encoder_fc = nn.Linear(64 * 7 * 7, latent_dim)

            # Decoder
            self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)
            self.decoder_conv = nn.Sequential(
                nn.Unflatten(1, (64, 7, 7)),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7 -> 14
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # 14 -> 28
                nn.Sigmoid(),
            )

        def encode(self, x):
            x = self.encoder_conv(x)
            z = self.encoder_fc(x)
            return z

        def decode(self, z):
            x = self.decoder_fc(z)
            x = self.decoder_conv(x)
            return x

        def forward(self, x):
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z

    # Create and test
    model = CompleteAutoencoder(latent_dim=32)

    # Test shapes
    x = torch.randn(4, 1, 28, 28)
    x_recon, z = model(x)

    test.check_shape(z, (4, 32), "latent shape is (batch, 32)")
    test.check_shape(x_recon, (4, 1, 28, 28), "reconstruction shape matches input")

    # Test encode/decode methods
    z_encoded = model.encode(x)
    x_decoded = model.decode(z_encoded)

    test.check_shape(z_encoded, (4, 32), "encode() returns correct shape")
    test.check_shape(x_decoded, (4, 1, 28, 28), "decode() returns correct shape")

    # Verify output range
    test.check_true(
        x_recon.min() >= 0 and x_recon.max() <= 1,
        "output is in [0, 1] range"
    )

    if test.run():
        print("\nExcellent! Your autoencoder is complete!")
        print("This architecture can:")
        print("- Compress images 784 -> 32 (24x compression)")
        print("- Reconstruct images from latent codes")
        print("- Serve as the vision component for world models")
        return True
    return False

# =============================================================================
# MAIN
# =============================================================================
def main():
    """Run the lesson."""
    if '--test' in sys.argv:
        # Test mode
        results = []
        results.append(("Quiz 1", quiz_1()))
        results.append(("Exercise 1", exercise_1()))
        results.append(("Quiz 2", quiz_2()))
        results.append(("Exercise 2", exercise_2()))
        results.append(("Exercise 3", exercise_3()))
        results.append(("Quiz 3", quiz_3()))
        results.append(("Exercise 4", exercise_4()))
        results.append(("Final Challenge", final_challenge()))

        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        passed = sum(1 for _, r in results if r)
        total = len(results)
        for name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status}: {name}")
        print(f"\nTotal: {passed}/{total} tests passed")
        return passed == total
    else:
        # Interactive mode
        lesson.run_interactive([
            section_1,
            quiz_1,
            section_2,
            exercise_1,
            section_3,
            quiz_2,
            section_4,
            exercise_2,
            section_5,
            exercise_3,
            section_6,
            quiz_3,
            section_7,
            exercise_4,
            section_8,
            final_challenge,
        ])

        show_progress()

        print("\n" + "=" * 60)
        print("LESSON 9 COMPLETE!")
        print("=" * 60)
        print("""
        KEY TAKEAWAYS:

        1. Autoencoders compress data through a bottleneck
           - Encoder: input -> latent code
           - Decoder: latent code -> reconstruction

        2. Convolutional autoencoders work better for images
           - Preserve spatial structure
           - Fewer parameters

        3. Denoising autoencoders learn robust features
           - Add noise to input, reconstruct clean output

        4. Latent space enables:
           - Dimensionality reduction
           - Similarity search
           - Smooth interpolation

        5. World models use autoencoders for:
           - Compressing observations to latent codes
           - Predicting dynamics in latent space

        LIMITATION: Vanilla autoencoders have unstructured latent space

        NEXT: Lesson 10 - Variational Autoencoders (VAEs)
        Probabilistic latent spaces for sampling and generation!
        """)

if __name__ == "__main__":
    main()
