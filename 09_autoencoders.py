# =============================================================================
# LESSON 9: Autoencoders - Learning Compressed Representations
# =============================================================================
# Autoencoders learn to compress data into a latent space and reconstruct it.
# This is crucial for world models: we need compact state representations!

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# THE CONCEPT: Autoencoder Architecture
# -----------------------------------------------------------------------------
"""
INPUT -> [ENCODER] -> LATENT CODE -> [DECODER] -> RECONSTRUCTION
  x   ->    E(x)   ->      z      ->    D(z)   ->      x'

Loss = ||x - x'||^2  (reconstruction error)

WHY AUTOENCODERS FOR WORLD MODELS?
1. Compress high-dimensional observations (images) into compact latent codes
2. Latent space is easier to model/predict than raw pixels
3. Denoising autoencoders learn robust representations

TYPES:
- Vanilla Autoencoder: Direct compression
- Denoising Autoencoder: Reconstruct from corrupted input
- Sparse Autoencoder: Enforce sparsity in latent code
- Variational Autoencoder (VAE): Probabilistic latent space (next lesson!)
"""

# -----------------------------------------------------------------------------
# STEP 1: Load MNIST Data
# -----------------------------------------------------------------------------
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Image shape: {train_dataset[0][0].shape}")  # (1, 28, 28)

# -----------------------------------------------------------------------------
# STEP 2: Simple Fully Connected Autoencoder
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FULLY CONNECTED AUTOENCODER")
print("=" * 60)

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
fc_ae = FCAutoencoder(input_dim=784, latent_dim=32)
print(f"Latent dimension: 32")
print(f"Compression ratio: {784/32:.1f}x")

# Count parameters
total_params = sum(p.numel() for p in fc_ae.parameters())
print(f"Total parameters: {total_params:,}")

# -----------------------------------------------------------------------------
# STEP 3: Convolutional Autoencoder (Better for Images!)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CONVOLUTIONAL AUTOENCODER")
print("=" * 60)

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: Image -> Latent
        self.encoder = nn.Sequential(
            # (1, 28, 28) -> (32, 14, 14)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # (32, 14, 14) -> (64, 7, 7)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # (64, 7, 7) -> (128, 4, 4)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Flatten: (128, 4, 4) -> 2048
            nn.Flatten(),

            # To latent
            nn.Linear(128 * 4 * 4, latent_dim),
        )

        # Decoder: Latent -> Image
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)

        self.decoder_conv = nn.Sequential(
            # (128, 4, 4) -> (64, 7, 7)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),

            # (64, 7, 7) -> (32, 14, 14)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            # (32, 14, 14) -> (1, 28, 28)
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 128, 4, 4)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

conv_ae = ConvAutoencoder(latent_dim=64)
print(conv_ae)

# Test shapes
x = torch.randn(4, 1, 28, 28)
x_recon, z = conv_ae(x)
print(f"\nInput shape:          {x.shape}")
print(f"Latent shape:         {z.shape}")
print(f"Reconstruction shape: {x_recon.shape}")

# -----------------------------------------------------------------------------
# STEP 4: Training the Autoencoder
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING CONVOLUTIONAL AUTOENCODER")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder(latent_dim=64).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()  # Reconstruction loss

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for images, _ in train_loader:  # We don't need labels!
        images = images.to(device)

        # Forward
        recon, _ = model(images)
        loss = criterion(recon, images)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.6f}")

# -----------------------------------------------------------------------------
# STEP 5: Evaluate Reconstruction Quality
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("EVALUATING RECONSTRUCTION")
print("=" * 60)

model.eval()
with torch.no_grad():
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)
    recon, latents = model(test_images)

    # Compute test reconstruction error
    test_loss = criterion(recon, test_images)
    print(f"Test reconstruction loss: {test_loss.item():.6f}")

    # Check latent statistics
    print(f"\nLatent code statistics:")
    print(f"  Shape: {latents.shape}")
    print(f"  Mean:  {latents.mean().item():.4f}")
    print(f"  Std:   {latents.std().item():.4f}")
    print(f"  Min:   {latents.min().item():.4f}")
    print(f"  Max:   {latents.max().item():.4f}")

# -----------------------------------------------------------------------------
# STEP 6: Denoising Autoencoder
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DENOISING AUTOENCODER")
print("=" * 60)

"""
DENOISING AUTOENCODER:
- Add noise to input
- Train to reconstruct clean output
- Forces learning of robust features, not just identity mapping
"""

class DenoisingAutoencoder(ConvAutoencoder):
    def __init__(self, latent_dim=64, noise_factor=0.3):
        super().__init__(latent_dim)
        self.noise_factor = noise_factor

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_factor
        noisy = x + noise
        return torch.clamp(noisy, 0., 1.)  # Keep in valid range

    def forward(self, x, add_noise=True):
        if add_noise and self.training:
            x_input = self.add_noise(x)
        else:
            x_input = x

        z = self.encode(x_input)
        x_recon = self.decode(z)
        return x_recon, z

dae = DenoisingAutoencoder(latent_dim=64, noise_factor=0.3)

# Training uses noisy input, clean target
# loss = criterion(dae(noisy_x), clean_x)

print("Denoising AE: Input is corrupted, target is clean")
print("This learns more robust representations!")

# -----------------------------------------------------------------------------
# STEP 7: Using Latent Space
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("USING THE LATENT SPACE")
print("=" * 60)

model.eval()

# Get latent codes for some images
with torch.no_grad():
    images, labels = next(iter(test_loader))
    images = images.to(device)
    _, latents = model(images)

# Latent codes can be used for:
print("LATENT SPACE APPLICATIONS:")
print("1. Dimensionality reduction (like PCA but nonlinear)")
print("2. Feature extraction for downstream tasks")
print("3. Similarity search (compare latent codes)")
print("4. World models: predict dynamics in latent space!")

# Example: Find similar images by latent distance
def find_similar(query_latent, all_latents, top_k=5):
    distances = torch.norm(all_latents - query_latent, dim=1)
    _, indices = distances.topk(top_k, largest=False)
    return indices

query = latents[0:1]
similar_indices = find_similar(query, latents, top_k=5)
print(f"\nQuery image label: {labels[0].item()}")
print(f"Similar images labels: {[labels[i].item() for i in similar_indices]}")

# -----------------------------------------------------------------------------
# STEP 8: Interpolation in Latent Space
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("LATENT SPACE INTERPOLATION")
print("=" * 60)

def interpolate(model, z1, z2, steps=10):
    """Generate images by interpolating between two latent codes."""
    model.eval()
    interpolations = []

    with torch.no_grad():
        for alpha in torch.linspace(0, 1, steps):
            z = z1 * (1 - alpha) + z2 * alpha
            img = model.decode(z)
            interpolations.append(img)

    return torch.cat(interpolations, dim=0)

# Interpolate between two digits
z1 = latents[0:1]  # First image
z2 = latents[10:11]  # Another image
interp_images = interpolate(model, z1, z2, steps=8)

print(f"Interpolated {interp_images.shape[0]} images between two latent codes")
print("Smooth interpolations = good latent space structure!")

# -----------------------------------------------------------------------------
# KEY CONCEPTS FOR WORLD MODELS
# -----------------------------------------------------------------------------
"""
WHY AUTOENCODERS MATTER FOR WORLD MODELS:

1. OBSERVATION ENCODING:
   - Raw observations (images) are high-dimensional
   - Encode to compact latent code z = E(observation)
   - Much easier to predict/model in latent space

2. THE "V" IN WORLD MODELS (Ha & Schmidhuber):
   - Vision model V encodes observations to latent codes
   - Typically uses a VAE (next lesson!)
   - z_t = V(observation_t)

3. RECONSTRUCTION = VERIFICATION:
   - Can decode latent states back to observations
   - Useful for visualization and debugging
   - "What does the model think the world looks like?"

4. LATENT DYNAMICS:
   - Instead of predicting next frame directly
   - Predict next latent code: z_{t+1} = f(z_t, a_t)
   - Then decode if needed: frame_{t+1} = D(z_{t+1})

LIMITATION OF VANILLA AUTOENCODERS:
- Latent space may not be continuous or structured
- Hard to sample new valid latent codes
- Solution: Variational Autoencoders (VAEs) - next lesson!

NEXT: VAEs for probabilistic, well-structured latent spaces.
"""

# -----------------------------------------------------------------------------
# EXERCISE:
# 1. Train on a different dataset (Fashion-MNIST or CIFAR-10)
# 2. Experiment with different latent dimensions (8, 16, 32, 64, 128)
# 3. Add skip connections (U-Net style autoencoder)
# 4. Implement sparse autoencoder with L1 regularization on latent codes
# 5. Build a classifier using latent codes as features
# -----------------------------------------------------------------------------
