# =============================================================================
# LESSON 10: Variational Autoencoders (VAE)
# =============================================================================
# VAEs are the backbone of many world models. They learn probabilistic
# latent representations that are smooth, continuous, and generative.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# THE CONCEPT: From AE to VAE
# -----------------------------------------------------------------------------
"""
AUTOENCODER:
    z = encoder(x)           # Deterministic latent code
    x' = decoder(z)          # Reconstruction

VARIATIONAL AUTOENCODER:
    μ, σ = encoder(x)        # Learn distribution parameters
    z ~ N(μ, σ²)             # SAMPLE from the distribution
    x' = decoder(z)          # Reconstruction

KEY INSIGHT:
- Instead of mapping to a POINT in latent space
- We map to a DISTRIBUTION in latent space
- This makes the latent space smooth and continuous

WHY THIS MATTERS FOR WORLD MODELS:
1. Can SAMPLE new latent codes (generative!)
2. Latent space is regularized (no gaps or discontinuities)
3. Uncertainty quantification built-in
4. Ha & Schmidhuber's World Model uses a VAE for vision
"""

# -----------------------------------------------------------------------------
# STEP 1: Load Data
# -----------------------------------------------------------------------------
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Training samples: {len(train_dataset)}")

# -----------------------------------------------------------------------------
# STEP 2: VAE Architecture
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("VAE ARCHITECTURE")
print("=" * 60)

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: x -> hidden -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Two separate heads for mu and logvar
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z -> hidden -> x
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)  # log(σ²) for numerical stability
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick:
        z = μ + σ * ε, where ε ~ N(0, 1)

        This allows gradients to flow through the sampling!
        (Can't backprop through random sampling directly)
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²))
        eps = torch.randn_like(std)    # ε ~ N(0, 1)
        z = mu + std * eps
        return z

    def decode(self, z):
        """Decode latent code to reconstruction."""
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

model = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
print(model)
print(f"\nLatent dimension: {model.latent_dim}")

# -----------------------------------------------------------------------------
# STEP 3: VAE Loss Function (ELBO)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("VAE LOSS FUNCTION")
print("=" * 60)

"""
VAE LOSS = Reconstruction Loss + KL Divergence

1. RECONSTRUCTION LOSS:
   - How well can we reconstruct the input?
   - Binary cross-entropy or MSE

2. KL DIVERGENCE:
   - Regularizes latent space toward N(0, I)
   - Prevents encoder from just memorizing data
   - KL(q(z|x) || p(z)) where p(z) = N(0, I)

KL for Gaussians has closed form:
   KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
"""

def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    ELBO loss for VAE.

    Args:
        x_recon: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL term (beta-VAE)

    Returns:
        Total loss, reconstruction loss, KL loss
    """
    # Reconstruction loss (per sample, then mean over batch)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL divergence (closed form for Gaussian)
    # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss

# Quick test
x = torch.rand(32, 784)
x_recon, mu, logvar, z = model(x)
loss, recon, kl = vae_loss(x_recon, x, mu, logvar)
print(f"Total loss: {loss.item():.2f}")
print(f"Recon loss: {recon.item():.2f}")
print(f"KL loss:    {kl.item():.2f}")

# -----------------------------------------------------------------------------
# STEP 4: Training the VAE
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING VAE")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(input_dim=784, hidden_dim=400, latent_dim=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_recon = 0
    train_kl = 0

    for images, _ in train_loader:
        images = images.view(-1, 784).to(device)

        # Forward
        x_recon, mu, logvar, z = model(images)
        loss, recon, kl = vae_loss(x_recon, images, mu, logvar)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_recon += recon.item()
        train_kl += kl.item()

    n = len(train_loader.dataset)
    print(f"Epoch {epoch+1:2d}/{num_epochs} | "
          f"Loss: {train_loss/n:.2f} | "
          f"Recon: {train_recon/n:.2f} | "
          f"KL: {train_kl/n:.2f}")

# -----------------------------------------------------------------------------
# STEP 5: Generating New Samples
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("GENERATING NEW SAMPLES")
print("=" * 60)

model.eval()
with torch.no_grad():
    # Sample from prior p(z) = N(0, I)
    z_samples = torch.randn(16, model.latent_dim).to(device)

    # Decode to generate images
    generated = model.decode(z_samples)

print(f"Sampled {z_samples.shape[0]} latent codes from N(0, I)")
print(f"Generated {generated.shape[0]} images")
print("This is the GENERATIVE power of VAEs!")

# -----------------------------------------------------------------------------
# STEP 6: Latent Space Interpolation
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("LATENT SPACE INTERPOLATION")
print("=" * 60)

def interpolate_vae(model, x1, x2, steps=10):
    """Interpolate between two images in latent space."""
    model.eval()
    with torch.no_grad():
        # Encode both images
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)

        # Interpolate in latent space
        interpolations = []
        for alpha in torch.linspace(0, 1, steps):
            z = mu1 * (1 - alpha) + mu2 * alpha
            img = model.decode(z)
            interpolations.append(img)

        return torch.cat(interpolations, dim=0)

# Test interpolation
test_images, _ = next(iter(test_loader))
test_images = test_images.view(-1, 784).to(device)

interp = interpolate_vae(model, test_images[0:1], test_images[1:1], steps=8)
print(f"Created {interp.shape[0]} interpolation steps")
print("VAE latent spaces are SMOOTH - interpolations make sense!")

# -----------------------------------------------------------------------------
# STEP 7: Convolutional VAE (for Images)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CONVOLUTIONAL VAE")
print("=" * 60)

class ConvVAE(nn.Module):
    """VAE with convolutional encoder/decoder for images."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28->14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14->7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 7->4
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0),  # 4->7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # 14->28
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 128, 4, 4)
        return self.decoder_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

conv_vae = ConvVAE(latent_dim=32)
print(conv_vae)

# Test
x = torch.randn(4, 1, 28, 28)
x_recon, mu, logvar, z = conv_vae(x)
print(f"\nInput:  {x.shape}")
print(f"Latent: {z.shape}")
print(f"Output: {x_recon.shape}")

# -----------------------------------------------------------------------------
# STEP 8: Beta-VAE (Disentangled Representations)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("BETA-VAE FOR DISENTANGLEMENT")
print("=" * 60)

"""
BETA-VAE: Simply increase the weight of KL term!

Loss = Recon + β * KL

β > 1: Stronger regularization -> more disentangled features
       Each latent dimension captures independent factors
       e.g., one dimension for rotation, another for size

β = 1: Standard VAE

This is useful for world models because disentangled
representations are easier to interpret and predict!
"""

def train_beta_vae(model, loader, optimizer, beta=4.0):
    """Train with beta-VAE loss."""
    model.train()
    for images, _ in loader:
        images = images.to(device)
        x_recon, mu, logvar, z = model(images)
        loss, _, _ = vae_loss(
            x_recon.view(-1, 784),
            images.view(-1, 784),
            mu, logvar,
            beta=beta  # Higher beta = more disentanglement
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Beta-VAE uses β > 1 for disentangled representations")
print("Higher β = more regularization = more independent latent dimensions")

# -----------------------------------------------------------------------------
# KEY CONCEPTS FOR WORLD MODELS
# -----------------------------------------------------------------------------
"""
WHY VAE IS CENTRAL TO WORLD MODELS:

1. THE VISION MODEL (V):
   - In Ha & Schmidhuber's World Model architecture:
   - VAE encodes observations: z_t = V(o_t) (actually samples from q(z|o))
   - Latent z captures essential features of observation

2. COMPACT REPRESENTATIONS:
   - High-dim observation (64x64 image = 12,288 dims)
   - Compressed to latent (32-256 dims)
   - Dynamics model works in this compact space

3. GENERATIVE CAPABILITY:
   - Can "imagine" observations by decoding latent predictions
   - Useful for visualization and debugging
   - Can dream/simulate without real environment

4. PROBABILISTIC NATURE:
   - Latent distribution captures uncertainty
   - Important for stochastic environments
   - Can sample multiple possible outcomes

5. SMOOTH LATENT SPACE:
   - KL regularization ensures continuity
   - Small changes in z = small changes in decoded observation
   - Makes prediction/interpolation meaningful

WORLD MODEL ARCHITECTURE:
    Observation o_t
         |
         v
    [VAE Encoder] -> μ, σ
         |
         v (sample)
    Latent z_t
         |
    [Memory RNN] -> h_t (incorporates action a_t)
         |
         v
    [Controller] -> action a_{t+1}

NEXT: Attention mechanisms for more powerful sequence modeling.
"""

# -----------------------------------------------------------------------------
# EXERCISE:
# 1. Train ConvVAE on MNIST and visualize generated digits
# 2. Implement conditional VAE (CVAE) - condition on digit class
# 3. Train with different β values and visualize latent space
# 4. Implement VQ-VAE (discrete latent codes)
# 5. Try on colored images (CIFAR-10, adjust for 3 channels)
# -----------------------------------------------------------------------------
