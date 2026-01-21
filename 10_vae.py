# =============================================================================
# LESSON 10: Variational Autoencoders (VAE)
# =============================================================================
# VAEs are the backbone of many world models. They learn probabilistic
# latent representations that are smooth, continuous, and generative.
#
# Run interactively: python 10_vae.py
# Run tests only:    python 10_vae.py --test
# =============================================================================

import sys
sys.path.insert(0, '.')
from utils.lesson_utils import LessonRunner, ExerciseTest, ask_question, show_progress

import torch
import torch.nn as nn
import torch.nn.functional as F

# Create the lesson runner
lesson = LessonRunner("Lesson 10: Variational Autoencoders", total_points=10)

# =============================================================================
# SECTION 1: From Autoencoder to VAE
# =============================================================================
@lesson.section("From Autoencoder to VAE")
def section_1():
    """
    THE KEY INSIGHT: PROBABILISTIC LATENT SPACE
    ===========================================

    AUTOENCODER:
        z = encoder(x)           # Deterministic latent code
        x' = decoder(z)          # Reconstruction

    VARIATIONAL AUTOENCODER:
        mu, sigma = encoder(x)   # Learn distribution parameters
        z ~ N(mu, sigma^2)       # SAMPLE from the distribution
        x' = decoder(z)          # Reconstruction

    Instead of mapping to a POINT in latent space,
    we map to a DISTRIBUTION in latent space!

    WHY THIS MATTERS:
    =================
    1. Can SAMPLE new latent codes (generative!)
    2. Latent space is regularized (no gaps)
    3. Uncertainty quantification built-in
    4. Ha & Schmidhuber's World Model uses VAE for vision
    """
    print("From Autoencoder to Variational Autoencoder")
    print("=" * 50)

    print("""
    AUTOENCODER (Lesson 9):
    =======================
    Input x ──> Encoder ──> z (point) ──> Decoder ──> x'

    The latent code z is a single point.
    Problem: Latent space may have "holes" - not all z values are meaningful.


    VARIATIONAL AUTOENCODER:
    ========================
    Input x ──> Encoder ──> (mu, sigma) ──> Sample z ──> Decoder ──> x'

    The encoder outputs distribution PARAMETERS.
    We SAMPLE z from this distribution.

    Key benefits:
    - Fill the entire latent space with valid codes
    - Can generate NEW data by sampling z ~ N(0, I)
    - Smooth interpolations guaranteed
    """)

    print("-" * 50)
    print("VISUAL COMPARISON")
    print("-" * 50)

    print("""
    AUTOENCODER LATENT SPACE:
    ┌─────────────────────────┐
    │    *        *    *      │
    │  *     [holes]    *     │  <- Sparse, gaps everywhere
    │       *        *        │     Can't sample randomly
    │    *       *            │
    └─────────────────────────┘

    VAE LATENT SPACE:
    ┌─────────────────────────┐
    │  . . . . . . . . . .    │
    │  . . . . . . . . . .    │  <- Dense, continuous
    │  . . . . . . . . . .    │     Can sample from N(0, I)
    │  . . . . . . . . . .    │
    └─────────────────────────┘
    """)

    # Simple demonstration
    print("-" * 50)
    print("SAMPLING DEMONSTRATION")
    print("-" * 50)

    # Autoencoder: deterministic
    ae_latent = torch.tensor([1.0, 2.0])  # Fixed point
    print(f"AE latent code:  {ae_latent.tolist()}  (always the same)")

    # VAE: probabilistic
    mu = torch.tensor([1.0, 2.0])
    sigma = torch.tensor([0.5, 0.5])
    vae_sample1 = mu + sigma * torch.randn(2)
    vae_sample2 = mu + sigma * torch.randn(2)
    print(f"VAE sample 1:    {vae_sample1.tolist()}")
    print(f"VAE sample 2:    {vae_sample2.tolist()}")
    print("VAE samples different values from the same distribution!")

# -----------------------------------------------------------------------------
# QUIZ 1
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: VAE vs AE", points=1)
def quiz_1():
    if '--test' not in sys.argv:
        answer = ask_question(
            "What does a VAE encoder output instead of a single latent code?",
            [
                "Two latent codes",
                "Parameters of a distribution (mean and variance)",
                "A larger latent vector",
                "The reconstruction directly"
            ]
        )
        return answer == 1  # Distribution parameters
    return True

# =============================================================================
# SECTION 2: VAE Architecture
# =============================================================================
@lesson.section("VAE Architecture")
def section_2():
    """
    VAE ARCHITECTURE
    ================

    The encoder outputs TWO vectors:
    - mu (mean): Center of the distribution
    - logvar (log variance): Spread of the distribution

    We use log(variance) instead of variance for numerical stability.
    """
    print("Building the VAE Architecture")
    print("=" * 50)

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
            logvar = self.fc_logvar(h)  # log(sigma^2)
            return mu, logvar

        def decode(self, z):
            """Decode latent code to reconstruction."""
            return self.decoder(z)

        def forward(self, x):
            mu, logvar = self.encode(x)
            # We'll add reparameterization in the next section
            z = mu  # Placeholder for now
            x_recon = self.decode(z)
            return x_recon, mu, logvar, z

    model = VAE(input_dim=784, hidden_dim=256, latent_dim=20)

    print("VAE Architecture:")
    print("-" * 50)
    print("ENCODER:")
    print("  Input (784) -> Hidden (256) -> Hidden (256)")
    print("                    |")
    print("              ┌─────┴─────┐")
    print("              v           v")
    print("          fc_mu (20)  fc_logvar (20)")
    print()
    print("DECODER:")
    print("  Latent (20) -> Hidden (256) -> Hidden (256) -> Output (784)")

    print("\n" + "-" * 50)
    print("FORWARD PASS")
    print("-" * 50)

    x = torch.randn(4, 784)
    x_recon, mu, logvar, z = model(x)

    print(f"Input shape:     {list(x.shape)}")
    print(f"mu shape:        {list(mu.shape)}")
    print(f"logvar shape:    {list(logvar.shape)}")
    print(f"z (latent):      {list(z.shape)}")
    print(f"Reconstruction:  {list(x_recon.shape)}")

    return VAE, model

# -----------------------------------------------------------------------------
# EXERCISE 1: VAE Encoder
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Build VAE Encoder", points=1)
def exercise_1():
    """Build the encoder part of a VAE."""
    test = ExerciseTest("VAE Encoder")

    class VAEEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )
            # YOUR TASK: Create mu and logvar heads
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        def forward(self, x):
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

    # Test
    encoder = VAEEncoder(input_dim=100, hidden_dim=50, latent_dim=10)
    x = torch.randn(4, 100)
    mu, logvar = encoder(x)

    test.check_shape(mu, (4, 10), "mu shape")
    test.check_shape(logvar, (4, 10), "logvar shape")
    test.check_true(
        not torch.allclose(mu, logvar),
        "mu and logvar are different"
    )

    return test.run()

# =============================================================================
# SECTION 3: The Reparameterization Trick
# =============================================================================
@lesson.section("The Reparameterization Trick")
def section_3():
    """
    THE REPARAMETERIZATION TRICK
    ============================

    Problem: We need to SAMPLE from q(z|x) = N(mu, sigma^2)
    But we can't backpropagate through random sampling!

    Solution: Reparameterization Trick
        z = mu + sigma * epsilon
        where epsilon ~ N(0, 1)

    This moves the randomness to epsilon, which doesn't need gradients.
    Gradients flow through mu and sigma!

    Since we use logvar (log variance):
        sigma = exp(0.5 * logvar)
        z = mu + sigma * epsilon
    """
    print("The Reparameterization Trick")
    print("=" * 50)

    print("""
    PROBLEM: Can't Backpropagate Through Sampling
    =============================================

    z ~ N(mu, sigma^2)  # Sample operation

    How do you compute d(z)/d(mu)?
    The sampling is random - no gradient!


    SOLUTION: Reparameterization
    ============================

    Instead of:  z ~ N(mu, sigma^2)

    Do this:     epsilon ~ N(0, 1)       <- Random, but no gradient needed
                 z = mu + sigma * epsilon <- Deterministic transform

    Now we can compute:
        d(z)/d(mu) = 1
        d(z)/d(sigma) = epsilon
    """)

    def reparameterize(mu, logvar):
        """
        Sample z = mu + sigma * epsilon
        where sigma = exp(0.5 * logvar)
        """
        std = torch.exp(0.5 * logvar)  # sigma = sqrt(exp(logvar))
        eps = torch.randn_like(std)     # epsilon ~ N(0, 1)
        z = mu + std * eps
        return z

    print("-" * 50)
    print("DEMONSTRATION")
    print("-" * 50)

    mu = torch.tensor([[1.0, 2.0]])
    logvar = torch.tensor([[0.0, 0.0]])  # log(1) = 0, so sigma = 1

    print(f"mu:      {mu.squeeze().tolist()}")
    print(f"logvar:  {logvar.squeeze().tolist()}")
    print(f"sigma:   {torch.exp(0.5 * logvar).squeeze().tolist()}")

    # Sample multiple times
    print("\nSampling 5 times:")
    for i in range(5):
        z = reparameterize(mu, logvar)
        print(f"  z_{i+1}: {z.squeeze().tolist()}")

    print("\nSamples vary around mu with spread controlled by sigma!")

    return reparameterize

# -----------------------------------------------------------------------------
# QUIZ 2
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: Reparameterization", points=1)
def quiz_2():
    if '--test' not in sys.argv:
        answer = ask_question(
            "Why is the reparameterization trick necessary in VAEs?",
            [
                "To make training faster",
                "To allow backpropagation through the sampling operation",
                "To reduce memory usage",
                "To improve image quality"
            ]
        )
        return answer == 1  # Backprop through sampling
    return True

# =============================================================================
# SECTION 4: VAE Loss Function (ELBO)
# =============================================================================
@lesson.section("VAE Loss Function (ELBO)")
def section_4():
    """
    VAE LOSS FUNCTION
    =================

    Loss = Reconstruction Loss + KL Divergence

    1. RECONSTRUCTION LOSS:
       How well can we reconstruct the input?
       Uses MSE or Binary Cross-Entropy

    2. KL DIVERGENCE:
       Regularizes latent space toward N(0, I)
       Forces the encoder to use the entire latent space

    KL for Gaussians has a closed form:
       KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    print("VAE Loss: ELBO")
    print("=" * 50)

    print("""
    ELBO = Evidence Lower Bound
    ===========================

    We want to maximize log p(x), but it's intractable.
    Instead, maximize a lower bound:

    ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
           ^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^
           Reconstruction   Regularization
           (want to maximize)  (want to minimize)

    Rearranging for minimization:

    Loss = -E[log p(x|z)] + KL(q(z|x) || p(z))
         = Reconstruction_Loss + KL_Loss
    """)

    def vae_loss(x_recon, x, mu, logvar, beta=1.0):
        """
        VAE Loss = Reconstruction + beta * KL

        Args:
            x_recon: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            beta: Weight for KL term (beta > 1 for beta-VAE)
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        # KL divergence (closed form for Gaussian)
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    print("-" * 50)
    print("LOSS COMPONENTS")
    print("-" * 50)

    print("""
    RECONSTRUCTION LOSS:
    - Measures: How different is x' from x?
    - Goal: Make decoder produce accurate reconstructions
    - Formula: MSE(x, x') or BCE(x, x')

    KL DIVERGENCE:
    - Measures: How different is q(z|x) from N(0, I)?
    - Goal: Keep latent distributions close to standard normal
    - Formula: -0.5 * sum(1 + log(var) - mu^2 - var)

    The KL term is crucial! It:
    - Prevents encoder from making variance = 0 (just memorizing)
    - Ensures latent space is continuous and smooth
    - Allows sampling z ~ N(0, I) at test time
    """)

    # Demonstration
    print("-" * 50)
    print("DEMONSTRATION")
    print("-" * 50)

    x = torch.rand(4, 784)
    x_recon = torch.rand(4, 784)
    mu = torch.randn(4, 20)
    logvar = torch.zeros(4, 20)

    total, recon, kl = vae_loss(x_recon, x, mu, logvar)

    print(f"Reconstruction loss: {recon.item():.2f}")
    print(f"KL divergence:       {kl.item():.2f}")
    print(f"Total loss:          {total.item():.2f}")

    return vae_loss

# -----------------------------------------------------------------------------
# EXERCISE 2: Implement KL Divergence
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: KL Divergence", points=1)
def exercise_2():
    """Implement the KL divergence term for VAE."""
    test = ExerciseTest("KL Divergence")

    def kl_divergence(mu, logvar):
        """
        Compute KL divergence between q(z|x) and p(z).
        q(z|x) = N(mu, exp(logvar))
        p(z) = N(0, I)

        Formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl

    # Test 1: When mu=0 and logvar=0, KL should be 0
    mu_zero = torch.zeros(4, 10)
    logvar_zero = torch.zeros(4, 10)
    kl_zero = kl_divergence(mu_zero, logvar_zero)
    test.check_true(
        abs(kl_zero.item()) < 0.01,
        f"KL ~ 0 when q = p (got {kl_zero.item():.4f})"
    )

    # Test 2: KL should be positive for non-standard distributions
    mu_nonzero = torch.ones(4, 10) * 2
    kl_positive = kl_divergence(mu_nonzero, logvar_zero)
    test.check_true(
        kl_positive.item() > 0,
        "KL is positive when mu != 0"
    )

    return test.run()

# =============================================================================
# SECTION 5: Complete VAE Implementation
# =============================================================================
@lesson.section("Complete VAE Implementation")
def section_5():
    """
    PUTTING IT ALL TOGETHER
    =======================

    Complete VAE with:
    - Encoder outputting mu and logvar
    - Reparameterization for sampling
    - Decoder for reconstruction
    - ELBO loss for training
    """
    print("Complete VAE Implementation")
    print("=" * 50)

    class VAE(nn.Module):
        def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
            super().__init__()
            self.latent_dim = latent_dim

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid(),
            )

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar, z

    model = VAE(input_dim=784, hidden_dim=256, latent_dim=20)

    print("Complete VAE created!")
    print(f"  Input dimension:  784")
    print(f"  Hidden dimension: 256")
    print(f"  Latent dimension: 20")

    # Training demo
    print("\n" + "-" * 50)
    print("TRAINING DEMO")
    print("-" * 50)

    def vae_loss(x_recon, x, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create synthetic data
    train_data = torch.rand(100, 784)

    print("Training for 5 epochs...")
    for epoch in range(5):
        optimizer.zero_grad()
        x_recon, mu, logvar, z = model(train_data)
        loss, recon, kl = vae_loss(x_recon, train_data, mu, logvar)
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch+1}: Loss={loss.item()/100:.2f}, "
              f"Recon={recon.item()/100:.2f}, KL={kl.item()/100:.2f}")

    return VAE, model

# -----------------------------------------------------------------------------
# EXERCISE 3: Train VAE
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Train VAE", points=1)
def exercise_3():
    """Train a VAE and verify loss decreases."""
    test = ExerciseTest("VAE Training")

    class SimpleVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(100, 50)
            self.fc_mu = nn.Linear(50, 10)
            self.fc_logvar = nn.Linear(50, 10)
            self.decoder = nn.Sequential(
                nn.Linear(10, 50),
                nn.ReLU(),
                nn.Linear(50, 100),
                nn.Sigmoid(),
            )

        def forward(self, x):
            h = F.relu(self.encoder(x))
            mu, logvar = self.fc_mu(h), self.fc_logvar(h)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            return self.decoder(z), mu, logvar

    model = SimpleVAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data = torch.rand(50, 100)

    # Compute initial loss
    with torch.no_grad():
        x_r, mu, lv = model(data)
        initial_loss = F.mse_loss(x_r, data, reduction='sum')
        initial_loss += -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())

    # Train
    for _ in range(50):
        optimizer.zero_grad()
        x_r, mu, lv = model(data)
        loss = F.mse_loss(x_r, data, reduction='sum')
        loss += -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        loss.backward()
        optimizer.step()

    # Final loss
    with torch.no_grad():
        x_r, mu, lv = model(data)
        final_loss = F.mse_loss(x_r, data, reduction='sum')
        final_loss += -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())

    test.check_true(
        final_loss.item() < initial_loss.item(),
        f"loss decreased ({initial_loss.item():.1f} -> {final_loss.item():.1f})"
    )

    return test.run()

# =============================================================================
# SECTION 6: Generating New Samples
# =============================================================================
@lesson.section("Generating New Samples")
def section_6():
    """
    THE GENERATIVE POWER OF VAEs
    ============================

    At test time, we can generate NEW data by:
    1. Sample z from the prior: z ~ N(0, I)
    2. Decode: x = decoder(z)

    This works because:
    - KL loss forces q(z|x) to be close to N(0, I)
    - Any z sampled from N(0, I) should decode to valid data
    """
    print("Generating New Samples")
    print("=" * 50)

    print("""
    GENERATION PROCESS
    ==================

    Training time:
        x -> Encoder -> (mu, sigma) -> Sample z -> Decoder -> x'

    Generation time:
        Sample z ~ N(0, I) -> Decoder -> NEW x!

    We skip the encoder entirely when generating!
    Just sample from the prior and decode.
    """)

    # Simple VAE for demonstration
    class SimpleVAE(nn.Module):
        def __init__(self, latent_dim=10):
            super().__init__()
            self.latent_dim = latent_dim
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 50),
                nn.ReLU(),
                nn.Linear(50, 100),
                nn.Sigmoid(),
            )

        def generate(self, num_samples):
            """Generate new samples from the prior."""
            z = torch.randn(num_samples, self.latent_dim)
            return self.decoder(z)

    model = SimpleVAE(latent_dim=10)

    print("-" * 50)
    print("GENERATION DEMONSTRATION")
    print("-" * 50)

    # Generate samples
    generated = model.generate(num_samples=5)

    print(f"Generated {generated.shape[0]} new samples")
    print(f"Each sample has shape: {generated.shape[1]}")

    print("\nSample statistics:")
    print(f"  Min value:  {generated.min().item():.4f}")
    print(f"  Max value:  {generated.max().item():.4f}")
    print(f"  Mean value: {generated.mean().item():.4f}")

    print("\nThis is the GENERATIVE power of VAEs!")
    print("We can create new data that looks like the training data.")

    return model

# -----------------------------------------------------------------------------
# QUIZ 3
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: Generation", points=1)
def quiz_3():
    if '--test' not in sys.argv:
        answer = ask_question(
            "When generating new samples from a trained VAE, where do we sample z from?",
            [
                "The encoder q(z|x)",
                "A uniform distribution",
                "The prior p(z) = N(0, I)",
                "The decoder"
            ]
        )
        return answer == 2  # Prior N(0, I)
    return True

# =============================================================================
# SECTION 7: Convolutional VAE
# =============================================================================
@lesson.section("Convolutional VAE")
def section_7():
    """
    CONVOLUTIONAL VAE FOR IMAGES
    ============================

    For images, we use convolutional layers:
    - Encoder: Conv2d with stride > 1 to downsample
    - Decoder: ConvTranspose2d to upsample
    """
    print("Convolutional VAE for Images")
    print("=" * 50)

    class ConvVAE(nn.Module):
        def __init__(self, latent_dim=32):
            super().__init__()
            self.latent_dim = latent_dim

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28 -> 14
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14 -> 7
                nn.ReLU(),
                nn.Flatten(),
            )
            self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
            self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

            # Decoder
            self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (64, 7, 7)),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7 -> 14
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # 14 -> 28
                nn.Sigmoid(),
            )

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)

        def decode(self, z):
            h = self.decoder_fc(z)
            return self.decoder(h)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar, z

    model = ConvVAE(latent_dim=32)

    print("ConvVAE Architecture:")
    print("-" * 50)
    print("ENCODER:")
    print("  (1, 28, 28) -> Conv2d -> (32, 14, 14)")
    print("  -> Conv2d -> (64, 7, 7) -> Flatten -> (3136)")
    print("  -> fc_mu (32), fc_logvar (32)")
    print("\nDECODER:")
    print("  (32) -> Linear -> (3136)")
    print("  -> Reshape (64, 7, 7) -> ConvT2d -> (32, 14, 14)")
    print("  -> ConvT2d -> (1, 28, 28)")

    # Test
    print("\n" + "-" * 50)
    print("FORWARD PASS")
    print("-" * 50)

    x = torch.randn(4, 1, 28, 28)
    x_recon, mu, logvar, z = model(x)

    print(f"Input:          {list(x.shape)}")
    print(f"Latent (z):     {list(z.shape)}")
    print(f"Reconstruction: {list(x_recon.shape)}")

    return ConvVAE, model

# -----------------------------------------------------------------------------
# EXERCISE 4: ConvVAE
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: ConvVAE Forward", points=1)
def exercise_4():
    """Test a ConvVAE forward pass."""
    test = ExerciseTest("ConvVAE Forward")

    class TinyConvVAE(nn.Module):
        def __init__(self, latent_dim=16):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.fc_mu = nn.Linear(16 * 14 * 14, latent_dim)
            self.fc_logvar = nn.Linear(16 * 14 * 14, latent_dim)
            self.decoder_fc = nn.Linear(latent_dim, 16 * 14 * 14)
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (16, 14, 14)),
                nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            h = self.encoder(x)
            mu, logvar = self.fc_mu(h), self.fc_logvar(h)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            h = self.decoder_fc(z)
            return self.decoder(h), mu, logvar, z

    model = TinyConvVAE(latent_dim=16)
    x = torch.randn(2, 1, 28, 28)
    x_recon, mu, logvar, z = model(x)

    test.check_shape(z, (2, 16), "latent shape")
    test.check_shape(x_recon, (2, 1, 28, 28), "reconstruction shape")
    test.check_true(
        x_recon.min() >= 0 and x_recon.max() <= 1,
        "reconstruction in [0, 1]"
    )

    return test.run()

# =============================================================================
# SECTION 8: VAE for World Models
# =============================================================================
@lesson.section("VAE for World Models")
def section_8():
    """
    VAE AS THE VISION COMPONENT
    ===========================

    In world models (Ha & Schmidhuber 2018):
    - VAE is the VISION model (V)
    - Encodes high-dimensional observations to compact latent codes
    - Enables prediction in latent space instead of pixel space
    """
    print("VAE for World Models")
    print("=" * 50)

    print("""
    WORLD MODEL ARCHITECTURE
    ========================

    ┌────────────────────────────────────────────────┐
    │                 WORLD MODEL                     │
    │                                                 │
    │  Observation ──> ┌─────────┐                   │
    │  (image)         │   VAE   │ ──> z_t           │
    │                  │ Encoder │   (latent)        │
    │                  └─────────┘                   │
    │                       │                        │
    │                       v                        │
    │  z_{t-1} ──────> ┌─────────┐                  │
    │  action  ──────> │   RNN   │ ──> h_t          │
    │                  │ Memory  │  (hidden state)  │
    │                  └─────────┘                   │
    │                       │                        │
    │                       v                        │
    │  (z_t, h_t) ────> ┌─────────┐                 │
    │                   │Controller│ ──> action     │
    │                   └─────────┘                  │
    └────────────────────────────────────────────────┘

    The VAE provides:
    - Compact representation z for efficient computation
    - Smooth latent space for meaningful interpolation
    - Probabilistic encoding (uncertainty!)
    - Generative capability for "imagination"
    """)

    print("-" * 50)
    print("WHY VAE INSTEAD OF AUTOENCODER?")
    print("-" * 50)

    print("""
    1. REGULARIZED LATENT SPACE
       - KL loss ensures no "holes" in latent space
       - Any z ~ N(0, I) decodes to valid observation
       - Important when RNN predicts future z values

    2. UNCERTAINTY QUANTIFICATION
       - sigma tells us how confident the encoding is
       - Useful for decision-making under uncertainty

    3. GENERATION / DREAMING
       - Can "imagine" observations by sampling z
       - Enables training in "dream" environment
       - See Dreamer, DreamerV2, DreamerV3

    4. STOCHASTICITY
       - World can be stochastic (same action, different outcomes)
       - Sampling captures this inherent randomness
    """)

    print("-" * 50)
    print("BETA-VAE FOR DISENTANGLEMENT")
    print("-" * 50)

    print("""
    Beta-VAE uses Loss = Recon + BETA * KL (where beta > 1)

    Higher beta = more regularization = more disentangled features

    Disentangled features are useful because:
    - One latent dimension might encode position
    - Another might encode velocity
    - Another might encode object type
    - Easier to interpret and predict!

    This is especially valuable for world models where we want
    to predict how specific aspects of the world change.
    """)

# -----------------------------------------------------------------------------
# FINAL CHALLENGE
# -----------------------------------------------------------------------------
@lesson.exercise("Final Challenge: Complete VAE", points=2)
def final_challenge():
    """
    Build a complete VAE with all components!

    Requirements:
    - Encoder: Input -> hidden -> mu, logvar
    - Reparameterization
    - Decoder: z -> hidden -> reconstruction
    - Loss function
    """
    test = ExerciseTest("Complete VAE")

    class CompleteVAE(nn.Module):
        def __init__(self, input_dim=784, hidden_dim=256, latent_dim=20):
            super().__init__()
            self.latent_dim = latent_dim

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid(),
            )

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar, z

        def loss(self, x_recon, x, mu, logvar):
            recon = F.mse_loss(x_recon, x, reduction='sum')
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon + kl, recon, kl

        def generate(self, num_samples):
            z = torch.randn(num_samples, self.latent_dim)
            return self.decode(z)

    # Test
    model = CompleteVAE(input_dim=784, hidden_dim=256, latent_dim=20)
    x = torch.rand(4, 784)

    # Test forward
    x_recon, mu, logvar, z = model(x)
    test.check_shape(mu, (4, 20), "mu shape")
    test.check_shape(z, (4, 20), "z shape")
    test.check_shape(x_recon, (4, 784), "reconstruction shape")

    # Test reparameterization is stochastic
    _, mu1, _, z1 = model(x)
    _, mu2, _, z2 = model(x)
    test.check_true(
        torch.allclose(mu1, mu2),
        "mu is deterministic (same input = same mu)"
    )
    test.check_true(
        not torch.allclose(z1, z2),
        "z is stochastic (same input = different z)"
    )

    # Test loss
    loss, recon, kl = model.loss(x_recon, x, mu, logvar)
    test.check_true(loss.item() > 0, "loss is positive")
    test.check_true(recon.item() > 0, "recon loss is positive")

    # Test generation
    generated = model.generate(num_samples=3)
    test.check_shape(generated, (3, 784), "generated shape")

    if test.run():
        print("\nExcellent! Your VAE is complete!")
        print("This VAE can:")
        print("- Encode inputs to probabilistic latent codes")
        print("- Generate new samples from the prior")
        print("- Serve as the vision component in world models")
        return True
    return False

# =============================================================================
# MAIN
# =============================================================================
def main():
    """Run the lesson."""
    if '--test' in sys.argv:
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
        print("LESSON 10 COMPLETE!")
        print("=" * 60)
        print("""
        KEY TAKEAWAYS:

        1. VAEs encode to distributions, not points
           - Encoder outputs mu and logvar
           - Sample z using reparameterization trick

        2. ELBO Loss = Reconstruction + KL Divergence
           - KL regularizes latent space toward N(0, I)

        3. Reparameterization trick enables backprop
           - z = mu + sigma * epsilon
           - Gradients flow through mu and sigma

        4. VAEs can GENERATE new data
           - Sample z ~ N(0, I), then decode

        5. World models use VAEs because:
           - Compact representations
           - Smooth latent space
           - Can "dream" / imagine

        NEXT: Lesson 11 - Attention Mechanisms
        Learn how to focus on relevant parts of sequences!
        """)

if __name__ == "__main__":
    main()
