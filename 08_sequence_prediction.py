# =============================================================================
# LESSON 8: Sequence Prediction and Time Series Forecasting
# =============================================================================
# Predicting the future - a core capability needed for world models.
# We'll learn to predict next values in sequences.

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# THE CONCEPT: Sequence Prediction
# -----------------------------------------------------------------------------
"""
Given: x_1, x_2, ..., x_t
Predict: x_{t+1} (or x_{t+1}, ..., x_{t+k} for multi-step)

This is fundamental to world models:
- Predict next observation given history
- Predict next state given current state + action

Approaches:
1. Autoregressive: Predict one step, feed back, repeat
2. Direct: Predict all future steps at once
3. Seq2Seq: Encoder-decoder for flexible horizons
"""

# -----------------------------------------------------------------------------
# STEP 1: Create Synthetic Time Series Data
# -----------------------------------------------------------------------------
print("=" * 60)
print("GENERATING SYNTHETIC DATA")
print("=" * 60)

def generate_sine_wave(n_samples, seq_length, n_features=1):
    """Generate sine waves with noise for prediction task."""
    X, y = [], []
    for _ in range(n_samples):
        # Random frequency and phase
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        noise = np.random.normal(0, 0.1, seq_length + 1)

        # Generate sequence
        t = np.linspace(0, 4 * np.pi, seq_length + 1)
        signal = np.sin(freq * t + phase) + noise

        X.append(signal[:-1])  # Input: all but last
        y.append(signal[1:])   # Target: all but first (shifted by 1)

    return np.array(X), np.array(y)

# Generate data
n_samples = 1000
seq_length = 50
X, y = generate_sine_wave(n_samples, seq_length)

# Convert to tensors
X = torch.FloatTensor(X).unsqueeze(-1)  # Add feature dimension
y = torch.FloatTensor(y).unsqueeze(-1)

print(f"Input shape:  {X.shape}")   # (samples, seq_len, features)
print(f"Target shape: {y.shape}")   # (samples, seq_len, features)

# Split data
train_X, test_X = X[:800], X[800:]
train_y, test_y = y[:800], y[800:]

# Create data loaders
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SequenceDataset(train_X, train_y), batch_size=32, shuffle=True)
test_loader = DataLoader(SequenceDataset(test_X, test_y), batch_size=32)

# -----------------------------------------------------------------------------
# STEP 2: Sequence Prediction Model
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SEQUENCE PREDICTION MODEL")
print("=" * 60)

class SequencePredictor(nn.Module):
    """Predict next value at each timestep."""
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out)
        return predictions

model = SequencePredictor(input_size=1, hidden_size=64, num_layers=2)
print(model)

# -----------------------------------------------------------------------------
# STEP 3: Training Loop
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = sum(criterion(model(bx), by).item()
                          for bx, by in test_loader) / len(test_loader)
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss/len(train_loader):.6f} | "
              f"Test Loss: {test_loss:.6f}")

# -----------------------------------------------------------------------------
# STEP 4: Autoregressive Generation (Multi-step Prediction)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("AUTOREGRESSIVE GENERATION")
print("=" * 60)

def generate_sequence(model, seed_sequence, n_steps):
    """
    Generate future values autoregressively.
    Feed each prediction back as input for next step.
    """
    model.eval()
    generated = seed_sequence.clone()

    with torch.no_grad():
        for _ in range(n_steps):
            # Predict next step
            output = model(generated)
            next_val = output[:, -1:, :]  # Last prediction

            # Append to sequence
            generated = torch.cat([generated, next_val], dim=1)

    return generated

# Generate from a seed
seed = test_X[0:1, :10, :]  # First 10 timesteps as seed
generated = generate_sequence(model, seed, n_steps=40)

print(f"Seed length:      {seed.shape[1]}")
print(f"Generated length: {generated.shape[1]}")
print(f"Prediction steps: {generated.shape[1] - seed.shape[1]}")

# -----------------------------------------------------------------------------
# STEP 5: Multi-Step Direct Prediction
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MULTI-STEP DIRECT PREDICTION")
print("=" * 60)

class MultiStepPredictor(nn.Module):
    """Predict multiple future steps directly."""
    def __init__(self, input_size, hidden_size, seq_len, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
        # Predict pred_len steps from final hidden state
        self.fc = nn.Linear(hidden_size, pred_len * input_size)
        self.pred_len = pred_len
        self.input_size = input_size

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        # Use last layer's hidden state
        out = self.fc(h_n[-1])
        # Reshape to (batch, pred_len, input_size)
        out = out.view(-1, self.pred_len, self.input_size)
        return out

# Predict 10 steps into the future from 40 step history
multi_model = MultiStepPredictor(input_size=1, hidden_size=64, seq_len=40, pred_len=10)

# Create dataset for multi-step prediction
# Input: steps 0-39, Target: steps 40-49
multi_X = X[:, :40, :]
multi_y = X[:, 40:50, :]  # Assuming seq_length >= 50
print(f"Multi-step input:  {multi_X.shape}")
print(f"Multi-step target: {multi_y.shape}")

# -----------------------------------------------------------------------------
# STEP 6: Encoder-Decoder for Flexible Horizons
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ENCODER-DECODER ARCHITECTURE")
print("=" * 60)

class Seq2SeqPredictor(nn.Module):
    """
    Encoder-Decoder architecture for sequence prediction.
    Encoder compresses input sequence into context.
    Decoder generates output sequence from context.
    """
    def __init__(self, input_size, hidden_size, output_len):
        super().__init__()
        self.output_len = output_len
        self.hidden_size = hidden_size

        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, 2, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, target_len=None):
        if target_len is None:
            target_len = self.output_len

        batch_size = x.size(0)

        # Encode
        _, (h, c) = self.encoder(x)

        # Decode: start with last input value
        decoder_input = x[:, -1:, :]  # (batch, 1, features)
        outputs = []

        for _ in range(target_len):
            decoder_out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = self.fc(decoder_out)
            outputs.append(pred)
            decoder_input = pred  # Autoregressive

        return torch.cat(outputs, dim=1)

seq2seq = Seq2SeqPredictor(input_size=1, hidden_size=64, output_len=10)
print(seq2seq)

# Test
x = torch.randn(4, 30, 1)
out = seq2seq(x, target_len=15)  # Can specify different target lengths!
print(f"\nInput shape:  {x.shape}")
print(f"Output shape: {out.shape}")

# -----------------------------------------------------------------------------
# STEP 7: Teacher Forcing (Training Trick)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEACHER FORCING")
print("=" * 60)

"""
TEACHER FORCING:
- During training: Feed ground truth as decoder input (not predictions)
- Faster convergence, but can cause exposure bias
- Solution: Scheduled sampling (gradually reduce teacher forcing)

Without teacher forcing:
    decode(decode(decode(x))) -> errors accumulate

With teacher forcing:
    decode(y_0), decode(y_1), decode(y_2) -> stable training
"""

class Seq2SeqWithTeacherForcing(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        batch_size = x.size(0)

        # Encode
        _, (h, c) = self.encoder(x)

        # Determine output length
        if target is not None:
            target_len = target.size(1)
        else:
            target_len = x.size(1)  # Default to input length

        outputs = []
        decoder_input = x[:, -1:, :]  # Start token

        for t in range(target_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = self.fc(out)
            outputs.append(pred)

            # Teacher forcing: use ground truth or prediction
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1, :]
            else:
                decoder_input = pred

        return torch.cat(outputs, dim=1)

# -----------------------------------------------------------------------------
# KEY CONCEPTS FOR WORLD MODELS
# -----------------------------------------------------------------------------
"""
WHY SEQUENCE PREDICTION MATTERS FOR WORLD MODELS:

1. STATE PREDICTION:
   - Given state s_t and action a_t, predict s_{t+1}
   - This is the "transition model" or "dynamics model"

2. IMAGINATION/PLANNING:
   - Generate imagined trajectories without real environment
   - "What if I take action A? What happens next?"

3. AUTOREGRESSIVE ROLLOUTS:
   - Chain predictions to imagine long sequences
   - Essential for model-based reinforcement learning

4. LATENT PREDICTION:
   - Often we predict in latent space (compressed representation)
   - More tractable than predicting raw pixels

ARCHITECTURE PATTERNS:
- RNN/LSTM: Classic, works well for shorter sequences
- Transformer: Better for longer sequences (covered later)
- SSM (State Space Models): Emerging efficient alternative

NEXT: Autoencoders to learn the latent representations we'll predict in.
"""

# -----------------------------------------------------------------------------
# EXERCISE:
# 1. Train the Seq2SeqPredictor on the sine wave data
# 2. Implement scheduled sampling (reduce teacher_forcing_ratio over epochs)
# 3. Add attention to the encoder-decoder model
# 4. Try predicting multiple features (multivariate time series)
# -----------------------------------------------------------------------------
