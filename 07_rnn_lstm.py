# =============================================================================
# LESSON 7: Recurrent Neural Networks (RNN, LSTM, GRU)
# =============================================================================
# Sequence modeling - the foundation for temporal understanding in world models.
# RNNs process sequences step-by-step, maintaining a hidden state.

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# THE CONCEPT: Why Recurrence?
# -----------------------------------------------------------------------------
"""
Feedforward networks: Fixed input size, no memory
Recurrent networks:   Process sequences, maintain hidden state

At each timestep t:
    h_t = f(x_t, h_{t-1})    # New hidden state depends on input AND previous state

This is how we model temporal dependencies!
"""

# -----------------------------------------------------------------------------
# BASIC RNN CELL
# -----------------------------------------------------------------------------
print("=" * 60)
print("BASIC RNN")
print("=" * 60)

# RNN parameters
input_size = 10      # Features per timestep
hidden_size = 20     # Hidden state dimension
num_layers = 1       # Number of stacked RNN layers

# Create RNN
rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

# Input shape: (batch, sequence_length, input_size)
batch_size = 3
seq_length = 5
x = torch.randn(batch_size, seq_length, input_size)

# Initial hidden state: (num_layers, batch, hidden_size)
h0 = torch.zeros(num_layers, batch_size, hidden_size)

# Forward pass
output, h_n = rnn(x, h0)

print(f"Input shape:         {x.shape}")
print(f"Initial hidden:      {h0.shape}")
print(f"Output shape:        {output.shape}")   # All hidden states
print(f"Final hidden state:  {h_n.shape}")      # Last hidden state only

# Output contains hidden state at EACH timestep
# h_n contains ONLY the final hidden state

# -----------------------------------------------------------------------------
# LSTM - Long Short-Term Memory
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("LSTM (Long Short-Term Memory)")
print("=" * 60)

"""
LSTM solves the vanishing gradient problem with:
- Cell state (c): Long-term memory highway
- Hidden state (h): Short-term/working memory
- Gates: Control information flow
    - Forget gate: What to remove from cell state
    - Input gate: What new info to add
    - Output gate: What to output from cell state
"""

lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

# LSTM has both hidden state AND cell state
h0 = torch.zeros(num_layers, batch_size, hidden_size)
c0 = torch.zeros(num_layers, batch_size, hidden_size)

# Forward pass
output, (h_n, c_n) = lstm(x, (h0, c0))

print(f"Input shape:         {x.shape}")
print(f"Output shape:        {output.shape}")
print(f"Final hidden state:  {h_n.shape}")
print(f"Final cell state:    {c_n.shape}")

# -----------------------------------------------------------------------------
# GRU - Gated Recurrent Unit
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("GRU (Gated Recurrent Unit)")
print("=" * 60)

"""
GRU is a simplified LSTM:
- Only hidden state (no cell state)
- Fewer parameters, often similar performance
- Reset gate + Update gate
"""

gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

h0 = torch.zeros(num_layers, batch_size, hidden_size)
output, h_n = gru(x, h0)

print(f"Input shape:         {x.shape}")
print(f"Output shape:        {output.shape}")
print(f"Final hidden state:  {h_n.shape}")

# -----------------------------------------------------------------------------
# BIDIRECTIONAL RNNs
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("BIDIRECTIONAL LSTM")
print("=" * 60)

"""
Bidirectional: Process sequence forwards AND backwards
Useful when you have access to the entire sequence (not real-time)
"""

bi_lstm = nn.LSTM(input_size, hidden_size, num_layers,
                  batch_first=True, bidirectional=True)

# Hidden states are doubled (forward + backward)
h0 = torch.zeros(num_layers * 2, batch_size, hidden_size)
c0 = torch.zeros(num_layers * 2, batch_size, hidden_size)

output, (h_n, c_n) = bi_lstm(x, (h0, c0))

print(f"Input shape:         {x.shape}")
print(f"Output shape:        {output.shape}")  # hidden_size * 2
print(f"Final hidden state:  {h_n.shape}")     # num_layers * 2

# -----------------------------------------------------------------------------
# STACKED (DEEP) RNNs
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STACKED LSTM (Multiple Layers)")
print("=" * 60)

num_layers = 3
stacked_lstm = nn.LSTM(input_size, hidden_size, num_layers,
                       batch_first=True, dropout=0.2)

h0 = torch.zeros(num_layers, batch_size, hidden_size)
c0 = torch.zeros(num_layers, batch_size, hidden_size)

output, (h_n, c_n) = stacked_lstm(x, (h0, c0))

print(f"Number of layers:    {num_layers}")
print(f"Output shape:        {output.shape}")   # Only final layer's outputs
print(f"All hidden states:   {h_n.shape}")      # One per layer

# -----------------------------------------------------------------------------
# BUILDING A SEQUENCE CLASSIFIER
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPLETE LSTM CLASSIFIER")
print("=" * 60)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM forward
        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # Use last hidden state for classification
        # out[:, -1, :] = last timestep's output
        out = self.fc(out[:, -1, :])
        return out

# Create model
model = LSTMClassifier(input_size=10, hidden_size=64, num_layers=2, num_classes=5)
print(model)

# Test
x = torch.randn(8, 20, 10)  # batch=8, seq_len=20, features=10
output = model(x)
print(f"\nInput: {x.shape}")
print(f"Output (logits): {output.shape}")

# -----------------------------------------------------------------------------
# SEQUENCE-TO-SEQUENCE (Many-to-Many)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SEQUENCE-TO-SEQUENCE LSTM")
print("=" * 60)

class Seq2SeqLSTM(nn.Module):
    """Output a prediction at every timestep."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Get outputs at all timesteps
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)

        # Apply linear layer to each timestep
        output = self.fc(lstm_out)  # (batch, seq, output_size)
        return output

model = Seq2SeqLSTM(input_size=10, hidden_size=32, output_size=3)
x = torch.randn(4, 15, 10)  # batch=4, seq=15, features=10
output = model(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")  # Prediction at each timestep!

# -----------------------------------------------------------------------------
# KEY CONCEPTS FOR WORLD MODELS
# -----------------------------------------------------------------------------
"""
WHY RNNs MATTER FOR WORLD MODELS:

1. TEMPORAL MODELING: World models need to understand how states evolve over time

2. HIDDEN STATE AS BELIEF STATE:
   - The hidden state h_t encodes a "belief" about the world
   - It summarizes all past observations

3. PREDICTION: Given current state, predict next state
   - h_{t+1} = f(h_t, action_t)
   - This is the "dynamics model" in world models

4. WORLD MODEL ARCHITECTURE (Ha & Schmidhuber 2018):
   - Vision (V): Encode observations -> latent z
   - Memory (M): LSTM/RNN to model temporal dynamics
   - Controller (C): Use V and M outputs to select actions

NEXT: We'll use RNNs for sequence prediction tasks.
"""

# -----------------------------------------------------------------------------
# EXERCISE:
# 1. Build an LSTM that takes 50 timesteps of sensor data (8 features)
#    and classifies the activity (6 classes)
# 2. Modify Seq2SeqLSTM to use GRU instead of LSTM
# 3. Add layer normalization to the LSTM classifier
# -----------------------------------------------------------------------------
