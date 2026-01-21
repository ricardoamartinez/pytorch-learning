# =============================================================================
# LESSON 8: Sequence Prediction and Time Series Forecasting
# =============================================================================
# Predicting the future - a core capability needed for world models.
# We'll learn to predict next values in sequences.
#
# Run interactively: python 08_sequence_prediction.py
# Run tests only:    python 08_sequence_prediction.py --test
# =============================================================================

import sys
sys.path.insert(0, '.')
from utils.lesson_utils import LessonRunner, ExerciseTest, ask_question, show_progress

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Create the lesson runner
lesson = LessonRunner("Lesson 8: Sequence Prediction", total_points=10)

# =============================================================================
# SECTION 1: The Concept of Sequence Prediction
# =============================================================================
@lesson.section("The Concept of Sequence Prediction")
def section_1():
    """
    SEQUENCE PREDICTION
    ==================

    Given: x_1, x_2, ..., x_t
    Predict: x_{t+1} (or x_{t+1}, ..., x_{t+k} for multi-step)

    This is fundamental to world models:
    - Predict next observation given history
    - Predict next state given current state + action

    THREE APPROACHES:
    ================

    1. AUTOREGRESSIVE: Predict one step, feed back, repeat
       - Simple and flexible
       - Errors can accumulate

    2. DIRECT: Predict all future steps at once
       - No error accumulation
       - Fixed prediction horizon

    3. SEQ2SEQ: Encoder-decoder for flexible horizons
       - Best of both worlds
       - More complex architecture
    """
    print("Sequence Prediction: Predicting the Future")
    print("=" * 50)

    print("""
    VISUAL: Autoregressive Prediction
    ==================================

    Input sequence:   [x_1, x_2, x_3, x_4, x_5]
                                           ↓
    Model predicts:                       x̂_6
                                           ↓
    Feed back:        [x_2, x_3, x_4, x_5, x̂_6]
                                           ↓
    Model predicts:                       x̂_7
                                           ↓
    And so on...

    This is how we "imagine" future sequences!
    """)

    print("-" * 50)
    print("WHY THIS MATTERS FOR WORLD MODELS")
    print("-" * 50)

    print("""
    World Model Prediction Loop:
    ============================

    1. CURRENT STATE: Where am I now? (latent representation z_t)

    2. ACTION: What do I do? (action a_t)

    3. PREDICTION: What happens next?
       z_{t+1} = f(z_t, a_t)

    4. REPEAT: Use z_{t+1} to predict z_{t+2}, and so on

    This lets an agent "imagine" the consequences of actions
    before actually taking them in the real world!
    """)

    # Simple demonstration
    print("-" * 50)
    print("DEMO: Simple Prediction")
    print("-" * 50)

    # Predict next value in a simple pattern
    sequence = [1, 2, 3, 4, 5]
    print(f"Given sequence: {sequence}")
    print(f"Pattern: each number is previous + 1")
    print(f"Predicted next: {sequence[-1] + 1}")

    # Non-trivial pattern
    fib = [1, 1, 2, 3, 5, 8]
    print(f"\nFibonacci: {fib}")
    print(f"Pattern: each number is sum of previous two")
    print(f"Predicted next: {fib[-1] + fib[-2]}")

# -----------------------------------------------------------------------------
# QUIZ 1
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: Prediction Approaches", points=1)
def quiz_1():
    if '--test' not in sys.argv:
        answer = ask_question(
            "In AUTOREGRESSIVE prediction, what do we do after predicting x̂_{t+1}?",
            [
                "Stop - we only predict one step",
                "Feed x̂_{t+1} back as input to predict x̂_{t+2}",
                "Restart from x_1 with the new prediction added",
                "Predict all remaining steps at once"
            ]
        )
        return answer == 1  # Feed back
    return True

# =============================================================================
# SECTION 2: Synthetic Data Generation
# =============================================================================
@lesson.section("Generating Time Series Data")
def section_2():
    """
    CREATING TRAINING DATA
    =====================

    For sequence prediction, we need:
    - Input: A sequence of values [x_1, ..., x_t]
    - Target: The next values [x_2, ..., x_{t+1}]

    We'll use synthetic sine waves because:
    - Easy to generate
    - Have clear patterns to learn
    - Can control difficulty (noise, frequency)
    """
    print("Generating Synthetic Time Series Data")
    print("=" * 50)

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
    n_samples = 100  # Small for demo
    seq_length = 50
    X, y = generate_sine_wave(n_samples, seq_length)

    print(f"Generated {n_samples} samples")
    print(f"Sequence length: {seq_length}")

    print("\n" + "-" * 50)
    print("INPUT/TARGET RELATIONSHIP")
    print("-" * 50)

    print("\nFor each sample:")
    print("  Input X:  [x_1, x_2, x_3, ..., x_49, x_50]")
    print("  Target y: [x_2, x_3, x_4, ..., x_50, x_51]")
    print("\nNote: Target is shifted by 1 timestep!")
    print("At each position, we predict the NEXT value.")

    # Convert to tensors
    X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # Add feature dimension
    y_tensor = torch.FloatTensor(y).unsqueeze(-1)

    print(f"\nTensor shapes:")
    print(f"  Input X:  {list(X_tensor.shape)}")
    print(f"            [samples, seq_length, features]")
    print(f"  Target y: {list(y_tensor.shape)}")

    # Show example
    print("\n" + "-" * 50)
    print("EXAMPLE DATA POINT")
    print("-" * 50)
    print(f"Input (first 5 values):  {X[0, :5].round(3)}")
    print(f"Target (first 5 values): {y[0, :5].round(3)}")
    print("Notice: target[i] ≈ input[i+1]")

    return X_tensor, y_tensor, generate_sine_wave

# -----------------------------------------------------------------------------
# EXERCISE 1: Dataset Creation
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Create Dataset", points=1)
def exercise_1():
    """Create a PyTorch Dataset for sequence prediction."""
    test = ExerciseTest("Dataset Creation")

    class SequenceDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # Create dummy data
    X = torch.randn(100, 50, 1)
    y = torch.randn(100, 50, 1)

    dataset = SequenceDataset(X, y)

    # Test
    test.check_equal(len(dataset), 100, "dataset length")

    sample_x, sample_y = dataset[0]
    test.check_shape(sample_x, (50, 1), "sample X shape")
    test.check_shape(sample_y, (50, 1), "sample y shape")

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    test.check_shape(batch_x, (16, 50, 1), "batch X shape")

    return test.run()

# =============================================================================
# SECTION 3: Basic Sequence Predictor
# =============================================================================
@lesson.section("Basic Sequence Predictor")
def section_3():
    """
    SEQUENCE PREDICTOR MODEL
    ========================

    Architecture:
    1. LSTM processes the input sequence
    2. At each timestep, predict the next value
    3. Output same length as input (shifted predictions)

    This is "many-to-many" prediction:
    - Input: [x_1, x_2, ..., x_T]
    - Output: [x̂_2, x̂_3, ..., x̂_{T+1}]
    """
    print("Building a Sequence Prediction Model")
    print("=" * 50)

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

    print("Model architecture:")
    print(model)

    print("\n" + "-" * 50)
    print("FORWARD PASS")
    print("-" * 50)

    # Test
    x = torch.randn(8, 50, 1)  # batch=8, seq=50, features=1
    output = model(x)

    print(f"Input shape:  {list(x.shape)}")
    print(f"              [batch, seq_length, features]")
    print(f"Output shape: {list(output.shape)}")
    print(f"              [batch, seq_length, features]")

    print("\nAt each timestep t, output[t] predicts input[t+1]")

    # Loss calculation
    print("\n" + "-" * 50)
    print("TRAINING SETUP")
    print("-" * 50)

    criterion = nn.MSELoss()
    y = torch.randn(8, 50, 1)  # Target
    loss = criterion(output, y)

    print(f"Loss function: MSELoss (for regression)")
    print(f"Example loss: {loss.item():.4f}")

    return SequencePredictor, model

# -----------------------------------------------------------------------------
# QUIZ 2
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: Prediction Loss", points=1)
def quiz_2():
    if '--test' not in sys.argv:
        answer = ask_question(
            "For predicting continuous values (like time series), which loss is most appropriate?",
            [
                "CrossEntropyLoss (for classification)",
                "MSELoss (Mean Squared Error)",
                "NLLLoss (Negative Log Likelihood)",
                "HingeLoss (for SVMs)"
            ]
        )
        return answer == 1  # MSELoss
    return True

# =============================================================================
# SECTION 4: Training the Predictor
# =============================================================================
@lesson.section("Training the Predictor")
def section_4():
    """
    TRAINING LOOP
    =============

    Standard training with one addition:
    - We compare predictions shifted by 1 to targets
    - Loss = MSE between predicted and actual next values
    """
    print("Training the Sequence Predictor")
    print("=" * 50)

    # Generate data
    def generate_sine_wave(n_samples, seq_length):
        X, y = [], []
        for _ in range(n_samples):
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            noise = np.random.normal(0, 0.1, seq_length + 1)
            t = np.linspace(0, 4 * np.pi, seq_length + 1)
            signal = np.sin(freq * t + phase) + noise
            X.append(signal[:-1])
            y.append(signal[1:])
        return np.array(X), np.array(y)

    X, y = generate_sine_wave(500, 50)
    X = torch.FloatTensor(X).unsqueeze(-1)
    y = torch.FloatTensor(y).unsqueeze(-1)

    # Split
    train_X, test_X = X[:400], X[400:]
    train_y, test_y = y[:400], y[400:]

    # Dataset
    class SeqDataset(Dataset):
        def __init__(self, X, y):
            self.X, self.y = X, y
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_loader = DataLoader(SeqDataset(train_X, train_y), batch_size=32, shuffle=True)

    # Model
    class SequencePredictor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.1)
            self.fc = nn.Linear(hidden_size, input_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out)

    model = SequencePredictor(input_size=1, hidden_size=64, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training configuration:")
    print(f"  - Train samples: {len(train_X)}")
    print(f"  - Test samples: {len(test_X)}")
    print(f"  - Batch size: 32")
    print(f"  - Learning rate: 0.001")
    print(f"  - Epochs: 20")

    print("\n" + "-" * 50)
    print("TRAINING PROGRESS")
    print("-" * 50)

    # Training
    num_epochs = 20
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

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(test_X)
                test_loss = criterion(test_pred, test_y).item()
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss/len(train_loader):.6f} | "
                  f"Test Loss: {test_loss:.6f}")

    print("\nTraining complete!")

    return model, test_X, test_y

# -----------------------------------------------------------------------------
# EXERCISE 2: Training Loop
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Implement Training", points=1)
def exercise_2():
    """Complete a training loop for sequence prediction."""
    test = ExerciseTest("Training Loop")

    # Simple model
    class SimplePredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.fc = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out)

    model = SimplePredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Dummy data
    X = torch.sin(torch.linspace(0, 8*np.pi, 100)).view(10, 10, 1)
    y = torch.sin(torch.linspace(0.1, 8*np.pi+0.1, 100)).view(10, 10, 1)

    # YOUR TASK: Complete training loop
    initial_loss = criterion(model(X), y).item()

    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(X), y).item()

    test.check_true(
        final_loss < initial_loss,
        f"loss decreased (initial: {initial_loss:.4f}, final: {final_loss:.4f})"
    )
    test.check_true(
        final_loss < 0.5,
        f"final loss is reasonable ({final_loss:.4f} < 0.5)"
    )

    return test.run()

# =============================================================================
# SECTION 5: Autoregressive Generation
# =============================================================================
@lesson.section("Autoregressive Generation")
def section_5():
    """
    AUTOREGRESSIVE GENERATION
    =========================

    Multi-step prediction by feeding outputs back as inputs:

    1. Start with seed sequence [x_1, ..., x_T]
    2. Predict x̂_{T+1}
    3. Append: [x_1, ..., x_T, x̂_{T+1}]
    4. Predict x̂_{T+2}
    5. Repeat for desired number of steps

    This lets us generate arbitrarily long sequences!
    """
    print("Autoregressive Generation")
    print("=" * 50)

    # Build model
    class SequencePredictor(nn.Module):
        def __init__(self, hidden_size=64):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden_size, 2, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out)

    model = SequencePredictor()

    def generate_sequence(model, seed_sequence, n_steps):
        """
        Generate future values autoregressively.
        Feed each prediction back as input for next step.
        """
        model.eval()
        generated = seed_sequence.clone()

        with torch.no_grad():
            for step in range(n_steps):
                # Predict from current sequence
                output = model(generated)
                next_val = output[:, -1:, :]  # Take last prediction

                # Append to sequence
                generated = torch.cat([generated, next_val], dim=1)

                if step < 3:  # Show first few steps
                    print(f"  Step {step+1}: Generated value = {next_val[0,0,0].item():.4f}")

        return generated

    print("Generation process:")
    print("-" * 50)

    # Create seed
    seed = torch.sin(torch.linspace(0, 2*np.pi, 10)).view(1, 10, 1)
    print(f"Seed sequence length: {seed.shape[1]}")

    # Generate
    print("\nGenerating 5 new values...")
    generated = generate_sequence(model, seed, n_steps=5)

    print(f"\nSeed length:      {seed.shape[1]}")
    print(f"Generated length: {generated.shape[1]}")
    print(f"New values added: {generated.shape[1] - seed.shape[1]}")

    print("\n" + "-" * 50)
    print("KEY INSIGHT")
    print("-" * 50)
    print("""
    Autoregressive generation is how language models work!
    - GPT generates text one token at a time
    - Each new token is fed back as input
    - This is also how world models "imagine" futures
    """)

    return generate_sequence

# -----------------------------------------------------------------------------
# EXERCISE 3: Autoregressive Generator
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Autoregressive Generation", points=1)
def exercise_3():
    """Implement autoregressive generation."""
    test = ExerciseTest("Autoregressive Generation")

    class SimplePredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.fc = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out)

    model = SimplePredictor()

    def generate(model, seed, n_steps):
        """YOUR TASK: Complete this function."""
        model.eval()
        generated = seed.clone()

        with torch.no_grad():
            for _ in range(n_steps):
                # Predict
                output = model(generated)
                # Get last prediction
                next_val = output[:, -1:, :]
                # Append
                generated = torch.cat([generated, next_val], dim=1)

        return generated

    # Test
    seed = torch.randn(1, 5, 1)
    generated = generate(model, seed, n_steps=10)

    test.check_equal(
        generated.shape[1], 15,
        f"generated length is seed + n_steps (5 + 10 = 15)"
    )
    test.check_shape(generated, (1, 15, 1), "output shape")

    return test.run()

# =============================================================================
# SECTION 6: Multi-Step Direct Prediction
# =============================================================================
@lesson.section("Multi-Step Direct Prediction")
def section_6():
    """
    DIRECT MULTI-STEP PREDICTION
    ============================

    Instead of autoregressive (one-at-a-time):
    - Predict ALL future steps at once
    - No error accumulation
    - But fixed prediction horizon

    Architecture:
    1. LSTM encodes input sequence
    2. Final hidden state → FC layer
    3. FC outputs all future values at once
    """
    print("Multi-Step Direct Prediction")
    print("=" * 50)

    class MultiStepPredictor(nn.Module):
        """Predict multiple future steps directly."""
        def __init__(self, input_size, hidden_size, pred_len):
            super().__init__()
            self.pred_len = pred_len
            self.input_size = input_size

            self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
            # Predict pred_len steps from final hidden state
            self.fc = nn.Linear(hidden_size, pred_len * input_size)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            # Use last layer's hidden state
            out = self.fc(h_n[-1])
            # Reshape to (batch, pred_len, input_size)
            out = out.view(-1, self.pred_len, self.input_size)
            return out

    # Create model that predicts 10 steps ahead
    pred_len = 10
    model = MultiStepPredictor(input_size=1, hidden_size=64, pred_len=pred_len)

    print("Model architecture:")
    print(model)

    print("\n" + "-" * 50)
    print("FORWARD PASS")
    print("-" * 50)

    x = torch.randn(4, 40, 1)  # Input: 40 timesteps
    output = model(x)

    print(f"Input shape:  {list(x.shape)}")
    print(f"              40 timesteps of history")
    print(f"Output shape: {list(output.shape)}")
    print(f"              {pred_len} timesteps predicted at once!")

    print("\n" + "-" * 50)
    print("COMPARISON: AUTOREGRESSIVE vs DIRECT")
    print("-" * 50)
    print("""
    AUTOREGRESSIVE:
    + Flexible horizon (any number of steps)
    + Natural for generation tasks
    - Errors accumulate over time
    - Slower (sequential computation)

    DIRECT:
    + No error accumulation
    + Fast (single forward pass)
    - Fixed prediction horizon
    - May struggle with long horizons
    """)

    return MultiStepPredictor, model

# -----------------------------------------------------------------------------
# QUIZ 3
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: Prediction Methods", points=1)
def quiz_3():
    if '--test' not in sys.argv:
        answer = ask_question(
            "Which prediction method has NO error accumulation but a FIXED horizon?",
            [
                "Autoregressive prediction",
                "Direct multi-step prediction",
                "Teacher forcing",
                "Scheduled sampling"
            ]
        )
        return answer == 1  # Direct
    return True

# =============================================================================
# SECTION 7: Encoder-Decoder Architecture
# =============================================================================
@lesson.section("Encoder-Decoder Architecture")
def section_7():
    """
    SEQUENCE-TO-SEQUENCE (Encoder-Decoder)
    =====================================

    The best of both worlds:
    - Encoder: Compresses input into context vector
    - Decoder: Generates output autoregressively

    Benefits:
    - Flexible input AND output lengths
    - Context captures full input sequence
    - Foundation for attention mechanisms

    This is the architecture behind:
    - Machine translation
    - Text summarization
    - Time series forecasting
    """
    print("Encoder-Decoder Architecture")
    print("=" * 50)

    class Seq2SeqPredictor(nn.Module):
        """
        Encoder-Decoder for sequence prediction.
        Encoder compresses input, Decoder generates output.
        """
        def __init__(self, input_size, hidden_size, output_len):
            super().__init__()
            self.output_len = output_len
            self.hidden_size = hidden_size

            # Encoder: Process input sequence
            self.encoder = nn.LSTM(input_size, hidden_size, 2, batch_first=True)

            # Decoder: Generate output sequence
            self.decoder = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
            self.fc = nn.Linear(hidden_size, input_size)

        def forward(self, x, target_len=None):
            if target_len is None:
                target_len = self.output_len

            # ENCODE: Compress input into hidden state
            _, (h, c) = self.encoder(x)

            # DECODE: Generate output autoregressively
            decoder_input = x[:, -1:, :]  # Start with last input value
            outputs = []

            for _ in range(target_len):
                decoder_out, (h, c) = self.decoder(decoder_input, (h, c))
                pred = self.fc(decoder_out)
                outputs.append(pred)
                decoder_input = pred  # Feed prediction back

            return torch.cat(outputs, dim=1)

    model = Seq2SeqPredictor(input_size=1, hidden_size=64, output_len=10)

    print("Model architecture:")
    print(model)

    print("\n" + "-" * 50)
    print("FORWARD PASS")
    print("-" * 50)

    x = torch.randn(4, 30, 1)  # 30 timesteps input
    out = model(x, target_len=15)  # Generate 15 timesteps

    print(f"Input shape:  {list(x.shape)} (30 timesteps)")
    print(f"Output shape: {list(out.shape)} (15 timesteps)")
    print("\nNote: Output length is FLEXIBLE!")

    print("\n" + "-" * 50)
    print("ARCHITECTURE DIAGRAM")
    print("-" * 50)
    print("""
    ┌─────────────────────────────────────────────────┐
    │                 SEQ2SEQ                          │
    │                                                  │
    │  Input sequence     ┌──────────┐                │
    │  [x_1...x_T] ──────>│ ENCODER  │                │
    │                     │  (LSTM)  │                │
    │                     └────┬─────┘                │
    │                          │ (h, c)               │
    │                          v context              │
    │                     ┌──────────┐                │
    │  Last input ───────>│ DECODER  │───> y_1       │
    │                     │  (LSTM)  │                │
    │        y_1 ────────>│          │───> y_2       │
    │        y_2 ────────>│          │───> y_3       │
    │        ...          └──────────┘    ...        │
    └─────────────────────────────────────────────────┘
    """)

    return Seq2SeqPredictor, model

# -----------------------------------------------------------------------------
# EXERCISE 4: Encoder-Decoder Model
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Build Encoder-Decoder", points=1)
def exercise_4():
    """Build a simple encoder-decoder model."""
    test = ExerciseTest("Encoder-Decoder")

    class SimpleSeq2Seq(nn.Module):
        def __init__(self, hidden_size=32):
            super().__init__()
            self.encoder = nn.LSTM(1, hidden_size, batch_first=True)
            self.decoder = nn.LSTM(1, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x, output_len):
            # Encode
            _, (h, c) = self.encoder(x)

            # Decode
            decoder_input = x[:, -1:, :]
            outputs = []

            for _ in range(output_len):
                out, (h, c) = self.decoder(decoder_input, (h, c))
                pred = self.fc(out)
                outputs.append(pred)
                decoder_input = pred

            return torch.cat(outputs, dim=1)

    model = SimpleSeq2Seq()

    # Test with different output lengths
    x = torch.randn(2, 20, 1)

    out_5 = model(x, output_len=5)
    out_10 = model(x, output_len=10)

    test.check_shape(out_5, (2, 5, 1), "output length 5")
    test.check_shape(out_10, (2, 10, 1), "output length 10")
    test.check_true(
        out_5.shape[1] != out_10.shape[1],
        "model supports flexible output lengths"
    )

    return test.run()

# =============================================================================
# SECTION 8: Teacher Forcing
# =============================================================================
@lesson.section("Teacher Forcing")
def section_8():
    """
    TEACHER FORCING
    ===============

    A training technique for sequence-to-sequence models.

    PROBLEM: During autoregressive training, errors accumulate
    - Model makes mistake at step t
    - Mistake is fed back as input
    - Next prediction is based on wrong input
    - Errors compound!

    SOLUTION: Teacher Forcing
    - During training: Feed GROUND TRUTH as input (not predictions)
    - Model learns from correct inputs
    - Much faster convergence

    DRAWBACK: Exposure Bias
    - At test time, model only sees its own predictions
    - Never trained with its own mistakes
    - Can cause problems on long sequences

    SOLUTION TO DRAWBACK: Scheduled Sampling
    - Start with 100% teacher forcing
    - Gradually reduce over training
    - Eventually model trains on own predictions
    """
    print("Teacher Forcing")
    print("=" * 50)

    print("""
    WITHOUT TEACHER FORCING (at training):
    ======================================
    Input:   x_1, x_2, x_3
    Step 1:  Predict ŷ_1 (from x)
    Step 2:  Predict ŷ_2 (from ŷ_1) <- if ŷ_1 wrong, ŷ_2 worse!
    Step 3:  Predict ŷ_3 (from ŷ_2) <- errors accumulate!

    WITH TEACHER FORCING (at training):
    ====================================
    Input:   x_1, x_2, x_3
    Target:  y_1, y_2, y_3
    Step 1:  Predict ŷ_1 (from x)
    Step 2:  Predict ŷ_2 (from y_1) <- use ground truth!
    Step 3:  Predict ŷ_3 (from y_2) <- stable training!
    """)

    class Seq2SeqWithTeacherForcing(nn.Module):
        def __init__(self, hidden_size=32):
            super().__init__()
            self.encoder = nn.LSTM(1, hidden_size, batch_first=True)
            self.decoder = nn.LSTM(1, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x, target=None, teacher_forcing_ratio=0.5):
            # Encode
            _, (h, c) = self.encoder(x)

            # Determine output length
            if target is not None:
                target_len = target.size(1)
            else:
                target_len = x.size(1)

            outputs = []
            decoder_input = x[:, -1:, :]

            for t in range(target_len):
                out, (h, c) = self.decoder(decoder_input, (h, c))
                pred = self.fc(out)
                outputs.append(pred)

                # Teacher forcing decision
                if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    decoder_input = target[:, t:t+1, :]  # Use ground truth
                else:
                    decoder_input = pred  # Use prediction

            return torch.cat(outputs, dim=1)

    print("-" * 50)
    print("SCHEDULED SAMPLING EXAMPLE")
    print("-" * 50)

    print("\nTeacher forcing ratio schedule:")
    for epoch in range(0, 101, 20):
        ratio = max(0.0, 1.0 - epoch / 100)
        print(f"  Epoch {epoch:3d}: teacher_forcing_ratio = {ratio:.2f}")

    print("\nThis gradually transitions from:")
    print("  - Learning from ground truth (stable)")
    print("  - To learning from own predictions (realistic)")

    return Seq2SeqWithTeacherForcing

# -----------------------------------------------------------------------------
# FINAL CHALLENGE
# -----------------------------------------------------------------------------
@lesson.exercise("Final Challenge: Time Series Forecaster", points=2)
def final_challenge():
    """
    Build a complete time series forecasting system!

    Task: Create a model that:
    1. Takes 30 timesteps of history
    2. Predicts the next 10 timesteps
    3. Uses encoder-decoder architecture
    """
    test = ExerciseTest("Time Series Forecaster")

    class TimeSeriesForecaster(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, pred_len=10):
            super().__init__()
            self.pred_len = pred_len

            # Encoder
            self.encoder = nn.LSTM(input_size, hidden_size, 2, batch_first=True)

            # Decoder
            self.decoder = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
            self.fc = nn.Linear(hidden_size, input_size)

        def forward(self, x):
            # Encode history
            _, (h, c) = self.encoder(x)

            # Decode predictions
            decoder_input = x[:, -1:, :]
            predictions = []

            for _ in range(self.pred_len):
                out, (h, c) = self.decoder(decoder_input, (h, c))
                pred = self.fc(out)
                predictions.append(pred)
                decoder_input = pred

            return torch.cat(predictions, dim=1)

    # Create model
    model = TimeSeriesForecaster(input_size=1, hidden_size=64, pred_len=10)

    # Test
    history = torch.randn(8, 30, 1)  # 30 timesteps of history
    forecast = model(history)

    test.check_shape(forecast, (8, 10, 1), "forecast shape (batch=8, pred_len=10, features=1)")

    # Verify it's not just returning the input
    test.check_true(
        not torch.allclose(history[:, -10:, :], forecast),
        "forecast is generated, not copied"
    )

    # Check no NaN
    test.check_true(
        not torch.isnan(forecast).any(),
        "forecast contains no NaN values"
    )

    if test.run():
        print("\nExcellent! Your time series forecaster is complete!")
        print("This architecture can be used for:")
        print("- Weather prediction")
        print("- Stock price forecasting")
        print("- Energy demand prediction")
        print("- World model state prediction!")
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
        print("LESSON 8 COMPLETE!")
        print("=" * 60)
        print("""
        KEY TAKEAWAYS:

        1. Sequence prediction: Given history, predict future

        2. Three approaches:
           - Autoregressive: One step at a time, feed back
           - Direct: Predict all future steps at once
           - Seq2Seq: Encoder-decoder, flexible lengths

        3. Teacher forcing: Use ground truth during training
           - Faster convergence, but exposure bias
           - Use scheduled sampling for best results

        4. This is the foundation of world models:
           - Predict next state from current state + action
           - "Imagine" future trajectories

        NEXT: Lesson 9 - Autoencoders
        Learn to compress data into latent representations!
        """)

if __name__ == "__main__":
    main()
