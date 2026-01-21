# =============================================================================
# LESSON 7: Recurrent Neural Networks (RNN, LSTM, GRU)
# =============================================================================
# Sequence modeling - the foundation for temporal understanding in world models.
# RNNs process sequences step-by-step, maintaining a hidden state.
#
# Run interactively: python 07_rnn_lstm.py
# Run tests only:    python 07_rnn_lstm.py --test
# =============================================================================

import sys
sys.path.insert(0, '.')
from utils.lesson_utils import LessonRunner, ExerciseTest, ask_question, show_progress

import torch
import torch.nn as nn
import torch.nn.functional as F

# Create the lesson runner
lesson = LessonRunner("Lesson 7: Recurrent Neural Networks", total_points=10)

# =============================================================================
# SECTION 1: Why Recurrence?
# =============================================================================
@lesson.section("Why Recurrence?")
def section_1():
    """
    FEEDFORWARD VS RECURRENT NETWORKS
    ================================

    Feedforward networks: Fixed input size, no memory of previous inputs
    Recurrent networks:   Process sequences, maintain hidden state

    At each timestep t:
        h_t = f(x_t, h_{t-1})    # New hidden state depends on input AND previous state

    This is how we model temporal dependencies!

    ANALOGY: Reading a Sentence
    ---------------------------
    - Feedforward: See all words at once, but no concept of order
    - Recurrent: Read word by word, remember context from earlier words

    "The cat sat on the mat" - Understanding requires knowing word ORDER

    KEY INSIGHT: The hidden state h_t is like a "memory" that summarizes
    everything the network has seen so far in the sequence.
    """
    print("Why do we need recurrent networks?\n")

    # Demonstrate the problem with feedforward networks
    print("Problem: How do we process sequences of VARIABLE length?")
    print("- Video frames (100 frames? 1000 frames?)")
    print("- Sentences (5 words? 50 words?)")
    print("- Time series (daily? hourly? by minute?)")

    print("\nFeedforward approach (bad):")
    print("- Flatten everything into one big vector")
    print("- Fixed maximum sequence length")
    print("- Loses temporal structure")

    print("\nRecurrent approach (good):")
    print("- Process one step at a time")
    print("- Hidden state carries information forward")
    print("- Works with ANY sequence length")

    # Simple demonstration
    print("\n" + "-" * 50)
    print("RECURRENCE DEMO: Accumulating information")
    print("-" * 50)

    sequence = [1, 2, 3, 4, 5]
    hidden_state = 0  # Start with no memory

    print(f"\nSequence: {sequence}")
    print(f"Initial hidden state: {hidden_state}")

    for t, x_t in enumerate(sequence):
        # Simple recurrence: new_hidden = old_hidden + input
        new_hidden = hidden_state + x_t
        print(f"Step {t}: h_{t} = h_{t-1} + x_{t} = {hidden_state} + {x_t} = {new_hidden}")
        hidden_state = new_hidden

    print(f"\nFinal hidden state: {hidden_state}")
    print("The hidden state 'remembers' the sum of all inputs!")

# -----------------------------------------------------------------------------
# QUIZ 1
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: Recurrence Concept", points=1)
def quiz_1():
    if '--test' not in sys.argv:
        answer = ask_question(
            "In an RNN, the hidden state h_t depends on:",
            [
                "Only the current input x_t",
                "Only the previous hidden state h_{t-1}",
                "Both the current input x_t AND previous hidden state h_{t-1}",
                "All previous inputs x_0, x_1, ..., x_{t-1}"
            ]
        )
        return answer == 2  # Both input and previous hidden state
    return True

# =============================================================================
# SECTION 2: Basic RNN
# =============================================================================
@lesson.section("Basic RNN")
def section_2():
    """
    THE VANILLA RNN
    ==============

    Equation:
        h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)

    Where:
    - h_t: hidden state at time t
    - x_t: input at time t
    - W_hh: weights for hidden-to-hidden connection
    - W_xh: weights for input-to-hidden connection
    - tanh: activation function (keeps values in [-1, 1])

    PyTorch makes this easy with nn.RNN!
    """
    print("Creating a basic RNN in PyTorch")
    print("=" * 50)

    # RNN parameters
    input_size = 10      # Features per timestep
    hidden_size = 20     # Hidden state dimension
    num_layers = 1       # Number of stacked RNN layers

    print(f"\nRNN Configuration:")
    print(f"  Input size:   {input_size} features per timestep")
    print(f"  Hidden size:  {hidden_size} dimensions")
    print(f"  Num layers:   {num_layers}")

    # Create RNN
    rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    print(f"\nRNN created: {rnn}")

    # Input shape: (batch, sequence_length, input_size)
    batch_size = 3
    seq_length = 5
    x = torch.randn(batch_size, seq_length, input_size)

    print(f"\n" + "-" * 50)
    print("INPUT AND OUTPUT SHAPES")
    print("-" * 50)

    # Initial hidden state: (num_layers, batch, hidden_size)
    h0 = torch.zeros(num_layers, batch_size, hidden_size)

    print(f"\nInput shape:         {list(x.shape)}")
    print(f"                     [batch, sequence_length, input_features]")
    print(f"\nInitial hidden h0:   {list(h0.shape)}")
    print(f"                     [num_layers, batch, hidden_size]")

    # Forward pass
    output, h_n = rnn(x, h0)

    print(f"\nOutput shape:        {list(output.shape)}")
    print(f"                     [batch, sequence_length, hidden_size]")
    print(f"                     ^ Hidden state at EVERY timestep!")

    print(f"\nFinal hidden h_n:    {list(h_n.shape)}")
    print(f"                     [num_layers, batch, hidden_size]")
    print(f"                     ^ Only the LAST hidden state")

    # Show the relationship
    print("\n" + "-" * 50)
    print("KEY INSIGHT: output[:, -1, :] == h_n.squeeze(0)")
    print("-" * 50)
    last_output = output[:, -1, :]  # Last timestep from output
    final_hidden = h_n.squeeze(0)    # Remove layer dimension

    are_equal = torch.allclose(last_output, final_hidden)
    print(f"Are they equal? {are_equal}")

    return rnn, x, output, h_n

# -----------------------------------------------------------------------------
# EXERCISE 1: RNN Shapes
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: RNN Dimensions", points=1)
def exercise_1():
    """Create an RNN and verify the output shapes."""
    test = ExerciseTest("RNN Dimensions")

    # YOUR TASK: Create an RNN with these specifications:
    # - input_size: 8
    # - hidden_size: 16
    # - num_layers: 1
    # - batch_first: True

    # Create RNN
    rnn = nn.RNN(input_size=8, hidden_size=16, num_layers=1, batch_first=True)

    # Create input: batch=4, sequence=10, features=8
    x = torch.randn(4, 10, 8)
    h0 = torch.zeros(1, 4, 16)  # (num_layers, batch, hidden)

    # Forward pass
    output, h_n = rnn(x, h0)

    # Tests
    test.check_shape(output, (4, 10, 16), "output shape")
    test.check_shape(h_n, (1, 4, 16), "final hidden shape")
    test.check_true(
        torch.allclose(output[:, -1, :], h_n.squeeze(0)),
        "last output equals final hidden"
    )

    return test.run()

# =============================================================================
# SECTION 3: LSTM - Long Short-Term Memory
# =============================================================================
@lesson.section("LSTM - Long Short-Term Memory")
def section_3():
    """
    THE VANISHING GRADIENT PROBLEM
    =============================

    Basic RNNs struggle with long sequences because:
    - Gradients get multiplied at each timestep
    - After many steps, gradients either:
      * Vanish (approach 0) -> can't learn long-term dependencies
      * Explode (become huge) -> unstable training

    LSTM SOLUTION: Gated Memory Cells
    =================================

    LSTM has TWO states:
    1. Cell state (c_t): Long-term memory "highway"
    2. Hidden state (h_t): Short-term/working memory

    THREE gates control information flow:
    1. Forget gate (f_t): What to REMOVE from cell state
    2. Input gate (i_t): What NEW info to ADD
    3. Output gate (o_t): What to OUTPUT from cell state

    This lets gradients flow unchanged through the cell state!
    """
    print("LSTM: Long Short-Term Memory")
    print("=" * 50)

    print("\n" + "-" * 50)
    print("THE THREE GATES")
    print("-" * 50)

    print("""
    FORGET GATE: f_t = σ(W_f · [h_{t-1}, x_t])
        "Should I forget the old cell state?"
        Output: values between 0 (forget) and 1 (keep)

    INPUT GATE: i_t = σ(W_i · [h_{t-1}, x_t])
        "How much new info should I add?"
        Combined with candidate values: c̃_t = tanh(W_c · [h_{t-1}, x_t])

    OUTPUT GATE: o_t = σ(W_o · [h_{t-1}, x_t])
        "What part of cell state should I output?"

    CELL UPDATE:
        c_t = f_t * c_{t-1} + i_t * c̃_t

    HIDDEN OUTPUT:
        h_t = o_t * tanh(c_t)
    """)

    # Create LSTM
    input_size = 10
    hidden_size = 20
    num_layers = 1
    batch_size = 3
    seq_length = 5

    lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    print("-" * 50)
    print("LSTM IN PYTORCH")
    print("-" * 50)

    # LSTM has both hidden state AND cell state
    h0 = torch.zeros(num_layers, batch_size, hidden_size)
    c0 = torch.zeros(num_layers, batch_size, hidden_size)

    x = torch.randn(batch_size, seq_length, input_size)

    print(f"\nInput shape:         {list(x.shape)}")
    print(f"Initial hidden h0:   {list(h0.shape)}")
    print(f"Initial cell c0:     {list(c0.shape)}")

    # Forward pass - note the tuple for states!
    output, (h_n, c_n) = lstm(x, (h0, c0))

    print(f"\nOutput shape:        {list(output.shape)}")
    print(f"Final hidden h_n:    {list(h_n.shape)}")
    print(f"Final cell c_n:      {list(c_n.shape)}")

    print("\n" + "-" * 50)
    print("KEY DIFFERENCE FROM RNN:")
    print("-" * 50)
    print("RNN:  output, h_n = rnn(x, h0)")
    print("LSTM: output, (h_n, c_n) = lstm(x, (h0, c0))")
    print("\nLSTM needs BOTH hidden AND cell state initialized!")

    return lstm, output, h_n, c_n

# -----------------------------------------------------------------------------
# QUIZ 2
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: LSTM Gates", points=1)
def quiz_2():
    if '--test' not in sys.argv:
        answer = ask_question(
            "Which LSTM gate decides what information to REMOVE from the cell state?",
            [
                "Input gate",
                "Forget gate",
                "Output gate",
                "Cell gate"
            ]
        )
        return answer == 1  # Forget gate
    return True

# =============================================================================
# SECTION 4: GRU - Gated Recurrent Unit
# =============================================================================
@lesson.section("GRU - Gated Recurrent Unit")
def section_4():
    """
    GRU: A SIMPLIFIED LSTM
    =====================

    GRU combines the forget and input gates into a single "update gate"
    and has no separate cell state.

    Advantages:
    - Fewer parameters (faster to train)
    - Often similar performance to LSTM
    - Simpler to understand

    TWO gates:
    1. Reset gate (r_t): How much past info to forget
    2. Update gate (z_t): How much to update hidden state

    z_t = σ(W_z · [h_{t-1}, x_t])
    r_t = σ(W_r · [h_{t-1}, x_t])
    h̃_t = tanh(W · [r_t * h_{t-1}, x_t])
    h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
    """
    print("GRU: Gated Recurrent Unit")
    print("=" * 50)

    print("\nGRU vs LSTM comparison:")
    print("-" * 40)
    print(f"{'Feature':<20} {'LSTM':<15} {'GRU':<15}")
    print("-" * 40)
    print(f"{'States':<20} {'2 (h, c)':<15} {'1 (h only)':<15}")
    print(f"{'Gates':<20} {'3':<15} {'2':<15}")
    print(f"{'Parameters':<20} {'More':<15} {'Fewer':<15}")
    print(f"{'Typical Use':<20} {'Long sequences':<15} {'Shorter seq.':<15}")
    print("-" * 40)

    # Create GRU
    input_size = 10
    hidden_size = 20
    num_layers = 1
    batch_size = 3
    seq_length = 5

    gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    print("\nGRU in PyTorch:")
    print("-" * 50)

    h0 = torch.zeros(num_layers, batch_size, hidden_size)
    x = torch.randn(batch_size, seq_length, input_size)

    print(f"Input shape:         {list(x.shape)}")
    print(f"Initial hidden h0:   {list(h0.shape)}")

    # Forward pass - same as RNN (no cell state)
    output, h_n = gru(x, h0)

    print(f"\nOutput shape:        {list(output.shape)}")
    print(f"Final hidden h_n:    {list(h_n.shape)}")

    print("\n" + "-" * 50)
    print("USAGE COMPARISON:")
    print("-" * 50)
    print("RNN:  output, h_n = rnn(x, h0)")
    print("GRU:  output, h_n = gru(x, h0)       # Same as RNN!")
    print("LSTM: output, (h_n, c_n) = lstm(x, (h0, c0))")

    # Parameter count comparison
    rnn = nn.RNN(input_size, hidden_size, num_layers)
    lstm = nn.LSTM(input_size, hidden_size, num_layers)

    rnn_params = sum(p.numel() for p in rnn.parameters())
    gru_params = sum(p.numel() for p in gru.parameters())
    lstm_params = sum(p.numel() for p in lstm.parameters())

    print("\n" + "-" * 50)
    print("PARAMETER COUNT (same config):")
    print("-" * 50)
    print(f"RNN:  {rnn_params:,} parameters")
    print(f"GRU:  {gru_params:,} parameters ({gru_params/rnn_params:.1f}x RNN)")
    print(f"LSTM: {lstm_params:,} parameters ({lstm_params/rnn_params:.1f}x RNN)")

    return gru, output, h_n

# -----------------------------------------------------------------------------
# EXERCISE 2: Compare RNN Types
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: RNN Type Comparison", points=1)
def exercise_2():
    """Create all three RNN types and compare their outputs."""
    test = ExerciseTest("RNN Type Comparison")

    # Configuration
    input_size = 8
    hidden_size = 16
    batch_size = 2
    seq_length = 10

    # Create all three types
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    gru = nn.GRU(input_size, hidden_size, batch_first=True)
    lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    # Same input for all
    x = torch.randn(batch_size, seq_length, input_size)
    h0 = torch.zeros(1, batch_size, hidden_size)
    c0 = torch.zeros(1, batch_size, hidden_size)  # Only for LSTM

    # Forward passes
    rnn_out, rnn_h = rnn(x, h0)
    gru_out, gru_h = gru(x, h0)
    lstm_out, (lstm_h, lstm_c) = lstm(x, (h0, c0))

    # All should have same output shape
    expected_out_shape = (batch_size, seq_length, hidden_size)
    expected_h_shape = (1, batch_size, hidden_size)

    test.check_shape(rnn_out, expected_out_shape, "RNN output shape")
    test.check_shape(gru_out, expected_out_shape, "GRU output shape")
    test.check_shape(lstm_out, expected_out_shape, "LSTM output shape")
    test.check_shape(lstm_c, expected_h_shape, "LSTM cell state shape")

    return test.run()

# =============================================================================
# SECTION 5: Bidirectional and Stacked RNNs
# =============================================================================
@lesson.section("Bidirectional and Stacked RNNs")
def section_5():
    """
    BIDIRECTIONAL RNNs
    ==================

    Process sequence in BOTH directions:
    - Forward: t=0 -> t=T (normal)
    - Backward: t=T -> t=0 (reversed)

    Useful when you have the ENTIRE sequence available
    (not for real-time/streaming applications)

    Examples:
    - Text classification (have whole sentence)
    - Speech recognition (have whole audio)
    - NOT for: real-time translation, live predictions

    STACKED (DEEP) RNNs
    ===================

    Multiple RNN layers stacked on top of each other:
    - Layer 1 processes input, outputs hidden states
    - Layer 2 takes Layer 1's output as input
    - And so on...

    This allows learning hierarchical representations!
    """
    print("Bidirectional and Stacked RNNs")
    print("=" * 50)

    input_size = 10
    hidden_size = 20
    batch_size = 3
    seq_length = 5
    x = torch.randn(batch_size, seq_length, input_size)

    # BIDIRECTIONAL
    print("\n" + "-" * 50)
    print("BIDIRECTIONAL LSTM")
    print("-" * 50)

    bi_lstm = nn.LSTM(input_size, hidden_size, num_layers=1,
                      batch_first=True, bidirectional=True)

    # Hidden states are doubled (forward + backward)
    h0 = torch.zeros(2, batch_size, hidden_size)  # 2 = num_layers * 2 directions
    c0 = torch.zeros(2, batch_size, hidden_size)

    output, (h_n, c_n) = bi_lstm(x, (h0, c0))

    print(f"Input shape:         {list(x.shape)}")
    print(f"Initial h0 shape:    {list(h0.shape)} (2 directions)")
    print(f"Output shape:        {list(output.shape)}")
    print(f"                     hidden_size * 2 = {hidden_size * 2}")
    print(f"Final h_n shape:     {list(h_n.shape)}")
    print("\nOutput contains concatenated [forward_hidden, backward_hidden]")

    # STACKED
    print("\n" + "-" * 50)
    print("STACKED (DEEP) LSTM")
    print("-" * 50)

    num_layers = 3
    stacked_lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)

    h0 = torch.zeros(num_layers, batch_size, hidden_size)
    c0 = torch.zeros(num_layers, batch_size, hidden_size)

    output, (h_n, c_n) = stacked_lstm(x, (h0, c0))

    print(f"Number of layers:    {num_layers}")
    print(f"Input shape:         {list(x.shape)}")
    print(f"Initial h0 shape:    {list(h0.shape)} (one per layer)")
    print(f"Output shape:        {list(output.shape)}")
    print(f"                     ^ Only FINAL layer's outputs!")
    print(f"Final h_n shape:     {list(h_n.shape)}")
    print(f"                     ^ One hidden state per layer")

    print("\n" + "-" * 50)
    print("STACKED + BIDIRECTIONAL")
    print("-" * 50)

    deep_bi_lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.1)

    # num_layers * num_directions
    h0 = torch.zeros(2 * 2, batch_size, hidden_size)
    c0 = torch.zeros(2 * 2, batch_size, hidden_size)

    output, (h_n, c_n) = deep_bi_lstm(x, (h0, c0))

    print(f"2 layers + bidirectional:")
    print(f"  h0 shape: {list(h0.shape)} (layers * directions = 4)")
    print(f"  Output:   {list(output.shape)} (hidden * 2 directions = {hidden_size * 2})")

    return bi_lstm, stacked_lstm, deep_bi_lstm

# -----------------------------------------------------------------------------
# QUIZ 3
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: Bidirectional Output", points=1)
def quiz_3():
    if '--test' not in sys.argv:
        answer = ask_question(
            "A bidirectional LSTM with hidden_size=32 will have output of size:",
            [
                "32 (same as hidden_size)",
                "64 (hidden_size * 2)",
                "16 (hidden_size / 2)",
                "Depends on sequence length"
            ]
        )
        return answer == 1  # 64 (doubled)
    return True

# =============================================================================
# SECTION 6: Building a Sequence Classifier
# =============================================================================
@lesson.section("Building a Sequence Classifier")
def section_6():
    """
    COMPLETE LSTM CLASSIFIER
    ========================

    Use case: Given a sequence, predict ONE label
    - Sentiment analysis: sequence of words -> positive/negative
    - Activity recognition: sequence of sensor data -> activity type
    - Time series classification: stock patterns -> trend direction

    Architecture:
    1. LSTM processes the entire sequence
    2. Take the LAST hidden state (summary of sequence)
    3. Pass through fully-connected layer for classification
    """
    print("Building an LSTM Sequence Classifier")
    print("=" * 50)

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
            device = x.device
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

            # LSTM forward
            out, (h_n, c_n) = self.lstm(x, (h0, c0))

            # Use last hidden state for classification
            # out[:, -1, :] = last timestep's output
            out = self.fc(out[:, -1, :])
            return out

    # Create model
    model = LSTMClassifier(
        input_size=10,    # 10 features per timestep
        hidden_size=64,   # 64-dim hidden state
        num_layers=2,     # 2 stacked LSTM layers
        num_classes=5     # 5 output classes
    )

    print("Model architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Test forward pass
    print("\n" + "-" * 50)
    print("FORWARD PASS TEST")
    print("-" * 50)

    x = torch.randn(8, 20, 10)  # batch=8, seq_len=20, features=10
    output = model(x)

    print(f"Input shape:  {list(x.shape)}")
    print(f"              [batch=8, seq_len=20, features=10]")
    print(f"Output shape: {list(output.shape)}")
    print(f"              [batch=8, num_classes=5]")
    print(f"\nOutput (logits) for first sample:")
    print(f"  {output[0].detach().numpy()}")
    print(f"\nPredicted class: {output[0].argmax().item()}")

    return LSTMClassifier, model

# -----------------------------------------------------------------------------
# EXERCISE 3: Build a GRU Classifier
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: GRU Classifier", points=1)
def exercise_3():
    """Build a GRU-based sequence classifier."""
    test = ExerciseTest("GRU Classifier")

    class GRUClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.hidden_size = hidden_size

            # YOUR TASK: Create a GRU (not LSTM!)
            self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            batch_size = x.size(0)
            h0 = torch.zeros(1, batch_size, self.hidden_size)

            # GRU forward (no cell state!)
            out, h_n = self.gru(x, h0)

            # Use last timestep
            out = self.fc(out[:, -1, :])
            return out

    # Create and test
    model = GRUClassifier(input_size=8, hidden_size=32, num_classes=3)
    x = torch.randn(4, 15, 8)  # batch=4, seq=15, features=8
    output = model(x)

    test.check_shape(output, (4, 3), "GRU classifier output shape")
    test.check_true(
        hasattr(model, 'gru') and isinstance(model.gru, nn.GRU),
        "model uses GRU (not LSTM)"
    )

    return test.run()

# =============================================================================
# SECTION 7: Sequence-to-Sequence Models
# =============================================================================
@lesson.section("Sequence-to-Sequence Models")
def section_7():
    """
    SEQUENCE-TO-SEQUENCE (Many-to-Many)
    ===================================

    Sometimes we need an output at EVERY timestep:
    - Part-of-speech tagging: word -> tag for EACH word
    - Named entity recognition: token -> entity type for EACH token
    - Time series forecasting: predict next value at EACH step

    Instead of using only the last hidden state,
    we apply the output layer to ALL hidden states!
    """
    print("Sequence-to-Sequence LSTM")
    print("=" * 50)

    class Seq2SeqLSTM(nn.Module):
        """Output a prediction at every timestep."""
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

            # Get outputs at ALL timesteps
            lstm_out, _ = self.lstm(x, (h0, c0))  # (batch, seq, hidden)

            # Apply linear layer to EACH timestep
            output = self.fc(lstm_out)  # (batch, seq, output_size)
            return output

    model = Seq2SeqLSTM(input_size=10, hidden_size=32, output_size=3)

    print("Model architecture:")
    print(model)

    print("\n" + "-" * 50)
    print("FORWARD PASS")
    print("-" * 50)

    x = torch.randn(4, 15, 10)  # batch=4, seq=15, features=10
    output = model(x)

    print(f"Input shape:  {list(x.shape)}")
    print(f"              [batch=4, seq_len=15, features=10]")
    print(f"Output shape: {list(output.shape)}")
    print(f"              [batch=4, seq_len=15, output_size=3]")
    print(f"\n^ We get a prediction at EACH of the 15 timesteps!")

    # Comparison
    print("\n" + "-" * 50)
    print("CLASSIFIER vs SEQ2SEQ")
    print("-" * 50)
    print("Classifier: input (batch, seq, feat) -> output (batch, classes)")
    print("            Uses only LAST hidden state")
    print("\nSeq2Seq:    input (batch, seq, feat) -> output (batch, seq, classes)")
    print("            Uses ALL hidden states")

    return Seq2SeqLSTM, model

# -----------------------------------------------------------------------------
# EXERCISE 4: Seq2Seq with GRU
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Seq2Seq GRU", points=1)
def exercise_4():
    """Create a sequence-to-sequence model using GRU."""
    test = ExerciseTest("Seq2Seq GRU")

    class Seq2SeqGRU(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # GRU forward
            gru_out, _ = self.gru(x)
            # Apply fc to all timesteps
            output = self.fc(gru_out)
            return output

    # Test
    model = Seq2SeqGRU(input_size=5, hidden_size=20, output_size=2)
    x = torch.randn(3, 10, 5)  # batch=3, seq=10, features=5
    output = model(x)

    test.check_shape(output, (3, 10, 2), "Seq2Seq output has prediction per timestep")
    test.check_true(
        output.shape[1] == x.shape[1],
        "output sequence length matches input sequence length"
    )

    return test.run()

# =============================================================================
# SECTION 8: RNNs for World Models
# =============================================================================
@lesson.section("RNNs for World Models")
def section_8():
    """
    WHY RNNs MATTER FOR WORLD MODELS
    ================================

    World models learn to predict how the world evolves over time.
    RNNs are perfect for this because:

    1. TEMPORAL MODELING
       World states evolve over time: s_t -> s_{t+1} -> s_{t+2}
       RNNs naturally model these transitions

    2. HIDDEN STATE AS BELIEF STATE
       - The hidden state h_t encodes a "belief" about the world
       - It summarizes ALL past observations
       - Even with partial observations, RNN maintains internal model

    3. PREDICTION / IMAGINATION
       Given current state and action, predict next state:
           h_{t+1} = f(h_t, action_t)
       This is the "dynamics model" - we can "imagine" future states!

    4. WORLD MODEL ARCHITECTURE (Ha & Schmidhuber 2018)
       - Vision (V): Encode observations -> latent z
       - Memory (M): LSTM/RNN to model temporal dynamics
       - Controller (C): Use V and M outputs to select actions
    """
    print("RNNs for World Models")
    print("=" * 50)

    print("""
    WORLD MODEL ARCHITECTURE
    ========================

    ┌─────────────────────────────────────────────────┐
    │                   WORLD MODEL                    │
    │                                                  │
    │   Observation      ┌───────┐                    │
    │   (image) ────────>│   V   │──> latent z        │
    │                    │ (VAE) │                    │
    │                    └───────┘                    │
    │                        │                        │
    │                        v                        │
    │   Previous h ───> ┌───────┐                    │
    │                   │   M   │──> next h, pred z  │
    │   Action ────────>│(LSTM) │                    │
    │                   └───────┘                    │
    │                        │                        │
    │                        v                        │
    │                   ┌───────┐                    │
    │   z, h ─────────> │   C   │──> action          │
    │                   │ (MLP) │                    │
    │                   └───────┘                    │
    └─────────────────────────────────────────────────┘

    The MEMORY (M) component is typically an LSTM/GRU!
    """)

    # Simple world model memory component
    print("-" * 50)
    print("SIMPLE MEMORY COMPONENT")
    print("-" * 50)

    class WorldModelMemory(nn.Module):
        """
        Memory component of a world model.
        Takes latent observation + action, predicts next latent.
        """
        def __init__(self, latent_size, action_size, hidden_size):
            super().__init__()
            self.lstm = nn.LSTMCell(latent_size + action_size, hidden_size)
            self.predict = nn.Linear(hidden_size, latent_size)

        def forward(self, z, action, hidden):
            """
            z: current latent observation
            action: current action
            hidden: (h, c) tuple from previous step
            """
            # Concatenate latent and action
            x = torch.cat([z, action], dim=-1)

            # LSTM step
            h, c = hidden
            h_new, c_new = self.lstm(x, (h, c))

            # Predict next latent
            z_pred = self.predict(h_new)

            return z_pred, (h_new, c_new)

    # Demo
    latent_size = 32
    action_size = 4
    hidden_size = 64
    batch_size = 2

    memory = WorldModelMemory(latent_size, action_size, hidden_size)

    # Simulate one step
    z = torch.randn(batch_size, latent_size)  # Current latent
    action = torch.randn(batch_size, action_size)  # Current action
    h = torch.zeros(batch_size, hidden_size)  # Hidden state
    c = torch.zeros(batch_size, hidden_size)  # Cell state

    z_pred, (h_new, c_new) = memory(z, action, (h, c))

    print(f"Current latent z:    {list(z.shape)}")
    print(f"Current action:      {list(action.shape)}")
    print(f"Previous hidden h:   {list(h.shape)}")
    print(f"Predicted next z:    {list(z_pred.shape)}")
    print(f"New hidden h:        {list(h_new.shape)}")

    print("\nThis memory can 'imagine' future states by:")
    print("1. Take current state z and action a")
    print("2. Predict next state z'")
    print("3. Use z' as input for next prediction")
    print("4. Repeat to imagine trajectories!")

    return WorldModelMemory

# -----------------------------------------------------------------------------
# FINAL CHALLENGE
# -----------------------------------------------------------------------------
@lesson.exercise("Final Challenge: Activity Classifier", points=2)
def final_challenge():
    """
    Build a complete activity classifier!

    Task: Create an LSTM that classifies "activities" from sensor sequences.
    - Input: 50 timesteps of sensor data (8 features: accelerometer, gyroscope)
    - Output: 6 activity classes (walking, running, sitting, standing, etc.)
    """
    test = ExerciseTest("Activity Classifier")

    class ActivityClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            # Specifications:
            # - Input: 8 features per timestep
            # - Hidden: 64 dimensions
            # - Layers: 2 stacked LSTM layers
            # - Output: 6 classes
            # - Use dropout of 0.3

            self.lstm = nn.LSTM(
                input_size=8,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=0.3
            )
            self.fc = nn.Linear(64, 6)

        def forward(self, x):
            # x shape: (batch, 50, 8)
            batch_size = x.size(0)

            # Initialize states
            h0 = torch.zeros(2, batch_size, 64)
            c0 = torch.zeros(2, batch_size, 64)

            # LSTM forward
            out, _ = self.lstm(x, (h0, c0))

            # Use last timestep for classification
            out = self.fc(out[:, -1, :])
            return out

    # Create and test
    model = ActivityClassifier()

    # Test input: batch of 16 samples, 50 timesteps, 8 features
    x = torch.randn(16, 50, 8)
    output = model(x)

    # Verify
    test.check_shape(output, (16, 6), "output shape is (batch, num_classes)")

    # Check model structure
    test.check_true(
        hasattr(model, 'lstm') and model.lstm.input_size == 8,
        "LSTM has correct input size (8)"
    )
    test.check_true(
        model.lstm.hidden_size == 64,
        "LSTM has correct hidden size (64)"
    )
    test.check_true(
        model.lstm.num_layers == 2,
        "LSTM has 2 layers"
    )

    # Check that it produces valid logits
    test.check_true(
        not torch.isnan(output).any(),
        "output contains no NaN values"
    )

    if test.run():
        print("\nExcellent! Your activity classifier is ready!")
        print("In a real scenario, you would:")
        print("1. Collect sensor data from smartphones/wearables")
        print("2. Train on labeled activity sequences")
        print("3. Deploy for real-time activity recognition")
        return True
    return False

# =============================================================================
# MAIN
# =============================================================================
def main():
    """Run the lesson."""
    # Check for test mode
    if '--test' in sys.argv:
        # Run all exercises in test mode
        results = []

        # Run each exercise
        results.append(("Quiz 1", quiz_1()))
        results.append(("Exercise 1", exercise_1()))
        results.append(("Quiz 2", quiz_2()))
        results.append(("Exercise 2", exercise_2()))
        results.append(("Quiz 3", quiz_3()))
        results.append(("Exercise 3", exercise_3()))
        results.append(("Exercise 4", exercise_4()))
        results.append(("Final Challenge", final_challenge()))

        # Summary
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
            quiz_3,
            section_6,
            exercise_3,
            section_7,
            exercise_4,
            section_8,
            final_challenge,
        ])

        # Show progress
        show_progress()

        print("\n" + "=" * 60)
        print("LESSON 7 COMPLETE!")
        print("=" * 60)
        print("""
        KEY TAKEAWAYS:

        1. RNNs process sequences step-by-step with hidden state memory

        2. LSTM solves vanishing gradients with:
           - Cell state (long-term memory)
           - Gates (forget, input, output)

        3. GRU is a simpler alternative with fewer parameters

        4. Bidirectional RNNs see sequences both ways

        5. Stacked RNNs learn hierarchical representations

        6. World models use RNNs as the "memory" component
           to predict how states evolve over time

        NEXT: Lesson 8 - Sequence Prediction
        We'll use RNNs for time series forecasting and
        autoregressive generation!
        """)

if __name__ == "__main__":
    main()
