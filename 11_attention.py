# =============================================================================
# LESSON 11: Attention Mechanisms
# =============================================================================
# Attention allows models to focus on relevant parts of the input.
# It's the foundation of Transformers and modern sequence models.
#
# Run interactively: python 11_attention.py
# Run tests only:    python 11_attention.py --test
# =============================================================================

import sys
sys.path.insert(0, '.')
from utils.lesson_utils import LessonRunner, ExerciseTest, ask_question, show_progress

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Create the lesson runner
lesson = LessonRunner("Lesson 11: Attention Mechanisms", total_points=10)

# =============================================================================
# SECTION 1: Why Attention?
# =============================================================================
@lesson.section("Why Attention?")
def section_1():
    """
    THE PROBLEM WITH RNNs
    =====================

    RNNs have an information bottleneck:
    - Entire sequence compressed into fixed hidden state
    - Long-range dependencies are hard to capture
    - Sequential processing: can't parallelize

    ATTENTION SOLUTION:
    - Let the model "look back" at all previous states
    - Compute relevance scores (attention weights)
    - Weighted sum of values based on relevance

    Key insight: Not all parts of input are equally relevant!
    """
    print("Why Attention?")
    print("=" * 50)

    print("""
    RNN BOTTLENECK PROBLEM
    ======================

    Input: "The cat sat on the mat"

    RNN Processing:
    The -> [h1] -> cat -> [h2] -> sat -> [h3] -> on -> [h4] -> the -> [h5] -> mat -> [h6]

    All information must flow through h6!
    By the time we reach the end, early information may be lost.


    ATTENTION SOLUTION
    ==================

    When processing position t, we can DIRECTLY look at all positions:

    Current position: "mat"
    Attention asks: "How relevant is each word?"

    "The"  -> 0.15 relevance
    "cat"  -> 0.35 relevance  <- Important! What sat on the mat?
    "sat"  -> 0.20 relevance
    "on"   -> 0.10 relevance
    "the"  -> 0.05 relevance
    "mat"  -> 0.15 relevance

    Weighted combination gives context vector!
    """)

    print("-" * 50)
    print("QUERY-KEY-VALUE FRAMEWORK")
    print("-" * 50)

    print("""
    Three components:
    - QUERY (Q): What am I looking for?
    - KEY (K):   What do I match against?
    - VALUE (V): What do I retrieve?

    Process:
    1. Compute similarity: score = Q dot K
    2. Normalize: weights = softmax(score)
    3. Retrieve: output = weights dot V

    Analogy: Library Search
    - Query: "books about cats"
    - Keys: Book titles/descriptions
    - Values: Book contents
    - Attention weights: Relevance of each book
    """)

# -----------------------------------------------------------------------------
# QUIZ 1
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: Attention Concept", points=1)
def quiz_1():
    if '--test' not in sys.argv:
        answer = ask_question(
            "What problem does attention solve compared to RNNs?",
            [
                "Makes models smaller",
                "Allows direct access to all positions, avoiding information bottleneck",
                "Removes the need for training",
                "Only works with images"
            ]
        )
        return answer == 1
    return True

# =============================================================================
# SECTION 2: Dot-Product Attention
# =============================================================================
@lesson.section("Dot-Product Attention")
def section_2():
    """
    SIMPLE DOT-PRODUCT ATTENTION
    ============================

    Attention(Q, K, V) = softmax(QK^T) V

    The dot product measures similarity between query and keys.
    """
    print("Dot-Product Attention")
    print("=" * 50)

    def simple_attention(query, keys, values):
        """
        Simplest form of attention.

        Args:
            query: What we're looking for (d_k,)
            keys: What we match against (seq_len, d_k)
            values: What we retrieve (seq_len, d_v)
        """
        # Compute similarity scores
        scores = torch.matmul(keys, query)  # (seq_len,)

        # Normalize to get attention weights
        weights = F.softmax(scores, dim=0)  # (seq_len,)

        # Weighted sum of values
        output = torch.matmul(weights, values)  # (d_v,)

        return output, weights

    # Example
    seq_len, d_k = 5, 8
    query = torch.randn(d_k)
    keys = torch.randn(seq_len, d_k)
    values = torch.randn(seq_len, d_k)

    output, weights = simple_attention(query, keys, values)

    print("Example:")
    print(f"  Query shape:   {query.shape}")
    print(f"  Keys shape:    {keys.shape}")
    print(f"  Values shape:  {values.shape}")
    print(f"  Output shape:  {output.shape}")
    print(f"  Weights:       {weights.data.tolist()}")
    print(f"  Weights sum:   {weights.sum().item():.4f} (should be 1.0)")

    return simple_attention

# -----------------------------------------------------------------------------
# EXERCISE 1: Simple Attention
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Implement Attention", points=1)
def exercise_1():
    """Implement simple dot-product attention."""
    test = ExerciseTest("Simple Attention")

    def attention(query, keys, values):
        """YOUR TASK: Implement attention."""
        # 1. Compute scores: keys @ query
        scores = torch.matmul(keys, query)
        # 2. Softmax to get weights
        weights = F.softmax(scores, dim=0)
        # 3. Weighted sum of values
        output = torch.matmul(weights, values)
        return output, weights

    # Test
    query = torch.randn(8)
    keys = torch.randn(5, 8)
    values = torch.randn(5, 8)

    output, weights = attention(query, keys, values)

    test.check_shape(output, (8,), "output shape")
    test.check_shape(weights, (5,), "weights shape")
    test.check_true(
        abs(weights.sum().item() - 1.0) < 0.001,
        "weights sum to 1"
    )

    return test.run()

# =============================================================================
# SECTION 3: Scaled Dot-Product Attention
# =============================================================================
@lesson.section("Scaled Dot-Product Attention")
def section_3():
    """
    SCALED DOT-PRODUCT ATTENTION
    ============================

    From "Attention Is All You Need" (Transformer paper)

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Why scale by sqrt(d_k)?
    - Dot products grow large with high dimensions
    - Large values -> softmax saturates -> vanishing gradients
    - Scaling keeps variance reasonable
    """
    print("Scaled Dot-Product Attention")
    print("=" * 50)

    def scaled_dot_product_attention(query, key, value, mask=None):
        """
        Scaled dot-product attention for batched inputs.

        Args:
            query: (batch, seq_q, d_k)
            key:   (batch, seq_k, d_k)
            value: (batch, seq_k, d_v)
            mask:  Optional mask for padding/causal
        """
        d_k = query.size(-1)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

    # Test
    batch_size, seq_len, d_k = 2, 10, 64
    query = torch.randn(batch_size, seq_len, d_k)
    key = torch.randn(batch_size, seq_len, d_k)
    value = torch.randn(batch_size, seq_len, d_k)

    output, weights = scaled_dot_product_attention(query, key, value)

    print("Scaled attention (batched):")
    print(f"  Query/Key/Value: {query.shape}")
    print(f"  Output:          {output.shape}")
    print(f"  Attention:       {weights.shape}")

    print("\n" + "-" * 50)
    print("WHY SCALING MATTERS")
    print("-" * 50)

    print(f"""
    Without scaling (d_k = {d_k}):
    - Dot products have variance ~ d_k = {d_k}
    - Values can be very large/small
    - Softmax saturates (all weight on one element)

    With scaling by sqrt({d_k}) = {math.sqrt(d_k):.2f}:
    - Dot products have variance ~ 1
    - Values stay in reasonable range
    - Softmax gives meaningful probabilities
    """)

    return scaled_dot_product_attention

# -----------------------------------------------------------------------------
# QUIZ 2
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: Scaling", points=1)
def quiz_2():
    if '--test' not in sys.argv:
        answer = ask_question(
            "Why do we scale by sqrt(d_k) in scaled dot-product attention?",
            [
                "To make computation faster",
                "To prevent softmax from saturating with large dimension",
                "To reduce memory usage",
                "It's arbitrary and doesn't matter"
            ]
        )
        return answer == 1
    return True

# =============================================================================
# SECTION 4: Multi-Head Attention
# =============================================================================
@lesson.section("Multi-Head Attention")
def section_4():
    """
    MULTI-HEAD ATTENTION
    ====================

    Instead of one attention function:
    - Run h attention operations in parallel (different heads)
    - Each head can focus on different aspects
    - Concatenate and project results

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
    where head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
    """
    print("Multi-Head Attention")
    print("=" * 50)

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            assert d_model % num_heads == 0

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            # Linear projections
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

        def forward(self, query, key, value, mask=None):
            batch_size = query.size(0)

            # 1. Project Q, K, V
            Q = self.W_q(query)
            K = self.W_k(key)
            V = self.W_v(value)

            # 2. Reshape for multi-head
            # (batch, seq, d_model) -> (batch, heads, seq, d_k)
            Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

            # 3. Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            weights = F.softmax(scores, dim=-1)
            context = torch.matmul(weights, V)

            # 4. Concatenate heads
            context = context.transpose(1, 2).contiguous()
            context = context.view(batch_size, -1, self.d_model)

            # 5. Output projection
            output = self.W_o(context)

            return output, weights

    # Test
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(2, 10, 512)
    output, weights = mha(x, x, x)  # Self-attention

    print("Multi-Head Attention:")
    print(f"  d_model: 512, num_heads: 8")
    print(f"  d_k per head: 512 / 8 = 64")
    print(f"\n  Input:   {x.shape}")
    print(f"  Output:  {output.shape}")
    print(f"  Weights: {weights.shape} (batch, heads, seq, seq)")

    print("\n" + "-" * 50)
    print("WHY MULTIPLE HEADS?")
    print("-" * 50)
    print("""
    Each head can learn different patterns:
    - Head 1: Focus on nearby words
    - Head 2: Focus on syntactic relationships
    - Head 3: Focus on semantic similarity
    - etc.

    Like having multiple "perspectives" on the data!
    """)

    return MultiHeadAttention

# -----------------------------------------------------------------------------
# EXERCISE 2: Multi-Head Attention
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Use Multi-Head Attention", points=1)
def exercise_2():
    """Use multi-head attention for self-attention."""
    test = ExerciseTest("Multi-Head Attention")

    class MHA(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.d_k = d_model // num_heads
            self.num_heads = num_heads
            self.d_model = d_model
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

        def forward(self, x):
            batch = x.size(0)
            Q = self.W_q(x).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            weights = F.softmax(scores, dim=-1)
            context = torch.matmul(weights, V)

            context = context.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
            return self.W_o(context)

    mha = MHA(d_model=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    output = mha(x)

    test.check_shape(output, (2, 8, 64), "output shape matches input")
    test.check_true(
        mha.d_k == 16,
        f"d_k = d_model / num_heads = 64 / 4 = 16"
    )

    return test.run()

# =============================================================================
# SECTION 5: Self-Attention vs Cross-Attention
# =============================================================================
@lesson.section("Self-Attention vs Cross-Attention")
def section_5():
    """
    SELF-ATTENTION vs CROSS-ATTENTION
    =================================

    SELF-ATTENTION:
        Q, K, V all come from the same sequence
        Used in: Encoders, understanding context within a sequence

    CROSS-ATTENTION:
        Q from one sequence, K & V from another
        Used in: Decoders attending to encoder output
    """
    print("Self-Attention vs Cross-Attention")
    print("=" * 50)

    print("""
    SELF-ATTENTION
    ==============
    Input: "The cat sat on the mat"
    Query, Key, Value ALL come from this sentence.

    Each word attends to ALL other words in the same sentence.
    Used to understand relationships within the input.

    Example: What does "it" refer to in "The cat ate because it was hungry"?
    Self-attention learns that "it" attends strongly to "cat".


    CROSS-ATTENTION
    ===============
    Encoder output: [encoded French sentence]
    Decoder state:  [partial English translation]

    Query: from decoder (what English word am I generating?)
    Key/Value: from encoder (what French words are relevant?)

    This is how translation models "look back" at the source.
    """)

    # Demonstration
    class SimpleAttention(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.d_model = d_model

        def forward(self, query, key, value):
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_model)
            weights = F.softmax(scores, dim=-1)
            return torch.matmul(weights, value)

    attn = SimpleAttention(d_model=64)

    print("-" * 50)
    print("DEMONSTRATION")
    print("-" * 50)

    # Self-attention
    x = torch.randn(2, 10, 64)  # Sequence of 10 tokens
    self_out = attn(x, x, x)
    print(f"Self-attention: Q=K=V=x")
    print(f"  Input: {x.shape} -> Output: {self_out.shape}")

    # Cross-attention
    encoder_out = torch.randn(2, 15, 64)  # 15 source tokens
    decoder_state = torch.randn(2, 5, 64)  # 5 target tokens so far
    cross_out = attn(decoder_state, encoder_out, encoder_out)
    print(f"\nCross-attention: Q=decoder, K=V=encoder")
    print(f"  Decoder: {decoder_state.shape}")
    print(f"  Encoder: {encoder_out.shape}")
    print(f"  Output:  {cross_out.shape}")

# -----------------------------------------------------------------------------
# QUIZ 3
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: Attention Types", points=1)
def quiz_3():
    if '--test' not in sys.argv:
        answer = ask_question(
            "In cross-attention, where does the Query come from?",
            [
                "The same sequence as Key and Value",
                "A different sequence (e.g., decoder queries encoder)",
                "It's randomly initialized",
                "There is no Query in cross-attention"
            ]
        )
        return answer == 1
    return True

# =============================================================================
# SECTION 6: Causal (Masked) Attention
# =============================================================================
@lesson.section("Causal (Masked) Attention")
def section_6():
    """
    CAUSAL ATTENTION
    ================

    For autoregressive generation (language models, etc.):
    - Position t should only attend to positions 0, 1, ..., t
    - NOT to future positions t+1, t+2, ...
    - Achieved with a triangular mask
    """
    print("Causal (Masked) Attention")
    print("=" * 50)

    def create_causal_mask(seq_len):
        """Create lower triangular mask."""
        return torch.tril(torch.ones(seq_len, seq_len))

    mask = create_causal_mask(5)
    print("Causal mask (5 positions):")
    print(mask)

    print("""
    How to read this mask:
    - Row = current position (what I'm computing)
    - Column = what I can attend to
    - 1 = can attend, 0 = cannot attend

    Position 0: Can only see position 0 (itself)
    Position 1: Can see positions 0, 1
    Position 2: Can see positions 0, 1, 2
    ...
    Position 4: Can see all positions 0-4
    """)

    print("-" * 50)
    print("WHY CAUSAL MASKING?")
    print("-" * 50)
    print("""
    For GENERATION (predicting next token):
    - We can't look at future tokens (they don't exist yet!)
    - We predict position t using only 0, 1, ..., t-1

    Example: Generating "The cat sat"
    - To predict "cat", we only see "The"
    - To predict "sat", we only see "The cat"
    - We NEVER peek ahead!

    This is how GPT, LLaMA, and other language models work.
    """)

    return create_causal_mask

# -----------------------------------------------------------------------------
# EXERCISE 3: Causal Mask
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Create Causal Mask", points=1)
def exercise_3():
    """Create and apply a causal mask."""
    test = ExerciseTest("Causal Mask")

    def create_causal_mask(seq_len):
        """Create lower triangular mask."""
        return torch.tril(torch.ones(seq_len, seq_len))

    mask = create_causal_mask(4)

    # Check it's lower triangular
    test.check_true(
        mask[0, 1] == 0 and mask[0, 2] == 0,
        "position 0 cannot attend to future"
    )
    test.check_true(
        mask[3, 0] == 1 and mask[3, 1] == 1 and mask[3, 2] == 1 and mask[3, 3] == 1,
        "position 3 can attend to all past positions"
    )
    test.check_true(
        mask.sum() == 4 * 5 / 2,  # Sum of 1+2+3+4 = 10
        "correct number of 1s in mask"
    )

    return test.run()

# =============================================================================
# SECTION 7: Positional Encoding
# =============================================================================
@lesson.section("Positional Encoding")
def section_7():
    """
    POSITIONAL ENCODING
    ===================

    Attention has no inherent notion of position!
    Unlike RNNs which process sequentially, attention sees all at once.

    Solution: Add positional information to embeddings.

    Sinusoidal encoding (from Transformer paper):
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    print("Positional Encoding")
    print("=" * 50)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]

    print("""
    WHY POSITIONAL ENCODING?
    ========================

    Without it, attention treats these the same:
    - "dog bites man"
    - "man bites dog"

    With positional encoding, each position has a unique signature.
    The model learns that position matters!


    SINUSOIDAL ENCODING PROPERTIES:
    ===============================
    1. Different frequencies for different dimensions
    2. Can extrapolate to longer sequences
    3. Relative positions can be computed from encodings
    """)

    pos_enc = PositionalEncoding(d_model=64, max_len=100)
    x = torch.randn(2, 10, 64)
    x_with_pos = pos_enc(x)

    print(f"Input: {x.shape}")
    print(f"With positional encoding: {x_with_pos.shape}")
    print("Shape unchanged - position info is ADDED to embeddings")

    return PositionalEncoding

# -----------------------------------------------------------------------------
# EXERCISE 4: Positional Encoding
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Positional Encoding", points=1)
def exercise_4():
    """Test positional encoding."""
    test = ExerciseTest("Positional Encoding")

    class PosEnc(nn.Module):
        def __init__(self, d_model, max_len=100):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len).float().unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]

    pos_enc = PosEnc(d_model=32, max_len=50)
    x = torch.zeros(2, 10, 32)
    out = pos_enc(x)

    # Since x is zeros, output should be the positional encoding
    test.check_true(
        not torch.allclose(out[0, 0], out[0, 1]),
        "different positions have different encodings"
    )
    test.check_true(
        torch.allclose(out[0], out[1]),
        "same positions in different batches have same encoding"
    )

    return test.run()

# =============================================================================
# SECTION 8: Attention for World Models
# =============================================================================
@lesson.section("Attention for World Models")
def section_8():
    """
    ATTENTION IN WORLD MODELS
    =========================

    Modern world models use attention for:
    1. Spatial attention (focus on relevant image regions)
    2. Temporal attention (attend to relevant past states)
    3. Memory attention (query past experiences)
    """
    print("Attention for World Models")
    print("=" * 50)

    print("""
    APPLICATIONS IN WORLD MODELS
    ============================

    1. SPATIAL ATTENTION
       - "Where should I look in this image?"
       - Attend to relevant objects/regions
       - Example: Focus on the ball when predicting its trajectory

    2. TEMPORAL ATTENTION
       - "Which past frames are relevant now?"
       - Better than RNN's compressed hidden state
       - Can directly access any past moment

    3. TRANSFORMER WORLD MODELS
       - Replace RNN with Transformer for dynamics
       - Parallel training (much faster!)
       - Better long-range dependencies

    REAL EXAMPLES:
    ==============
    - IRIS (2023): Transformer predicts next latent tokens
    - TransDreamer: Transformer replaces RNN in Dreamer
    - GAIA-1: Attention over video frames for driving
    """)

    print("-" * 50)
    print("SIMPLE TEMPORAL ATTENTION")
    print("-" * 50)

    class TemporalAttention(nn.Module):
        """Attend to relevant past states."""
        def __init__(self, state_dim):
            super().__init__()
            self.query = nn.Linear(state_dim, state_dim)
            self.key = nn.Linear(state_dim, state_dim)
            self.value = nn.Linear(state_dim, state_dim)

        def forward(self, current_state, past_states):
            """
            current_state: (batch, state_dim)
            past_states: (batch, num_past, state_dim)
            """
            Q = self.query(current_state).unsqueeze(1)  # (batch, 1, dim)
            K = self.key(past_states)                    # (batch, past, dim)
            V = self.value(past_states)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
            weights = F.softmax(scores, dim=-1)
            context = torch.matmul(weights, V).squeeze(1)  # (batch, dim)

            return context, weights.squeeze(1)

    attn = TemporalAttention(state_dim=64)
    current = torch.randn(2, 64)
    past = torch.randn(2, 10, 64)  # 10 past states

    context, weights = attn(current, past)
    print(f"Current state:   {current.shape}")
    print(f"Past states:     {past.shape}")
    print(f"Context vector:  {context.shape}")
    print(f"Attention weights: {weights.shape}")

# -----------------------------------------------------------------------------
# FINAL CHALLENGE
# -----------------------------------------------------------------------------
@lesson.exercise("Final Challenge: Attention Layer", points=2)
def final_challenge():
    """Build a complete attention layer for sequences."""
    test = ExerciseTest("Attention Layer")

    class AttentionLayer(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            assert d_model % num_heads == 0
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

        def forward(self, x, mask=None):
            batch = x.size(0)

            Q = self.W_q(x).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            weights = F.softmax(scores, dim=-1)
            context = torch.matmul(weights, V)

            context = context.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
            return self.W_o(context), weights

    # Test
    layer = AttentionLayer(d_model=64, num_heads=4)
    x = torch.randn(2, 10, 64)
    output, weights = layer(x)

    test.check_shape(output, (2, 10, 64), "output shape")
    test.check_shape(weights, (2, 4, 10, 10), "weights shape (batch, heads, seq, seq)")

    # Test with causal mask
    mask = torch.tril(torch.ones(10, 10)).unsqueeze(0).unsqueeze(0)
    output_masked, weights_masked = layer(x, mask)
    test.check_shape(output_masked, (2, 10, 64), "masked output shape")

    if test.run():
        print("\nYour attention layer is complete!")
        print("This is the core component of Transformers.")
        return True
    return False

# =============================================================================
# MAIN
# =============================================================================
def main():
    if '--test' in sys.argv:
        results = []
        results.append(("Quiz 1", quiz_1()))
        results.append(("Exercise 1", exercise_1()))
        results.append(("Quiz 2", quiz_2()))
        results.append(("Exercise 2", exercise_2()))
        results.append(("Quiz 3", quiz_3()))
        results.append(("Exercise 3", exercise_3()))
        results.append(("Exercise 4", exercise_4()))
        results.append(("Final Challenge", final_challenge()))

        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        passed = sum(1 for _, r in results if r)
        for name, result in results:
            status = "PASS" if result else "FAIL"
            print(f"  {status}: {name}")
        print(f"\nTotal: {passed}/{len(results)} tests passed")
        return passed == len(results)
    else:
        lesson.run_interactive([
            section_1, quiz_1,
            section_2, exercise_1,
            section_3, quiz_2,
            section_4, exercise_2,
            section_5, quiz_3,
            section_6, exercise_3,
            section_7, exercise_4,
            section_8, final_challenge,
        ])
        show_progress()
        print("\n" + "=" * 60)
        print("LESSON 11 COMPLETE!")
        print("=" * 60)
        print("""
        KEY TAKEAWAYS:

        1. Attention allows direct access to all positions
           - Solves RNN information bottleneck

        2. Scaled dot-product attention: softmax(QK^T/sqrt(d_k))V
           - Scaling prevents softmax saturation

        3. Multi-head attention: multiple perspectives
           - Each head learns different patterns

        4. Self vs Cross attention:
           - Self: Q, K, V from same sequence
           - Cross: Q from decoder, K/V from encoder

        5. Causal masking for autoregressive models
           - Lower triangular mask prevents seeing future

        6. Positional encoding adds position information
           - Attention has no inherent notion of order

        NEXT: Lesson 12 - Transformers
        """)

if __name__ == "__main__":
    main()
