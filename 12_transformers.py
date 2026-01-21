# =============================================================================
# LESSON 12: Transformers - The Modern Sequence Architecture
# =============================================================================
# Transformers have revolutionized ML. Understanding them is essential for
# modern world models like IRIS, Genie, and GAIA.
#
# Run interactively: python 12_transformers.py
# Run tests only:    python 12_transformers.py --test
# =============================================================================

import sys
sys.path.insert(0, '.')
from utils.lesson_utils import LessonRunner, ExerciseTest, ask_question, show_progress

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Create the lesson runner
lesson = LessonRunner("Lesson 12: Transformers", total_points=10)

# =============================================================================
# SECTION 1: The Transformer Architecture
# =============================================================================
@lesson.section("The Transformer Architecture")
def section_1():
    """
    "ATTENTION IS ALL YOU NEED" (Vaswani et al., 2017)

    Key innovations:
    1. Self-attention instead of recurrence
    2. Parallel processing (no sequential dependency)
    3. Better at long-range dependencies
    4. Scales to massive datasets

    Two main variants:
    - ENCODER-DECODER: For seq2seq (translation, summarization)
    - DECODER-ONLY: For generation (GPT, language models)
    """
    print("The Transformer Architecture")
    print("=" * 50)

    print("""
    TRANSFORMER LAYER STRUCTURE
    ===========================

    Each transformer layer has:
    1. Multi-Head Attention
    2. Add & Norm (residual connection + layer norm)
    3. Feed-Forward Network
    4. Add & Norm


    ENCODER LAYER:
    ==============
    Input x
        |
        v
    [Multi-Head Self-Attention]
        |
        v
    [Add & LayerNorm] <-- residual from x
        |
        v
    [Feed-Forward Network]
        |
        v
    [Add & LayerNorm] <-- residual
        |
        v
    Output


    DECODER LAYER (additional cross-attention):
    ===========================================
    1. Masked Self-Attention (causal)
    2. Add & Norm
    3. Cross-Attention (to encoder output)
    4. Add & Norm
    5. Feed-Forward
    6. Add & Norm
    """)

    print("-" * 50)
    print("WHY TRANSFORMERS DOMINATE")
    print("-" * 50)
    print("""
    vs RNNs:
    - Parallel training (10-100x faster)
    - No vanishing gradients over long distances
    - Direct attention to any position

    vs CNNs:
    - Global context from first layer
    - Better for variable-length sequences
    - More flexible receptive field
    """)

# -----------------------------------------------------------------------------
# QUIZ 1
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: Transformer Basics", points=1)
def quiz_1():
    if '--test' not in sys.argv:
        answer = ask_question(
            "What is a key advantage of Transformers over RNNs?",
            [
                "Transformers use less memory",
                "Transformers can be trained in parallel (not sequential)",
                "Transformers are simpler to implement",
                "Transformers don't need positional encoding"
            ]
        )
        return answer == 1
    return True

# =============================================================================
# SECTION 2: Feed-Forward Network
# =============================================================================
@lesson.section("Feed-Forward Network")
def section_2():
    """
    POSITION-WISE FEED-FORWARD NETWORK
    ==================================

    Applied to each position independently.
    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

    Typically expands dimension by 4x, then projects back.
    """
    print("Feed-Forward Network")
    print("=" * 50)

    class FeedForward(nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            return self.linear2(self.dropout(F.relu(self.linear1(x))))

    # Example
    ff = FeedForward(d_model=512, d_ff=2048)

    print("Feed-Forward structure:")
    print(f"  d_model: 512")
    print(f"  d_ff: 2048 (4x expansion)")
    print(f"  Linear(512 -> 2048) -> ReLU -> Linear(2048 -> 512)")

    x = torch.randn(2, 10, 512)
    out = ff(x)
    print(f"\n  Input:  {x.shape}")
    print(f"  Output: {out.shape}")

    print("\n" + "-" * 50)
    print("WHY EXPAND AND PROJECT?")
    print("-" * 50)
    print("""
    The expansion allows:
    - More expressive transformations
    - Non-linear feature combinations
    - Similar to "wide" layers in MLPs

    Think of it as a 2-layer MLP applied to each position.
    """)

    return FeedForward

# -----------------------------------------------------------------------------
# EXERCISE 1: Build Feed-Forward
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Feed-Forward Network", points=1)
def exercise_1():
    """Build a feed-forward network."""
    test = ExerciseTest("Feed-Forward")

    class FFN(nn.Module):
        def __init__(self, d_model, d_ff):
            super().__init__()
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    ffn = FFN(d_model=64, d_ff=256)
    x = torch.randn(2, 10, 64)
    out = ffn(x)

    test.check_shape(out, (2, 10, 64), "output shape matches input")
    test.check_true(ffn.fc1.out_features == 256, "expansion to d_ff")
    test.check_true(ffn.fc2.out_features == 64, "projection back to d_model")

    return test.run()

# =============================================================================
# SECTION 3: Transformer Encoder Layer
# =============================================================================
@lesson.section("Transformer Encoder Layer")
def section_3():
    """
    ENCODER LAYER
    =============

    1. Multi-head self-attention
    2. Add & Norm
    3. Feed-forward
    4. Add & Norm
    """
    print("Transformer Encoder Layer")
    print("=" * 50)

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, q, k, v, mask=None):
            batch = q.size(0)
            Q = self.W_q(q).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(k).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(v).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = self.dropout(F.softmax(scores, dim=-1))
            out = torch.matmul(attn, V)
            out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
            return self.W_o(out)

    class FeedForward(nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
        def forward(self, x):
            return self.net(x)

    class EncoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
            super().__init__()
            self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
            self.ff = FeedForward(d_model, d_ff, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            # Self-attention with residual
            attn_out = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_out))
            # Feed-forward with residual
            ff_out = self.ff(x)
            x = self.norm2(x + self.dropout(ff_out))
            return x

    layer = EncoderLayer(d_model=512, num_heads=8, d_ff=2048)

    print("Encoder Layer components:")
    print("  1. MultiHeadAttention(d_model=512, heads=8)")
    print("  2. LayerNorm + Residual")
    print("  3. FeedForward(512 -> 2048 -> 512)")
    print("  4. LayerNorm + Residual")

    x = torch.randn(2, 10, 512)
    out = layer(x)
    print(f"\n  Input:  {x.shape}")
    print(f"  Output: {out.shape}")

    return EncoderLayer

# -----------------------------------------------------------------------------
# QUIZ 2
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: Residual Connections", points=1)
def quiz_2():
    if '--test' not in sys.argv:
        answer = ask_question(
            "What is the purpose of residual connections in Transformers?",
            [
                "To reduce memory usage",
                "To allow gradients to flow and enable deeper networks",
                "To speed up inference",
                "To reduce the number of parameters"
            ]
        )
        return answer == 1
    return True

# =============================================================================
# SECTION 4: Decoder-Only Transformer (GPT)
# =============================================================================
@lesson.section("Decoder-Only Transformer (GPT)")
def section_4():
    """
    GPT-STYLE ARCHITECTURE
    ======================

    For autoregressive generation:
    - No encoder, just decoder
    - Causal masking: can only attend to past
    - Used in GPT, LLaMA, and world models
    """
    print("Decoder-Only Transformer (GPT-style)")
    print("=" * 50)

    class GPTBlock(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            batch, seq_len = x.size(0), x.size(1)

            # Pre-norm attention
            h = self.ln1(x)
            Q = self.W_q(h).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(h).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(h).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

            # Causal mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, V)
            out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
            out = self.W_o(out)

            x = x + self.dropout(out)

            # Pre-norm feed-forward
            x = x + self.dropout(self.ff(self.ln2(x)))
            return x

    block = GPTBlock(d_model=256, num_heads=4, d_ff=1024)

    print("GPT Block structure:")
    print("  Pre-LayerNorm (more stable training)")
    print("  Causal self-attention (built-in mask)")
    print("  Residual connection")
    print("  Pre-LayerNorm")
    print("  Feed-forward with GELU activation")
    print("  Residual connection")

    x = torch.randn(2, 10, 256)
    out = block(x)
    print(f"\n  Input:  {x.shape}")
    print(f"  Output: {out.shape}")

    return GPTBlock

# -----------------------------------------------------------------------------
# EXERCISE 2: GPT Block
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Simple GPT Block", points=1)
def exercise_2():
    """Build a simplified GPT block."""
    test = ExerciseTest("GPT Block")

    class SimpleGPTBlock(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            self.attn = nn.Linear(d_model, 3 * d_model)  # Q, K, V together
            self.proj = nn.Linear(d_model, d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)

        def forward(self, x):
            batch, seq = x.size(0), x.size(1)

            # Attention
            h = self.ln1(x)
            qkv = self.attn(h).chunk(3, dim=-1)
            Q, K, V = [t.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2) for t in qkv]

            mask = torch.tril(torch.ones(seq, seq, device=x.device)).unsqueeze(0).unsqueeze(0)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch, seq, self.d_model)
            x = x + self.proj(out)

            # Feed-forward
            x = x + self.ff(self.ln2(x))
            return x

    block = SimpleGPTBlock(d_model=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out = block(x)

    test.check_shape(out, (2, 8, 64), "output shape")
    test.check_true(block.d_k == 16, "d_k = d_model / num_heads")

    return test.run()

# =============================================================================
# SECTION 5: Complete GPT Model
# =============================================================================
@lesson.section("Complete GPT Model")
def section_5():
    """
    FULL GPT MODEL
    ==============

    - Token embeddings
    - Positional embeddings
    - Stack of GPT blocks
    - Final projection to vocabulary
    """
    print("Complete GPT Model")
    print("=" * 50)

    class GPT(nn.Module):
        def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len=512):
            super().__init__()
            self.d_model = d_model
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len, d_model)

            # Simplified blocks
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=4*d_model,
                    batch_first=True,
                ) for _ in range(num_layers)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, x):
            batch, seq = x.shape
            tok_emb = self.token_emb(x)
            pos = torch.arange(seq, device=x.device).unsqueeze(0)
            pos_emb = self.pos_emb(pos)
            x = tok_emb + pos_emb

            # Causal mask
            mask = nn.Transformer.generate_square_subsequent_mask(seq, device=x.device)

            for block in self.blocks:
                x = block(x, src_mask=mask, is_causal=True)

            x = self.ln_f(x)
            return self.head(x)

        @torch.no_grad()
        def generate(self, idx, max_new_tokens, temperature=1.0):
            for _ in range(max_new_tokens):
                logits = self(idx)[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_tok], dim=1)
            return idx

    gpt = GPT(vocab_size=1000, d_model=128, num_heads=4, num_layers=4)
    params = sum(p.numel() for p in gpt.parameters())

    print(f"GPT Configuration:")
    print(f"  Vocab size: 1000")
    print(f"  d_model: 128")
    print(f"  Heads: 4")
    print(f"  Layers: 4")
    print(f"  Parameters: {params:,}")

    # Test forward
    x = torch.randint(0, 1000, (2, 20))
    logits = gpt(x)
    print(f"\n  Input tokens: {x.shape}")
    print(f"  Output logits: {logits.shape}")

    # Test generation
    start = torch.randint(0, 1000, (1, 5))
    generated = gpt.generate(start, max_new_tokens=10)
    print(f"\n  Start: {start.shape}")
    print(f"  Generated: {generated.shape}")

    return GPT

# -----------------------------------------------------------------------------
# QUIZ 3
# -----------------------------------------------------------------------------
@lesson.exercise("Quiz: GPT Generation", points=1)
def quiz_3():
    if '--test' not in sys.argv:
        answer = ask_question(
            "How does GPT generate text autoregressively?",
            [
                "It generates all tokens at once",
                "It predicts one token at a time, feeding it back as input",
                "It uses a separate decoder network",
                "It requires ground truth for each step"
            ]
        )
        return answer == 1
    return True

# =============================================================================
# SECTION 6: PyTorch Built-in Transformer
# =============================================================================
@lesson.section("PyTorch Built-in Transformer")
def section_6():
    """
    PYTORCH TRANSFORMER MODULES
    ===========================

    PyTorch provides optimized implementations.
    """
    print("PyTorch Built-in Transformer")
    print("=" * 50)

    # TransformerEncoderLayer
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        batch_first=True
    )

    x = torch.randn(2, 10, 256)
    out = encoder_layer(x)
    print("nn.TransformerEncoderLayer:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")

    # TransformerEncoder (stack of layers)
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    out = encoder(x)
    print(f"\nnn.TransformerEncoder (6 layers):")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")

    # Full Transformer
    transformer = nn.Transformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        batch_first=True
    )

    src = torch.randn(2, 10, 256)
    tgt = torch.randn(2, 15, 256)
    out = transformer(src, tgt)
    print(f"\nnn.Transformer (full encoder-decoder):")
    print(f"  Source: {src.shape}")
    print(f"  Target: {tgt.shape}")
    print(f"  Output: {out.shape}")

    return encoder_layer

# -----------------------------------------------------------------------------
# EXERCISE 3: PyTorch Transformer
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Use PyTorch Transformer", points=1)
def exercise_3():
    """Use PyTorch's built-in transformer."""
    test = ExerciseTest("PyTorch Transformer")

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=64,
        nhead=4,
        dim_feedforward=256,
        batch_first=True
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    x = torch.randn(4, 8, 64)
    out = encoder(x)

    test.check_shape(out, (4, 8, 64), "encoder output shape")
    test.check_true(
        isinstance(encoder.layers[0], nn.TransformerEncoderLayer),
        "uses TransformerEncoderLayer"
    )

    return test.run()

# =============================================================================
# SECTION 7: Positional Embeddings
# =============================================================================
@lesson.section("Positional Embeddings")
def section_7():
    """
    LEARNED VS SINUSOIDAL POSITIONAL EMBEDDINGS
    ==========================================

    Sinusoidal: Fixed, can extrapolate
    Learned: More flexible, limited to max_len
    """
    print("Positional Embeddings")
    print("=" * 50)

    # Sinusoidal (from Lesson 11)
    class SinusoidalPE(nn.Module):
        def __init__(self, d_model, max_len=512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len).float().unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    # Learned (GPT-style)
    class LearnedPE(nn.Module):
        def __init__(self, d_model, max_len=512):
            super().__init__()
            self.pe = nn.Embedding(max_len, d_model)

        def forward(self, x):
            pos = torch.arange(x.size(1), device=x.device)
            return x + self.pe(pos)

    print("Comparison:")
    print("-" * 50)
    print("SINUSOIDAL:")
    print("  + No extra parameters")
    print("  + Can extrapolate to longer sequences")
    print("  - Less flexible")
    print("\nLEARNED:")
    print("  + More flexible (learned from data)")
    print("  + Often works better in practice")
    print("  - Limited to max_len")
    print("  - More parameters")

    # Demo
    sin_pe = SinusoidalPE(d_model=64, max_len=100)
    learn_pe = LearnedPE(d_model=64, max_len=100)

    x = torch.randn(2, 10, 64)
    print(f"\n  Input: {x.shape}")
    print(f"  Sinusoidal PE output: {sin_pe(x).shape}")
    print(f"  Learned PE output: {learn_pe(x).shape}")

    return SinusoidalPE, LearnedPE

# -----------------------------------------------------------------------------
# EXERCISE 4: Positional Embeddings
# -----------------------------------------------------------------------------
@lesson.exercise("Exercise: Positional Embeddings", points=1)
def exercise_4():
    """Compare positional embedding types."""
    test = ExerciseTest("Positional Embeddings")

    class LearnedPE(nn.Module):
        def __init__(self, d_model, max_len):
            super().__init__()
            self.pe = nn.Embedding(max_len, d_model)

        def forward(self, x):
            pos = torch.arange(x.size(1), device=x.device)
            return x + self.pe(pos)

    pe = LearnedPE(d_model=32, max_len=50)
    x = torch.zeros(2, 10, 32)
    out = pe(x)

    # Different positions should have different encodings
    test.check_true(
        not torch.allclose(out[0, 0], out[0, 1]),
        "different positions have different embeddings"
    )

    # Same position in different batches should be same
    test.check_true(
        torch.allclose(out[0, 0], out[1, 0]),
        "same position same across batch"
    )

    return test.run()

# =============================================================================
# SECTION 8: Transformers for World Models
# =============================================================================
@lesson.section("Transformers for World Models")
def section_8():
    """
    TRANSFORMERS IN WORLD MODELS
    ============================

    Modern world models increasingly use Transformers
    instead of RNNs for dynamics modeling.
    """
    print("Transformers for World Models")
    print("=" * 50)

    print("""
    WHY TRANSFORMERS FOR WORLD MODELS?
    ==================================

    1. PARALLEL TRAINING
       - RNNs: Process steps sequentially (slow!)
       - Transformers: All steps at once (fast!)
       - 10-100x speedup in training

    2. LONG-RANGE DEPENDENCIES
       - Direct attention to distant past states
       - No information bottleneck

    3. SCALABILITY
       - Transformers scale better with data/compute
       - Bigger models = better performance


    TRANSFORMER WORLD MODEL EXAMPLES
    ================================

    IRIS (2023):
    - Discretize latent space (VQ-VAE)
    - Transformer predicts next latent tokens
    - State-of-the-art on Atari

    GENIE (2024):
    - Learns world model from videos
    - Generates playable video game worlds
    - Attention over video frames

    GAIA-1 (2023):
    - World model for autonomous driving
    - Attention over past frames + actions
    - Generates realistic driving videos


    ARCHITECTURE PATTERN
    ====================

    [Observation] -> [VAE Encoder] -> [Discrete Tokens]
                                           |
    [Action] -----> [Transformer] ----> [Next Tokens]
                                           |
                   [VAE Decoder] <- [Predicted Tokens]
                         |
                    [Predicted Frame]
    """)

    print("-" * 50)
    print("SIMPLE WORLD MODEL TRANSFORMER")
    print("-" * 50)

    class WorldModelTransformer(nn.Module):
        """Simplified transformer for world modeling."""
        def __init__(self, state_dim, action_dim, d_model, num_heads, num_layers):
            super().__init__()
            self.state_proj = nn.Linear(state_dim, d_model)
            self.action_proj = nn.Linear(action_dim, d_model)
            self.pos_emb = nn.Embedding(100, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads,
                dim_feedforward=4*d_model, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

            self.output = nn.Linear(d_model, state_dim)

        def forward(self, states, actions):
            """
            states: (batch, seq, state_dim)
            actions: (batch, seq, action_dim)
            """
            batch, seq = states.shape[:2]

            # Project to d_model
            state_emb = self.state_proj(states)
            action_emb = self.action_proj(actions)

            # Interleave: s1, a1, s2, a2, ...
            x = torch.stack([state_emb, action_emb], dim=2).view(batch, 2*seq, -1)

            # Add positional embedding
            pos = torch.arange(2*seq, device=x.device)
            x = x + self.pos_emb(pos)

            # Causal mask
            mask = nn.Transformer.generate_square_subsequent_mask(2*seq, device=x.device)

            # Transform
            x = self.transformer(x, mask=mask, is_causal=True)

            # Predict next states (from action positions)
            return self.output(x[:, 1::2, :])  # Every other position

    model = WorldModelTransformer(
        state_dim=32, action_dim=4,
        d_model=64, num_heads=4, num_layers=2
    )

    states = torch.randn(2, 10, 32)
    actions = torch.randn(2, 10, 4)
    next_states = model(states, actions)

    print(f"Input states:      {states.shape}")
    print(f"Input actions:     {actions.shape}")
    print(f"Predicted states:  {next_states.shape}")

# -----------------------------------------------------------------------------
# FINAL CHALLENGE
# -----------------------------------------------------------------------------
@lesson.exercise("Final Challenge: Mini Transformer", points=2)
def final_challenge():
    """Build a complete mini transformer."""
    test = ExerciseTest("Mini Transformer")

    class MiniTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len=128):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads,
                dim_feedforward=4*d_model, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            seq = x.size(1)
            x = self.tok_emb(x) + self.pos_emb(torch.arange(seq, device=x.device))
            mask = nn.Transformer.generate_square_subsequent_mask(seq, device=x.device)
            x = self.encoder(x, mask=mask, is_causal=True)
            return self.head(self.ln(x))

    # Test
    model = MiniTransformer(vocab_size=100, d_model=32, num_heads=2, num_layers=2)
    x = torch.randint(0, 100, (4, 16))
    logits = model(x)

    test.check_shape(logits, (4, 16, 100), "output logits shape")
    test.check_true(
        hasattr(model, 'tok_emb') and hasattr(model, 'pos_emb'),
        "has token and positional embeddings"
    )
    test.check_true(
        isinstance(model.encoder, nn.TransformerEncoder),
        "uses TransformerEncoder"
    )

    if test.run():
        print("\nYour mini transformer is complete!")
        print("This architecture can be used for language modeling and world models.")
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
        print("LESSON 12 COMPLETE!")
        print("=" * 60)
        print("""
        KEY TAKEAWAYS:

        1. Transformers: Self-attention + Feed-forward + Residuals
           - Parallel training, better long-range dependencies

        2. Encoder layer: Self-attention -> Add&Norm -> FF -> Add&Norm

        3. Decoder-only (GPT): Causal masking for autoregressive generation

        4. Positional embeddings: Sinusoidal or Learned

        5. PyTorch provides nn.TransformerEncoder/Decoder

        6. World models use Transformers for:
           - Faster training (parallel)
           - Better temporal modeling
           - State-of-the-art results (IRIS, GENIE)

        NEXT: Lesson 13 - Reinforcement Learning
        """)

if __name__ == "__main__":
    main()
