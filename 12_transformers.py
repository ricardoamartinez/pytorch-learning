# =============================================================================
# LESSON 12: Transformers - The Modern Sequence Architecture
# =============================================================================
# Transformers have revolutionized ML. Understanding them is essential for
# modern world models like IRIS, Genie, and GAIA.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------------------------------------------------------
# THE CONCEPT: Transformer Architecture
# -----------------------------------------------------------------------------
"""
"Attention Is All You Need" (Vaswani et al., 2017)

Key innovations:
1. Self-attention instead of recurrence
2. Parallel processing (no sequential dependency)
3. Better at long-range dependencies
4. Scales to massive datasets

ENCODER: Understands input (bidirectional attention)
DECODER: Generates output (causal attention + cross-attention)

For world models, we often use decoder-only (GPT-style) or
encoder-only (BERT-style) architectures.
"""

# -----------------------------------------------------------------------------
# STEP 1: Feed-Forward Network
# -----------------------------------------------------------------------------
print("=" * 60)
print("TRANSFORMER COMPONENTS")
print("=" * 60)

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Applied independently to each position.
    FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

print("Feed-Forward: Linear -> ReLU -> Linear")
print("Expands to d_ff (usually 4x d_model), then projects back")

# -----------------------------------------------------------------------------
# STEP 2: Layer Normalization
# -----------------------------------------------------------------------------

"""
Layer Norm vs Batch Norm:
- Batch Norm: normalize across batch dimension (used in CNNs)
- Layer Norm: normalize across feature dimension (used in Transformers)

Layer Norm is more stable for variable-length sequences.
"""

# PyTorch provides nn.LayerNorm
layer_norm = nn.LayerNorm(512)
x = torch.randn(2, 10, 512)
x_normed = layer_norm(x)
print(f"\nLayerNorm: {x.shape} -> {x_normed.shape}")

# -----------------------------------------------------------------------------
# STEP 3: Multi-Head Attention (from Lesson 11)
# -----------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context), attn

# -----------------------------------------------------------------------------
# STEP 4: Transformer Encoder Layer
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TRANSFORMER ENCODER LAYER")
print("=" * 60)

class TransformerEncoderLayer(nn.Module):
    """
    Single encoder layer:
    1. Multi-head self-attention
    2. Add & Norm (residual connection + layer norm)
    3. Feed-forward network
    4. Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x

encoder_layer = TransformerEncoderLayer(d_model=512, num_heads=8, d_ff=2048)
print(encoder_layer)

# Test
x = torch.randn(2, 10, 512)
out = encoder_layer(x)
print(f"\nInput:  {x.shape}")
print(f"Output: {out.shape}")

# -----------------------------------------------------------------------------
# STEP 5: Transformer Decoder Layer
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TRANSFORMER DECODER LAYER")
print("=" * 60)

class TransformerDecoderLayer(nn.Module):
    """
    Single decoder layer:
    1. Masked multi-head self-attention (causal)
    2. Add & Norm
    3. Multi-head cross-attention (attend to encoder)
    4. Add & Norm
    5. Feed-forward network
    6. Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # Masked self-attention
        attn_out, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention to encoder output
        attn_out, _ = self.cross_attn(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x

decoder_layer = TransformerDecoderLayer(d_model=512, num_heads=8, d_ff=2048)
print(decoder_layer)

# -----------------------------------------------------------------------------
# STEP 6: Positional Encoding
# -----------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# -----------------------------------------------------------------------------
# STEP 7: Complete Transformer (Encoder-Decoder)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPLETE TRANSFORMER")
print("=" * 60)

class Transformer(nn.Module):
    """Full encoder-decoder transformer for seq2seq tasks."""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Encoder and Decoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask=None):
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, encoder_output, tgt_mask=None, cross_mask=None):
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, cross_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask)
        logits = self.output_proj(decoder_output)
        return logits

# Create transformer
transformer = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048
)

# Count parameters
total_params = sum(p.numel() for p in transformer.parameters())
print(f"Total parameters: {total_params:,}")

# Test
src = torch.randint(0, 10000, (2, 20))  # batch=2, src_len=20
tgt = torch.randint(0, 10000, (2, 15))  # batch=2, tgt_len=15
logits = transformer(src, tgt)
print(f"\nSource:  {src.shape}")
print(f"Target:  {tgt.shape}")
print(f"Logits:  {logits.shape}")

# -----------------------------------------------------------------------------
# STEP 8: Decoder-Only Transformer (GPT-style)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DECODER-ONLY TRANSFORMER (GPT-STYLE)")
print("=" * 60)

"""
For autoregressive generation (language models, world models):
- No encoder, just decoder
- Causal masking: can only attend to past
- Used in GPT, LLaMA, and many world models
"""

class GPTBlock(nn.Module):
    """Single GPT-style transformer block."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Pre-norm architecture (more stable)
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), mask)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    """Decoder-only transformer for autoregressive modeling."""
    def __init__(self, vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights (common practice)
        self.token_embedding.weight = self.head.weight

    def forward(self, x):
        batch, seq_len = x.shape
        device = x.device

        # Embeddings
        tok_emb = self.token_embedding(x)
        pos = torch.arange(0, seq_len, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

gpt = GPT(vocab_size=10000, d_model=256, num_heads=4, num_layers=4)
print(f"GPT parameters: {sum(p.numel() for p in gpt.parameters()):,}")

# Test generation
start_tokens = torch.randint(0, 10000, (1, 5))
generated = gpt.generate(start_tokens, max_new_tokens=10)
print(f"\nStart tokens: {start_tokens.shape}")
print(f"Generated:    {generated.shape}")

# -----------------------------------------------------------------------------
# STEP 9: Using PyTorch's Built-in Transformer
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PYTORCH BUILT-IN TRANSFORMER")
print("=" * 60)

# PyTorch provides optimized implementations
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

x = torch.randn(10, 2, 512)  # (seq, batch, d_model) - note different order!
out = encoder(x)
print(f"nn.TransformerEncoder: {x.shape} -> {out.shape}")

# Full transformer
transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
src = torch.randn(10, 2, 512)
tgt = torch.randn(20, 2, 512)
out = transformer(src, tgt)
print(f"nn.Transformer: src={src.shape}, tgt={tgt.shape} -> {out.shape}")

# -----------------------------------------------------------------------------
# KEY CONCEPTS FOR WORLD MODELS
# -----------------------------------------------------------------------------
"""
WHY TRANSFORMERS FOR WORLD MODELS:

1. PARALLEL TRAINING:
   - RNNs: Sequential, slow
   - Transformers: Parallel, fast (huge speedup)

2. LONG-RANGE DEPENDENCIES:
   - Direct attention to any past state
   - No information bottleneck

3. SCALABILITY:
   - Transformers scale better with data and compute
   - Empirically: larger = better

TRANSFORMER WORLD MODELS:

1. IRIS (2023):
   - Transformer predicts next tokens in latent space
   - Discrete tokens from VQ-VAE
   - State-of-the-art on Atari

2. GENIE (2024):
   - Transformer generates interactive video game worlds
   - Learns actions from unlabeled video

3. GAIA-1 (2023):
   - Transformer world model for autonomous driving
   - Attention over past frames

4. TransDreamer:
   - Replaces RNN with Transformer in Dreamer

ARCHITECTURE PATTERN FOR WORLD MODELS:
    [Observation] -> [Encoder (VAE)] -> [Discrete Tokens]
         |                                    |
         v                                    v
    [Action] -----> [Transformer] -------> [Next Tokens]
                          |
                          v
                    [Decoder] -> [Predicted Frame]

NEXT: Reinforcement Learning basics - where world models are used!
"""

# -----------------------------------------------------------------------------
# EXERCISE:
# 1. Implement FlashAttention-style chunked attention for long sequences
# 2. Add rotary positional encoding (RoPE) to the GPT model
# 3. Train a small GPT on character-level Shakespeare text
# 4. Implement KV-caching for faster inference
# 5. Add gradient checkpointing for memory efficiency
# -----------------------------------------------------------------------------
