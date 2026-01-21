# =============================================================================
# LESSON 11: Attention Mechanisms
# =============================================================================
# Attention allows models to focus on relevant parts of the input.
# It's the foundation of Transformers and modern sequence models.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------------------------------------------------------
# THE CONCEPT: Why Attention?
# -----------------------------------------------------------------------------
"""
PROBLEM WITH RNNs:
- Information bottleneck: entire sequence compressed into fixed hidden state
- Long-range dependencies are hard to capture
- Sequential processing: can't parallelize

ATTENTION SOLUTION:
- Let the model "look back" at all previous states
- Compute relevance scores (attention weights)
- Weighted sum of values based on relevance

Key insight: Not all parts of input are equally relevant!
"""

# -----------------------------------------------------------------------------
# STEP 1: Simple Dot-Product Attention
# -----------------------------------------------------------------------------
print("=" * 60)
print("DOT-PRODUCT ATTENTION")
print("=" * 60)

def simple_attention(query, keys, values):
    """
    Simplest form of attention.

    Args:
        query: What we're looking for (d_k,)
        keys: What we match against (seq_len, d_k)
        values: What we retrieve (seq_len, d_v)

    Returns:
        Weighted sum of values
    """
    # Compute similarity scores
    scores = torch.matmul(keys, query)  # (seq_len,)

    # Normalize to get attention weights
    weights = F.softmax(scores, dim=0)  # (seq_len,)

    # Weighted sum of values
    output = torch.matmul(weights, values)  # (d_v,)

    return output, weights

# Example
seq_len, d_k, d_v = 5, 8, 8
query = torch.randn(d_k)
keys = torch.randn(seq_len, d_k)
values = torch.randn(seq_len, d_v)

output, weights = simple_attention(query, keys, values)
print(f"Query shape:   {query.shape}")
print(f"Keys shape:    {keys.shape}")
print(f"Values shape:  {values.shape}")
print(f"Output shape:  {output.shape}")
print(f"Attention weights: {weights.data}")
print(f"Weights sum to: {weights.sum().item():.4f}")

# -----------------------------------------------------------------------------
# STEP 2: Scaled Dot-Product Attention (Transformer Attention)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SCALED DOT-PRODUCT ATTENTION")
print("=" * 60)

"""
The attention mechanism from "Attention Is All You Need"

    Attention(Q, K, V) = softmax(QK^T / √d_k) V

Why scale by √d_k?
- Dot products can get large with high dimensions
- Large values -> softmax saturates -> vanishing gradients
- Scaling keeps variance reasonable
"""

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Scaled dot-product attention.

    Args:
        query: (batch, seq_q, d_k)
        key:   (batch, seq_k, d_k)
        value: (batch, seq_k, d_v)
        mask:  Optional mask for padding/causal attention

    Returns:
        output: (batch, seq_q, d_v)
        attention_weights: (batch, seq_q, seq_k)
    """
    d_k = query.size(-1)

    # Compute attention scores
    # (batch, seq_q, d_k) @ (batch, d_k, seq_k) -> (batch, seq_q, seq_k)
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
print(f"Query/Key/Value: {query.shape}")
print(f"Output:          {output.shape}")
print(f"Attention:       {weights.shape}")

# -----------------------------------------------------------------------------
# STEP 3: Multi-Head Attention
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MULTI-HEAD ATTENTION")
print("=" * 60)

"""
Instead of one attention function:
- Run h attention operations in parallel (different heads)
- Each head can focus on different aspects
- Concatenate and project results

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
where head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. Linear projections
        Q = self.W_q(query)  # (batch, seq, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Reshape for multi-head: (batch, seq, d_model) -> (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Apply attention
        # scores: (batch, num_heads, seq_q, seq_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        # 4. Concatenate heads: (batch, num_heads, seq, d_k) -> (batch, seq, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. Final linear projection
        output = self.W_o(context)

        return output, attention_weights

# Test
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # (batch, seq, d_model)
output, weights = mha(x, x, x)  # Self-attention: Q=K=V=x

print(f"Input:             {x.shape}")
print(f"Output:            {output.shape}")
print(f"Attention weights: {weights.shape}")  # (batch, heads, seq, seq)

# -----------------------------------------------------------------------------
# STEP 4: Self-Attention vs Cross-Attention
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SELF-ATTENTION VS CROSS-ATTENTION")
print("=" * 60)

"""
SELF-ATTENTION:
    Q, K, V all come from the same sequence
    Used in: Encoders, understanding context within a sequence

CROSS-ATTENTION:
    Q from one sequence (decoder), K & V from another (encoder)
    Used in: Decoders attending to encoder output

Example:
    - Self-attention in encoder: understand "The cat sat on the mat"
    - Cross-attention in decoder: when generating translation,
      attend to relevant parts of source sentence
"""

# Self-attention example
encoder_output = torch.randn(2, 10, 512)
self_attn = MultiHeadAttention(512, 8)
out, _ = self_attn(encoder_output, encoder_output, encoder_output)
print(f"Self-attention: Same tensor for Q, K, V")

# Cross-attention example
decoder_state = torch.randn(2, 5, 512)  # Decoder queries
cross_attn = MultiHeadAttention(512, 8)
out, _ = cross_attn(decoder_state, encoder_output, encoder_output)  # Q from decoder, K,V from encoder
print(f"Cross-attention: Q from decoder, K/V from encoder")
print(f"Decoder queries encoder: {decoder_state.shape} attends to {encoder_output.shape}")

# -----------------------------------------------------------------------------
# STEP 5: Causal (Masked) Attention
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CAUSAL ATTENTION (FOR AUTOREGRESSIVE MODELS)")
print("=" * 60)

"""
For autoregressive generation (like language models):
- Position t should only attend to positions 0, 1, ..., t
- NOT to future positions t+1, t+2, ...
- Use a triangular mask
"""

def create_causal_mask(seq_len):
    """Create lower triangular mask for causal attention."""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

seq_len = 5
causal_mask = create_causal_mask(seq_len)
print("Causal mask (lower triangular):")
print(causal_mask)
print("\nPosition 0 can only see position 0")
print("Position 4 can see positions 0, 1, 2, 3, 4")

# Apply to attention
x = torch.randn(1, seq_len, 64)
mha = MultiHeadAttention(64, 4)

# Reshape mask for multi-head attention
mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
output, weights = mha(x, x, x, mask=mask)

print(f"\nAttention weights for head 0 (masked):")
print(weights[0, 0].data)  # Should be lower triangular

# -----------------------------------------------------------------------------
# STEP 6: Attention for Sequence-to-Sequence
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ATTENTION IN SEQ2SEQ (BAHDANAU ATTENTION)")
print("=" * 60)

class BahdanauAttention(nn.Module):
    """
    Additive attention from "Neural Machine Translation by
    Jointly Learning to Align and Translate" (Bahdanau et al.)

    Different from scaled dot-product: uses a learned alignment function.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.W_encoder = nn.Linear(encoder_dim, attention_dim)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs: (batch, src_len, encoder_dim)
            decoder_hidden: (batch, decoder_dim)

        Returns:
            context: (batch, encoder_dim)
            attention_weights: (batch, src_len)
        """
        # Project encoder outputs
        encoder_proj = self.W_encoder(encoder_outputs)  # (batch, src_len, attn_dim)

        # Project decoder hidden state and expand
        decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1)  # (batch, 1, attn_dim)

        # Compute alignment scores
        scores = self.v(torch.tanh(encoder_proj + decoder_proj))  # (batch, src_len, 1)
        scores = scores.squeeze(-1)  # (batch, src_len)

        # Attention weights
        attention_weights = F.softmax(scores, dim=1)

        # Context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (batch, encoder_dim)

        return context, attention_weights

# Test
bahdanau = BahdanauAttention(encoder_dim=256, decoder_dim=256, attention_dim=128)
encoder_out = torch.randn(2, 15, 256)  # 15 source tokens
decoder_h = torch.randn(2, 256)

context, weights = bahdanau(encoder_out, decoder_h)
print(f"Encoder outputs: {encoder_out.shape}")
print(f"Decoder hidden:  {decoder_h.shape}")
print(f"Context vector:  {context.shape}")
print(f"Attention:       {weights.shape}")

# -----------------------------------------------------------------------------
# STEP 7: Positional Encoding
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("POSITIONAL ENCODING")
print("=" * 60)

"""
Attention has no inherent notion of position/order!
Unlike RNNs which process sequentially, attention sees all at once.

Solution: Add positional information to the embeddings.

Sinusoidal encoding (from Transformer paper):
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Not a parameter, but saved with model

    def forward(self, x):
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

pos_encoder = PositionalEncoding(d_model=512, max_len=100)
x = torch.randn(2, 20, 512)
x_with_pos = pos_encoder(x)
print(f"Input: {x.shape}")
print(f"With positional encoding: {x_with_pos.shape}")
print("Positions are now distinguishable!")

# -----------------------------------------------------------------------------
# KEY CONCEPTS FOR WORLD MODELS
# -----------------------------------------------------------------------------
"""
WHY ATTENTION MATTERS FOR WORLD MODELS:

1. SPATIAL ATTENTION:
   - In vision models, attend to relevant parts of the image
   - "What should I focus on?"

2. TEMPORAL ATTENTION:
   - Attend to relevant past states when predicting future
   - Better than RNN's compressed hidden state
   - Can directly access relevant memories

3. TRANSFORMER WORLD MODELS:
   - Replace RNN with Transformer for dynamics modeling
   - Better at long-range dependencies
   - Parallel training (much faster!)

4. MEMORY ATTENTION:
   - Query a memory bank for relevant past experiences
   - Used in memory-augmented networks

EXAMPLES IN WORLD MODELS:
- IRIS: Transformer-based world model
- TransDreamer: Transformer for latent imagination
- GAIA-1: Attention over video frames

NEXT: Full Transformer architecture for world models.
"""

# -----------------------------------------------------------------------------
# EXERCISE:
# 1. Implement "relative positional encoding" (positions relative to each other)
# 2. Add attention to the sequence prediction model from Lesson 8
# 3. Visualize attention weights on a real task
# 4. Implement sliding window attention for long sequences
# 5. Compare compute/memory of MHA with different num_heads
# -----------------------------------------------------------------------------
