import math

import torch
from torch.nn import functional as F


def masked_multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads):
    """Run causal multi-head self_attention on decoder token vectors."""
    batch_size, seq_length, d_model = x.shape
    head_dim = d_model // num_heads

    # Project tokens into queries, keys, and values
    q = x @ w_q  # (batch_size, seq_length, d_model)
    k = x @ w_k  # (batch_size, seq_length, d_model)
    v = x @ w_v  # (batch_size, seq_length, d_model)

    # Split heads
    q = q.reshape(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)

    # Compute scaled attention scores
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(
        head_dim
    )  # (batch, heads, tokens, tokens)

    # Mark future positions; keep current and earlier positions visible
    future_mask = torch.triu(
        torch.ones(seq_length, seq_length, dtype=torch.bool, device=x.device),
        diagonal=1,
    )

    # Block future position before softmax
    scores = scores.masked_fill(future_mask, float("-inf"))

    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)  # (batch, heads, tokens, tokens)

    # Compute the weighted sum of value vectors.
    head_output = attn_weights @ v  # (batch, heads, tokens, head_dim)

    # Concatenate heads
    combined = head_output.transpose(1, 2).reshape(
        batch_size, seq_length, d_model
    )  # (batch, tokens, d_model)

    # Project back to the original feature space
    output = combined @ w_o  # (batch, tokens, d_model)

    return output, attn_weights


torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

d_model, num_heads = 4, 2

# Prepare decoder input: each position predicts the next token.
vocab = {"<BOS>": 0, "I": 1, "love": 2, "China": 3}
tokens = ["<BOS>", "I", "love", "China"]

# Calculate token IDs
token_ids = torch.tensor([vocab[token] for token in tokens])
seq_length = token_ids.shape[0]

# Construct token and positional embeddings
token_embedding = torch.nn.Embedding(len(vocab), d_model)
position_embedding = torch.nn.Embedding(seq_length, d_model)

positional_ids = torch.arange(seq_length).unsqueeze(0)  # (1, seq_length)

x = token_embedding(token_ids) + position_embedding(
    positional_ids
)  # (batch, seq_length, d_model)

# Initialize weights for multi-head attention
w_q = torch.nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
w_k = torch.nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
w_v = torch.nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
w_o = torch.nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))

# Run masked multi-head attention
output, attn_weights = masked_multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads)

print("output:", output)
print("attention weights:", attn_weights)
