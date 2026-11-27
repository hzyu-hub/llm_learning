import math

import torch
from torch import nn
from torch.nn import functional as F


def multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads):
    """Project features, split heads, calculate attention, nad merge."""
    batch_size, seq_lenth, d_model = x.shape
    head_dim = d_model // num_heads

    # Project token representations into queries, keys, and values
    q = x @ w_q  # (batch_size, seq_length, d_model)
    k = x @ w_k  # (batch_size, seq_length, d_model)
    v = x @ w_v  # (batch_size, seq_length, d_model)

    # Separete heads:(batch, heads, tokens, head_dim)
    q = q.reshape(batch_size, seq_lenth, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch_size, seq_lenth, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch_size, seq_lenth, num_heads, head_dim).transpose(1, 2)

    # Compute scaled attention scores within each head
    scores = (
        q @ k.transpose(-2, -1) / math.sqrt(head_dim)
    )  # (batch, heads, tokens, tokens)

    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)  # (batch, heads, tokens, tokens)

    # Compute the weighted sum of values
    attn_output = attn_weights @ v  # (batch, heads, tokens, head_dim)

    # Concatenate heads
    combined = attn_output.transpose(1, 2).reshape(
        batch_size, seq_lenth, d_model
    )  # (batch, tokens, d_model)

    # Project back to the original feature space
    output = combined @ w_o  # (batch, tokens, d_model)

    return output, attn_weights


def layer_norm(x, gamma, beta, eps=1e-5):
    """Normalize each token's features, then scale and shift."""
    # Calculate feature mean annd population variance for each token
    mean = x.mean(dim=-1, keepdim=True)  # (batch, tokens, 1)
    variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)  # (batch, tokens, 1)

    # Normalize features and apply learned scale and shift.
    normalized = (x - mean) / torch.sqrt(variance + eps)  # (batch, tokens, d_model)

    output = gamma * normalized + beta  # (batch, tokens, d_model)
    return output


def feed_forward(x, w1, b1, w2, b2):
    """Transform each token independently: linear, GELU, linear."""

    # Expand from d_model to d_ff features
    hidden = x @ w1 + b1  # (batch, tokens, d_ff

    # Apply non-linear activation without chaning shape
    hidden = F.gelu(hidden)  # (batch, tokens, d_ff)

    # Project back to d_model features.
    output = hidden @ w2 + b2  # (batch, tokens, d_model)
    return output


class TransformerBlock(nn.Module):
    """A simplified Post_layerNorm Transformer block."""

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        if d_model <= 0 or num_heads <= 0 or d_ff <= 0:
            raise ValueError("d_model, num_heads, and d_ff must be positive")

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads

        # Initialize attention projections with scaled random weights
        self.w_q = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.w_k = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.w_v = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.w_o = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))

        # Initialize feed-forward projections with scaled random weights
        self.w1 = nn.Parameter(torch.randn(d_model, d_ff) / math.sqrt(d_model))
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.w2 = nn.Parameter(torch.randn(d_ff, d_model) / math.sqrt(d_model))
        self.b2 = nn.Parameter(torch.zeros(d_model))

        # Give each LayerNorm its own learnable scale and shift parameters
        self.gamma1 = nn.Parameter(torch.ones(d_model))
        self.beta1 = nn.Parameter(torch.zeros(d_model))
        self.gamma2 = nn.Parameter(torch.ones(d_model))
        self.beta2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        """Apply attention and FFN with residual addition and LayerNorm."""

        # Let tokens gather information from other tokens.
        attention_output, attention_weights = multi_head_attention(
            x, self.w_q, self.w_k, self.w_v, self.w_o, self.num_heads
        )

        # Add the attention input, then normalize.
        residual_layernorm_output = layer_norm(
            x + attention_output, self.gamma1, self.beta1
        )

        # Transform each contextual token vector independently.
        ffn_output = feed_forward(
            residual_layernorm_output, self.w1, self.b1, self.w2, self.b2
        )

        # Add the FFN input, then normalize.
        output = layer_norm(
            residual_layernorm_output + ffn_output, self.gamma2, self.beta2
        )
        return output, attention_weights


torch.manual_seed(0)
# Example usage
d_model, num_heads, d_ff = 4, 2, 16
max_seq_length = 16

# 1. Convert our sentence into token IDs
sentence = "I love China"
tokens = sentence.split()  # ["I", "love", "China"]
vocab = {"I": 0, "love": 1, "China": 2}
tokens_ids = torch.tensor([[vocab[token] for token in tokens]])  # (1, 3)
seq_length = tokens_ids.shape[1]

# 2.Ceate token and positional embedding tables
token_embedding = nn.Embedding(len(vocab), d_model)
position_embedding = nn.Embedding(max_seq_length, d_model)

# 3. look up each token's feature vector
word_vectors = token_embedding(tokens_ids)  # (1, 3, d_model)

# 4. look up each position's feature vector
position_ids = torch.arange(seq_length).unsqueeze(0)  # (1, 3)
position_vectors = position_embedding(position_ids)  # (1, 3, d_model)

# 5. Combine token identity and position
x = word_vectors + position_vectors  # (1, 3, d_model)

# 6. Create a Transformer block and apply it to the input
transformer_block = TransformerBlock(d_model, num_heads, d_ff)


output, attention_weights = transformer_block(x)

print("tokens:", tokens)
print("input shape:", tuple(x.shape))
print("block output:", output)
print("block output:", attention_weights)
