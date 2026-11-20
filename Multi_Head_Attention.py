import math

import torch
from torch import nn


def multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads):
    """
    Compute multi-head attention.

    Args:
        x: Input tensor of shape (batch_size, seq_length, d_model)
        w_q: Weight matrix for queries of shape (d_model, d_k)
        w_k: Weight matrix for keys of shape (d_model, d_k)
        w_v: Weight matrix for values of shape (d_model, d_v)
        w_o: Weight matrix for output of shape (num_heads * d_v, d_model)
        num_heads: Number of attention heads

    Returns:
        Output tensor of shape (batch_size, seq_length, d_model)
    """
    batch_size, seq_length, d_model = x.size()

    # Determine how many projected features belong to each head
    head_dim = d_model // num_heads

    # project every token into query, key, and value features
    q = x @ w_q  # (batch_size, seq_length, d_k)
    k = x @ w_k  # (batch_size, seq_length, d_k)
    v = x @ w_v  # (batch_size, seq_length, d_v)

    # Split the projected features into multiple heads
    q = q.view(batch_size, seq_length, num_heads, head_dim).transpose(
        1, 2
    )  # (batch_size, num_heads, seq_length, head_dim)
    k = k.view(batch_size, seq_length, num_heads, head_dim).transpose(
        1, 2
    )  # (batch_size, num_heads, seq_length, head_dim)
    v = v.view(batch_size, seq_length, num_heads, head_dim).transpose(
        1, 2
    )  # (batch_size, num_heads, seq_length, head_dim)

    # Compute attention scores
    scores = q @ k.transpose(-2, -1)

    # Scale the scores by the square root of the head dimension
    scores = scores / math.sqrt(head_dim)

    # Apply softmax to get attention weights
    attn_weights = torch.softmax(
        scores, dim=-1
    )  # (batch_size, num_heads, seq_length, seq_length)

    # Compute the weighted sum of values
    attn_output = attn_weights @ v  # (batch_size, num_heads, seq_length, head_dim)

    # Concatenate the heads and project the output
    attn_output = (
        attn_output.transpose(1, 2)
        .contiguous()
        .view(batch_size, seq_length, num_heads * head_dim)
    )  # (batch_size, seq_length, num_heads * head_dim)
    output = attn_output @ w_o  # (batch_size, seq_length, d_model)

    return output, attn_weights


torch.manual_seed(0)

# Example usage
d_model = 4
num_heads = 2

# 1. Tokenize the sentence using a simple world-level tokenizer
sentence = "I love China"
tokens = sentence.split()  # ['I', 'love', 'China']

# 2. Convert the tokens into integer vocabulary IDs.
vocab = {"I": 0, "love": 1, "China": 2}
tokens_ids = torch.tensor([vocab[token] for token in tokens])  # [0, 1, 2]
seq_length = len(tokens_ids)  # 3

# 3. Convert the integer IDs into embeddings.
embedding_dim = d_model
token_embedding = nn.Embedding(len(vocab), embedding_dim)
word_vectors = token_embedding(tokens_ids)

# 4.Look up a trainable position vector for each token ID
position_embedding = nn.Embedding(seq_length, embedding_dim)
position_ids = torch.arange(seq_length).unsqueeze(0)  # (1, seq_length)
position_vectors = position_embedding(position_ids)

# 5. Add token and position vectors to construct the attetion input
x = word_vectors + position_vectors  # (batch_size, seq_length, d_model

# 6. Initialize trainable attention projection matrices
w_q = nn.Parameter(torch.randn(d_model, d_model))
w_k = nn.Parameter(torch.randn(d_model, d_model))
w_v = nn.Parameter(torch.randn(d_model, d_model))
w_o = nn.Parameter(torch.randn(d_model, d_model))

# 7. Compute multi-head attention
output, attn_weights = multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads)

print("tokens:", tokens)
print("tokens_ids:", tokens_ids.tolist())
print("position_ids:", position_ids.tolist())
print("word_vectors:", word_vectors)
print("position_vectors:", position_vectors)
print("attention_output:", output)
print("attention_weights:", attn_weights)
