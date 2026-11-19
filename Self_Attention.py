import math

import torch
from torch import nn


def self_attention(x, w_q, w_k, w_v):
    "Run single-head self-attention"

    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    # compute attention scores
    scores = q @ k.transpose(-2, -1)
    # scale scores by sqrt(d_k)
    scores = scores / math.sqrt(q.shape[-1])

    # apply softmax to get attention weights
    weights = torch.softmax(scores, dim=-1)

    # compute weighted sum of values
    output = weights @ v

    return output, weights


torch.manual_seed(0)

x = torch.randn(1, 3, 4)  # batch size = 1, sequence length = 3, embedding dimension = 4
w_q = torch.randn(4, 2)  # query weight matrix
w_k = torch.randn(4, 2)  # key weight matrix
w_v = torch.randn(4, 2)  # value weight matrix

output, weights = self_attention(x, w_q, w_k, w_v)

print("output shape:", tuple(output.shape))
print("weights shape:", tuple(weights.shape))
print(
    "row sums:", weights.sum(dim=-1).round(decimals=6).tolist()
)  # should be close to 1

print("output:", output)
print("weights:", weights)
